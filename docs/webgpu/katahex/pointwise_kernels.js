export const POINTWISE_ROWS_FOR_WIDE_TILE = 2048

export function pointwiseTiles(half, transformer) {
  if (!transformer) return [{ points: 32, columns: 32, rows: 4 }]
  return [
    { points: 32, columns: 32, rows: 2 },
    { points: 64, columns: half ? 64 : 32, rows: half ? 8 : 4 },
  ]
}

export function createPointwiseKernels(half, tiles, transformer, rotateQKV = false) {
  const modes = transformer ? ["pointwise", "pointwiseInPlace", "pointwiseSwiGLU"] : ["pointwise", "pointwiseWithSkip"]
  if (rotateQKV) modes.push("pointwiseRoPE")
  return Object.fromEntries(
    tiles.flatMap((tile, index) => modes.map((mode) => [mode + index, pointwiseKernel(half, tile, mode)])),
  )
}

function pointwiseKernel(half, { points, columns, rows }, mode) {
  const rope = mode === "pointwiseRoPE"
  const swiglu = mode === "pointwiseSwiGLU"
  const separateResidual = mode === "pointwiseWithSkip"
  const residual = separateResidual || mode === "pointwiseInPlace"
  const columnVectors = columns / 4
  const threads = columnVectors * points / rows
  const inputVectors = points * 8
  const weightVectors = 32 * columnVectors
  const repeat = (length, emit) => Array.from({ length }, (_, i) => emit(i)).join("\n")
  const accumulate = (row, sum, weight) => repeat(4, (k) => `      ${sum}${row} += input${row}[${k}u] * ${weight}${k};`)
  // Round the projection to tensor precision before its epilogue.
  const store = (row) =>
    swiglu
      ? `let gate = vec4<f32>(Vector(sum${row}));
      let value = vec4<f32>(Vector(valueSum${row}));
      output[offset] = Vector(gate / (1.0 + exp(-gate)) * value);`
      : residual
        ? `output[offset] = Vector(vec4<f32>(Vector(sum${row})) + vec4<f32>(${separateResidual ? "skip" : "output"}[offset]));`
        : rope
          ? `output[offset] = rotate(Vector(sum${row}), col, row + ${row}u);`
          : `output[offset] = Vector(sum${row});`

  return `${half ? "enable f16;" : ""}
alias Vector = vec4<${half ? "f16" : "f32"}>;
@group(0) @binding(0) var<storage, read> input: array<Vector>;
@group(0) @binding(1) var<storage, read> weights: array<Vector>;
@group(0) @binding(2) var<storage, read_write> output: array<Vector>;
${separateResidual ? "@group(0) @binding(3) var<storage, read> skip: array<Vector>;" : ""}
${rope ? "@group(0) @binding(3) var<storage, read> phases: array<vec2<f32>>;" : ""}
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(${separateResidual || rope ? 4 : 3}) var<uniform> params: Params;
var<workgroup> tileInput: array<Vector, ${inputVectors}>;
var<workgroup> tileWeight: array<Vector, ${weightVectors}>;
${swiglu ? `var<workgroup> tileValue: array<Vector, ${weightVectors}>;` : ""}
${rope ? `
fn rotate(projected: Vector, column: u32, index: u32) -> Vector {
  let points = params.a.x;
  let heads = params.b.x;
  let span = heads * 8u;
  if (column >= span * 2u) { return projected; }
  let first = (index % points) * (span * 2u) + (column % span) * 2u;
  let phase0 = phases[first];
  let phase1 = phases[first + 1u];
  let xy = vec4<f32>(projected);
  return Vector(vec4<f32>(
    xy.xy * phase0.x + vec2<f32>(-xy.y, xy.x) * phase0.y,
    xy.zw * phase1.x + vec2<f32>(-xy.w, xy.z) * phase1.y
  ));
}
` : ""}
@compute @workgroup_size(${columnVectors}, ${points / rows})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let inputChannels = params.a.y / 4u;
  let outputChannels = params.a.z / 4u;
  let rowLimit = params.a.x * params.a.w;
  let row = group.y * ${points}u + local.y * ${rows}u;
  let col = group.x * ${columnVectors}u + local.x;
${repeat(rows, (r) => `  var sum${r} = vec4<f32>(0.0);${swiglu ? ` var valueSum${r} = vec4<f32>(0.0);` : ""}`)}
  for (var base = 0u; base < inputChannels; base += 8u) {
    for (var i = lane; i < ${inputVectors}u; i += ${threads}u) {
      let r = group.y * ${points}u + i / 8u;
      let k = base + i % 8u;
      var x = Vector(0);
      if (r < rowLimit && k < inputChannels) { x = input[r * inputChannels + k]; }
      tileInput[i] = x;
    }
    for (var i = lane; i < ${weightVectors}u; i += ${threads}u) {
      let k = base * 4u + i / ${columnVectors}u;
      let c = group.x * ${columnVectors}u + i % ${columnVectors}u;
      var w = Vector(0);
      ${swiglu ? "var v = Vector(0);" : ""}
      if (k < inputChannels * 4u && c < outputChannels) {
        let offset = k * outputChannels${swiglu ? " * 2u" : ""} + c;
        w = weights[offset];
        ${swiglu ? "v = weights[offset + outputChannels];" : ""}
      }
      tileWeight[i] = w;
      ${swiglu ? "tileValue[i] = v;" : ""}
    }
    workgroupBarrier();
    for (var i = 0u; i < 8u; i++) {
${repeat(4, (k) => `      let weight${k} = vec4<f32>(tileWeight[(i * 4u + ${k}u) * ${columnVectors}u + local.x]);`)}
${swiglu ? repeat(4, (k) => `      let value${k} = vec4<f32>(tileValue[(i * 4u + ${k}u) * ${columnVectors}u + local.x]);`) : ""}
${repeat(rows, (r) => `      let input${r} = vec4<f32>(tileInput[(local.y * ${rows}u + ${r}u) * 8u + i]);
${accumulate(r, "sum", "weight")}${swiglu ? "\n" + accumulate(r, "valueSum", "value") : ""}`)}
    }
    workgroupBarrier();
  }
  if (col < outputChannels) {
${repeat(rows, (r) => `    if (row + ${r}u < rowLimit) {
      let offset = (row + ${r}u) * outputChannels + col;
      ${store(r)}
    }`)}
  }
}
`
}
