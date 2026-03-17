// SPDX-License-Identifier: MIT
export const WINOGRAD_OUTPUT_TILE = 4
const INPUT_TILE = WINOGRAD_OUTPUT_TILE + 2
export const WINOGRAD_COEFFICIENTS = INPUT_TILE ** 2

const ROOT_TWO = Math.SQRT2
const ROOT_HALF = Math.SQRT1_2
const INNER_FILTER = [-2 / 3, -ROOT_TWO / 3, -1 / 3]
const OUTER_FILTER = [1 / 6, ROOT_TWO / 6, 1 / 3]

export function createWinogradKernels(registerPrefetch) {
  const inputVector = `array<f32, ${INPUT_TILE}>`
  const inverseInput = `array<vec4<f32>, ${INPUT_TILE}>`
  const inverseOutput = `array<vec4<f32>, ${WINOGRAD_OUTPUT_TILE}>`
  const columnReads = Array.from({ length: INPUT_TILE }, (_, i) => `columns[${i}][y]`).join(", ")
  const roots = `
const ROOT_TWO: f32 = ${ROOT_TWO};
const ROOT_HALF: f32 = ${ROOT_HALF};
`
  const bindings = (input, output, weights = "f16") => `enable f16;
@group(0) @binding(0) var<storage, read> a: array<${input}>;
@group(0) @binding(1) var<storage, read> b: array<${weights}>;
@group(0) @binding(2) var<storage, read_write> result: array<${output}>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`
  const input = bindings("f16", "f16") + `
${roots}
fn inputTransform(d: ${inputVector}) -> ${inputVector} {
  let even = d[4] - 2.0 * d[2];
  let odd = ROOT_HALF * d[3] - ROOT_TWO * d[1];
  let evenOuter = d[4] - 0.5 * d[2];
  let oddOuter = ROOT_TWO * d[3] - ROOT_HALF * d[1];
  return ${inputVector}(
    d[0] - 2.5 * d[2] + d[4],
    even + odd, even - odd, evenOuter + oddOuter, evenOuter - oddOuter,
    d[1] - 2.5 * d[3] + d[5],
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ic = param(2u); let tiles = param(4u) * param(5u);
  if (id.x >= tiles * ic) { return; }
  let tile = id.x / ic; let channel = id.x % ic;
  let ox = i32(tile % param(4u)) * ${WINOGRAD_OUTPUT_TILE} - 1;
  let oy = i32(tile / param(4u)) * ${WINOGRAD_OUTPUT_TILE} - 1;
  var columns: array<${inputVector}, ${INPUT_TILE}>;
  for (var x = 0u; x < ${INPUT_TILE}u; x++) {
    let ix = ox + i32(x);
    var column: ${inputVector};
    for (var y = 0u; y < ${INPUT_TILE}u; y++) {
      let iy = oy + i32(y);
      var value = 0.0;
      if (ix >= 0 && iy >= 0 && ix < i32(param(0u)) && iy < i32(param(1u))) {
        value = f32(a[((id.z * param(1u) + u32(iy)) * param(0u) + u32(ix)) * ic + channel]);
      }
      column[y] = value;
    }
    columns[x] = inputTransform(column);
  }
  for (var y = 0u; y < ${INPUT_TILE}u; y++) {
    let row = inputTransform(${inputVector}(${columnReads}));
    for (var x = 0u; x < ${INPUT_TILE}u; x++) {
      result[((y * ${INPUT_TILE}u + x) * param(6u) * tiles + id.z * tiles + tile) * ic + channel] = f16(row[x]);
    }
  }
}
`
  // Keep the inner loop from dynamically indexing the row accumulators.
  const multiplyRows = (emit) => Array.from({ length: 4 }, (_, row) => emit(row)).join("\n")
  // The wider B18 convolutions benefit from statically named prefetch vectors.
  const prefetch = registerPrefetch ? multiplyRows((lane) => `var nextInput${lane} = vec4<f16>(0); var nextWeight${lane} = vec4<f16>(0);`) : `
    var nextInput: array<vec4<f16>, 4>;
    var nextWeight: array<vec4<f16>, 4>;`
  const loadPrefetch = registerPrefetch ? multiplyRows((lane) => `let i${lane} = lane + ${lane * 64}u;
      nextInput${lane} = inputTile(group.y * 32u + i${lane} / 8u, nextBase + i${lane} % 8u, rows, ic, group.z);
      nextWeight${lane} = weightTile(nextBase * 4u + i${lane} / 8u, group.x * 8u + i${lane} % 8u, ic, oc, group.z);`) : `
      for (var j = 0u; j < 4u; j++) {
        let i = lane + 64u * j;
        nextInput[j] = inputTile(group.y * 32u + i / 8u, nextBase + i % 8u, rows, ic, group.z);
        nextWeight[j] = weightTile(nextBase * 4u + i / 8u, group.x * 8u + i % 8u, ic, oc, group.z);
      }`
  const storePrefetch = registerPrefetch ? multiplyRows((lane) => `let i${lane} = nextBank + lane + ${lane * 64}u;
      tileInput[i${lane}] = nextInput${lane};
      tileWeight[i${lane}] = nextWeight${lane};`) : `
      for (var j = 0u; j < 4u; j++) {
        let i = nextBank + lane + 64u * j;
        tileInput[i] = nextInput[j];
        tileWeight[i] = nextWeight[j];
      }`
  const multiply = (aligned) => bindings("vec4<f16>", "vec4<f32>", "vec4<f16>") + `
var<workgroup> tileInput: array<vec4<f16>, 512>;
var<workgroup> tileWeight: array<vec4<f16>, 512>;

fn inputTile(row: u32, channel: u32, rows: u32, ic: u32, plane: u32) -> vec4<f16> {
  if (row < rows${aligned ? "" : " && channel < ic"}) {
    return a[(plane * rows + row) * ic + channel];
  }
  return vec4<f16>(0.0h);
}

fn weightTile(channel: u32, col: u32, ic: u32, oc: u32, plane: u32) -> vec4<f16> {
  ${aligned ? "" : "if (channel >= ic * 4u || col >= oc) { return vec4<f16>(0.0h); }"}
  return b[(plane * ic * 4u + channel) * oc + col];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let rows = param(4u) * param(5u) * param(6u);
  let ic = param(2u) / 4u; let oc = param(3u) / 4u;
  let row = group.y * 32u + local.y * 4u; let col = group.x * 8u + local.x;
  let lane = local.y * 8u + local.x;
${multiplyRows((r) => `  var sum${r} = vec4<f32>(0.0);`)}
  for (var i = lane; i < 256u; i += 64u) {
    tileInput[i] = inputTile(group.y * 32u + i / 8u, i % 8u, rows, ic, group.z);
    tileWeight[i] = weightTile(i / 8u, group.x * 8u + i % 8u, ic, oc, group.z);
  }
  workgroupBarrier();
  for (var base = 0u; base < ic; base += 8u) {
    let bank = (base / 8u % 2u) * 256u;
    let nextBank = 256u - bank;
    let nextBase = base + 8u;
    let hasNext = nextBase < ic;
    ${prefetch}
    if (hasNext) { ${loadPrefetch} }
    for (var k = 0u; k < 8u; k++) {
      let offset = bank + k * 32u + local.x;
      let b0 = vec4<f32>(tileWeight[offset]);
      let b1 = vec4<f32>(tileWeight[offset + 8u]);
      let b2 = vec4<f32>(tileWeight[offset + 16u]);
      let b3 = vec4<f32>(tileWeight[offset + 24u]);
${multiplyRows((r) => `      let a${r} = vec4<f32>(tileInput[bank + (local.y * 4u + ${r}u) * 8u + k]);
      sum${r} += a${r}.x * b0;
      sum${r} += a${r}.y * b1;
      sum${r} += a${r}.z * b2;
      sum${r} += a${r}.w * b3;`)}
    }
    if (hasNext) {
      ${storePrefetch}
      workgroupBarrier();
    }
  }
  ${aligned ? "" : "if (col >= oc) { return; }"}
${multiplyRows((r) =>
    `  if (row + ${r}u < rows) { result[(group.z * rows + row + ${r}u) * oc + col] = sum${r}; }`,
  )}
}
`
  const inverse = (residual) => bindings("vec4<f32>", "vec4<f16>", "vec4<f16>") + `
${roots}
fn outputTransform(d: ${inverseInput}) -> ${inverseOutput} {
  let inner = d[1] - d[2];
  let outer = d[3] - d[4];
  return ${inverseOutput}(
    d[0] + d[1] + d[2] + d[3] + d[4],
    ROOT_HALF * inner + ROOT_TWO * outer,
    0.5 * (d[1] + d[2]) + 2.0 * (d[3] + d[4]),
    (0.5 * ROOT_HALF) * inner + (2.0 * ROOT_TWO) * outer + d[5],
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let oc = param(3u) / 4u; let tiles = param(4u) * param(5u);
  if (id.x >= tiles * oc) { return; }
  let tile = id.x / oc; let channel = id.x % oc;
  let ox = tile % param(4u) * ${WINOGRAD_OUTPUT_TILE}u;
  let oy = tile / param(4u) * ${WINOGRAD_OUTPUT_TILE}u;
  let stride = param(6u) * tiles * oc; let index = (id.z * tiles + tile) * oc + channel;
  var columns: array<${inverseOutput}, ${INPUT_TILE}>;
  for (var x = 0u; x < ${INPUT_TILE}u; x++) {
    var column: ${inverseInput};
    for (var y = 0u; y < ${INPUT_TILE}u; y++) {
      column[y] = a[(y * ${INPUT_TILE}u + x) * stride + index];
    }
    columns[x] = outputTransform(column);
  }
  for (var y = 0u; y < ${WINOGRAD_OUTPUT_TILE}u; y++) {
    let row = outputTransform(${inverseInput}(${columnReads}));
    for (var x = 0u; x < ${WINOGRAD_OUTPUT_TILE}u; x++) {
      if (ox + x < param(0u) && oy + y < param(1u)) {
        let offset = ((id.z * param(1u) + oy + y) * param(0u) + ox + x) * oc + channel;
        let rounded = vec4<f16>(row[x]);
        result[offset] = ${residual ? "vec4<f16>(vec4<f32>(rounded) + vec4<f32>(b[offset]))" : "rounded"};
      }
    }
  }
}
`
  return {
    winogradInput: input,
    winogradMultiply: multiply(false),
    winogradMultiplyAligned: multiply(true),
    winogradOutput: inverse(false),
    winogradOutputResidual: inverse(true),
  }
}

export function transformWinogradWeights(layer, weights, columns, otherColumns) {
  const count = layer.input * layer.output
  const stride = 3 * count
  const [innerFirst, innerMiddle, innerLast] = INNER_FILTER
  const [outerFirst, outerMiddle, outerLast] = OUTER_FILTER

  const writeRow = (row, input) => {
    const offset = row * INPUT_TILE * count
    for (let i = 0; i < count; i++) {
      const left = input[i], center = input[count + i], right = input[2 * count + i]
      const innerLeft = innerFirst * left, innerCenter = innerMiddle * center, innerRight = innerLast * right
      const outerLeft = outerFirst * left, outerCenter = outerMiddle * center, outerRight = outerLast * right
      weights[offset + i] = (left + 0 * center) + 0 * right
      weights[offset + count + i] = (innerLeft + innerCenter) + innerRight
      weights[offset + 2 * count + i] = (innerLeft - innerCenter) + innerRight
      weights[offset + 3 * count + i] = (outerLeft + outerCenter) + outerRight
      weights[offset + 4 * count + i] = (outerLeft - outerCenter) + outerRight
      weights[offset + 5 * count + i] = (0 * left + 0 * center) + right
    }
  }
  for (let i = 0; i < stride; i++) {
    const left = layer.weights[i], center = layer.weights[stride + i], right = layer.weights[2 * stride + i]
    columns[i] = (left + 0 * center) + 0 * right
    otherColumns[i] = (0 * left + 0 * center) + right
  }
  writeRow(0, columns)
  writeRow(5, otherColumns)

  const writePair = (row, [first, middle, last]) => {
    for (let i = 0; i < stride; i++) {
      const left = first * layer.weights[i]
      const center = middle * layer.weights[stride + i]
      const right = last * layer.weights[2 * stride + i]
      columns[i] = (left + center) + right
      otherColumns[i] = (left - center) + right
    }
    writeRow(row, columns)
    writeRow(row + 1, otherColumns)
  }
  writePair(1, INNER_FILTER)
  writePair(3, OUTER_FILTER)
  return weights
}

let compiledWeights
let loadingWeights

export function preloadWinogradWeights() {
  if (typeof WebAssembly === "object" && !compiledWeights && !loadingWeights) {
    loadingWeights = compileWinogradWeights().catch(() => {}).finally(() => { loadingWeights = null })
  }
}

async function compileWinogradWeights() {
  const response = await fetch(new URL("./winograd_weights.wasm", import.meta.url))
  if (response.ok) compiledWeights = await WebAssembly.compile(await response.arrayBuffer())
}

export function createWinogradWeightTransform() {
  if (!compiledWeights) return null
  let instance
  try {
    instance = new WebAssembly.Instance(compiledWeights)
  } catch {
    return null
  }
  const { memory, transform, __heap_base } = instance.exports
  const output = __heap_base.value
  return (codes, scales) => {
    // Each view is consumed before the next layer reuses the linear memory.
    const count = codes.length / 9
    const length = WINOGRAD_COEFFICIENTS * count
    const source = output + length * Uint16Array.BYTES_PER_ELEMENT
    const scaleOffset = source + codes.byteLength
    const end = scaleOffset + scales.byteLength
    const pageBytes = 65536
    const missing = end - memory.buffer.byteLength
    if (missing > 0) memory.grow(Math.ceil(missing / pageBytes))
    const bytes = memory.buffer
    new Int8Array(bytes, source, codes.length).set(codes)
    new Float32Array(bytes, scaleOffset, scales.length).set(scales)
    transform(output, source, scaleOffset, count, scales.length)
    return new Uint16Array(bytes, output, length)
  }
}
