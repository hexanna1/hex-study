// SPDX-License-Identifier: MIT
// Winograd transforms follow KataGo's OpenCL kernels in cpp/neuralnet/openclkernels.cpp.
import { kernel, READ, WRITE } from "./kernel.js"
import { activationCases } from "./kernels.js"

export const WINOGRAD_OUTPUT_TILE = 4
const INPUT_TILE = WINOGRAD_OUTPUT_TILE + 2
export const WINOGRAD_COEFFICIENTS = INPUT_TILE ** 2

export function isWinogradLayer({ kx, ky, dx, dy, input, output }) {
  return kx === 3 && ky === 3 && dx === 1 && dy === 1 &&
    input >= 32 && output >= 32 && input % 4 === 0 && output % 4 === 0
}

export function countWinogradFilters(descriptor) {
  if (!descriptor || typeof descriptor !== "object") return 0
  if (isWinogradLayer(descriptor)) {
    return descriptor.input * descriptor.output
  }
  return Object.values(descriptor).reduce((total, child) => total + countWinogradFilters(child), 0)
}

const ROOT_TWO = Math.SQRT2
const ROOT_HALF = Math.SQRT1_2

const roots = `
const ROOT_TWO: f32 = ${ROOT_TWO};
const ROOT_HALF: f32 = ${ROOT_HALF};
`

function bindings(half, input, output, weights, extra = []) {
  return `${half ? "enable f16;" : ""}
alias Scalar = ${half ? "f16" : "f32"};
alias Vector = vec4<Scalar>;
@group(0) @binding(0) var<storage, read> a: array<${input}>;
${weights ? `@group(0) @binding(1) var<storage, read> b: array<${weights}>;` : ""}
@group(0) @binding(${weights ? 2 : 1}) var<storage, read_write> result: array<${output}>;
${extra.map((resource, index) => `@group(0) @binding(${(weights ? 3 : 2) + index}) ${resource}`).join("\n")}
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(${(weights ? 3 : 2) + extra.length}) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`
}

function inverseShader(half, outputTile, residual, normalize, keepInput) {
  const inputTile = outputTile + 2
  const compact = outputTile === 2
  const inverseInput = `array<vec4<f32>, ${inputTile}>`
  const inverseOutput = `array<vec4<f32>, ${outputTile}>`
  const columnReads = Array.from({ length: inputTile }, (_, i) => `columns[${i}][y]`).join(", ")
  const extra = []
  if (normalize) extra.push("var<storage, read> affine: array<Vector>;")
  if (keepInput) extra.push("var<storage, read_write> unnormalized: array<Vector>;")
  return bindings(half, "vec4<f32>", "Vector", residual ? "Vector" : null, extra) + `
${compact ? "" : roots}
${normalize ? `
fn activate(value: Vector, channel: u32, channels: u32) -> Vector {
  var x = vec4<f32>(value) * vec4<f32>(affine[channel]) + vec4<f32>(affine[channels + channel]);
${activationCases("param(7u)")}
  return Vector(x);
}
` : ""}
fn outputTransform(d: ${inverseInput}) -> ${inverseOutput} {
${compact ? `  return ${inverseOutput}(d[0] + d[1] + d[2], d[1] - d[2] - d[3]);` : `
  let inner = d[1] - d[2];
  let outer = d[3] - d[4];
  return ${inverseOutput}(
    d[0] + d[1] + d[2] + d[3] + d[4],
    ROOT_HALF * inner + ROOT_TWO * outer,
    0.5 * (d[1] + d[2]) + 2.0 * (d[3] + d[4]),
    (0.5 * ROOT_HALF) * inner + (2.0 * ROOT_TWO) * outer + d[5],
  );`}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let oc = param(3u) / 4u; let tiles = param(4u) * param(5u);
  if (id.x >= tiles * oc) { return; }
  let tile = id.x / oc; let channel = id.x % oc;
  let ox = tile % param(4u) * ${outputTile}u;
  let oy = tile / param(4u) * ${outputTile}u;
  let stride = param(6u) * tiles * oc; let index = (id.z * tiles + tile) * oc + channel;
  var columns: array<${inverseOutput}, ${inputTile}>;
  for (var x = 0u; x < ${inputTile}u; x++) {
    var column: ${inverseInput};
    for (var y = 0u; y < ${inputTile}u; y++) {
      column[y] = a[(y * ${inputTile}u + x) * stride + index];
    }
    columns[x] = outputTransform(column);
  }
  for (var y = 0u; y < ${outputTile}u; y++) {
    let row = outputTransform(${inverseInput}(${columnReads}));
    for (var x = 0u; x < ${outputTile}u; x++) {
      if (ox + x < param(0u) && oy + y < param(1u)) {
        let offset = ((id.z * param(1u) + oy + y) * param(0u) + ox + x) * oc + channel;
        let rounded = Vector(row[x]);
        let value = ${residual ? "Vector(vec4<f32>(rounded) + vec4<f32>(b[offset]))" : "rounded"};
        ${keepInput ? "unnormalized[offset] = value;" : ""}
        result[offset] = ${normalize ? "activate(value, channel, oc)" : "value"};
      }
    }
  }
}
`
}

export function winogradOutputKernels(name, half, outputTile, keepResidual) {
  const inverse = (residual, normalize, keepInput) => inverseShader(half, outputTile, residual, normalize, keepInput)
  return {
    [name]: kernel(inverse(false), [READ, WRITE]),
    [name + "Residual"]: kernel(inverse(true)),
    [name + "Norm"]: kernel(inverse(false, true), [READ, WRITE, READ]),
    [name + "ResidualNorm"]: kernel(inverse(true, true), [READ, READ, WRITE, READ]),
    ...(keepResidual && {
      [name + "ResidualNormKeep"]: kernel(inverse(true, true, true), [READ, READ, WRITE, READ, WRITE]),
    }),
  }
}

export function createWinogradKernels(largeCNN, half) {
  const inputVector = `array<f32, ${INPUT_TILE}>`
  const columnReads = Array.from({ length: INPUT_TILE }, (_, i) => `columns[${i}][y]`).join(", ")
  const input = bindings(half, "Scalar", "Scalar") + `
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
      result[((y * ${INPUT_TILE}u + x) * param(6u) * tiles + id.z * tiles + tile) * ic + channel] = Scalar(row[x]);
    }
  }
}
`
  // Keep the inner loop from dynamically indexing the row accumulators.
  const multiplyRows = (emit) => Array.from({ length: 4 }, (_, row) => emit(row)).join("\n")
  const prefetch = largeCNN ? multiplyRows((lane) => `var nextInput${lane} = Vector(0); var nextWeight${lane} = Vector(0);`) : `
    var nextInput: array<Vector, 4>;
    var nextWeight: array<Vector, 4>;`
  const loadPrefetch = largeCNN ? multiplyRows((lane) => `let i${lane} = lane + ${lane * 64}u;
      nextInput${lane} = inputTile(group.y * 32u + i${lane} / 8u, nextBase + i${lane} % 8u, rows, ic, group.z);
      nextWeight${lane} = weightTile(nextBase * 4u + i${lane} / 8u, group.x * 8u + i${lane} % 8u, ic, oc, group.z);`) : `
      for (var j = 0u; j < 4u; j++) {
        let i = lane + 64u * j;
        nextInput[j] = inputTile(group.y * 32u + i / 8u, nextBase + i % 8u, rows, ic, group.z);
        nextWeight[j] = weightTile(nextBase * 4u + i / 8u, group.x * 8u + i % 8u, ic, oc, group.z);
      }`
  const storePrefetch = largeCNN ? multiplyRows((lane) => `let i${lane} = nextBank + lane + ${lane * 64}u;
      tileInput[i${lane}] = nextInput${lane};
      tileWeight[i${lane}] = nextWeight${lane};`) : `
      for (var j = 0u; j < 4u; j++) {
        let i = nextBank + lane + 64u * j;
        tileInput[i] = nextInput[j];
        tileWeight[i] = nextWeight[j];
      }`
  // Weight loads are unguarded and require whole channel tiles.
  const multiply = bindings(half, "Vector", "vec4<f32>", "Vector") + `
var<workgroup> tileInput: array<Vector, 512>;
var<workgroup> tileWeight: array<Vector, 512>;

fn inputTile(row: u32, channel: u32, rows: u32, ic: u32, plane: u32) -> Vector {
  if (row < rows) {
    return a[(plane * rows + row) * ic + channel];
  }
  return Vector(0.0);
}

fn weightTile(channel: u32, col: u32, ic: u32, oc: u32, plane: u32) -> Vector {
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
${multiplyRows((r) =>
    `  if (row + ${r}u < rows) { result[(group.z * rows + row + ${r}u) * oc + col] = sum${r}; }`,
  )}
}
`
  return {
    winogradInput: kernel(input, [READ, WRITE]),
    winogradMultiply: kernel(multiply),
    ...winogradOutputKernels("winogradOutput", half, WINOGRAD_OUTPUT_TILE, largeCNN),
  }
}
