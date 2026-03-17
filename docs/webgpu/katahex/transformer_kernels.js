// SPDX-License-Identifier: MIT
// Transformer kernels for 32-channel attention heads.
export function createTransformerKernels(half, graphAttention = false) {
  const prefix = `${half ? "enable f16;" : ""}\nalias Scalar = ${half ? "f16" : "f32"};\n`
  const params = `struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
`
  const spatialBindings = (output, residual = false) =>
    prefix +
    `
@group(0) @binding(0) var<storage, read> a: array<Scalar>;
@group(0) @binding(1) var<storage, read> b: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> result: array<${output}>;
${residual ? "@group(0) @binding(3) var<storage, read> residual: array<f32>;" : ""}
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(${residual ? 4 : 3}) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`
  return {
    spatialRmsReduce:
      spatialBindings("f32") +
      `
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let count = param(0u); let channels = param(1u); let tiles = param(2u); let maskSize = param(3u);
  var sum = 0.0;
  for (var j = 0u; j < 4u; j++) {
    let i = group.x * 1024u + lane + j * 256u;
    if (i < count) {
      let p = i / channels;
      if (${graphAttention ? "true" : "p % maskSize + p / maskSize >= (maskSize - 1u) / 2u && p % maskSize + p / maskSize <= 3u * ((maskSize - 1u) / 2u)"}) {
        let x = f32(a[group.z * count + i]); sum += x * x;
      }
    }
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step /= 2u) {
    if (lane < step) { sums[lane] += sums[lane + step]; }
    workgroupBarrier();
  }
  if (lane == 0u) { result[group.z * tiles + group.x] = sums[0]; }
}
`,
    spatialRmsApply:
      spatialBindings("Scalar", true) +
      `
var<workgroup> scale: f32;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(global_invocation_id) id: vec3<u32>) {
  let count = param(0u); let channels = param(1u); let tiles = param(2u); let maskSize = param(3u);
  if (lane == 0u) {
    var sum = 0.0;
    for (var t = 0u; t < tiles; t++) { sum += residual[id.z * tiles + t]; }
    let epsilon = bitcast<f32>(param(5u));
    scale = inverseSqrt(sum / (f32(param(4u) * channels) + epsilon) + epsilon);
  }
  workgroupBarrier();
  let i = id.x;
  if (i >= count) { return; }
  let p = i / channels; let c = i % channels;
  ${graphAttention ? "" : "if (p % maskSize + p / maskSize < (maskSize - 1u) / 2u || p % maskSize + p / maskSize > 3u * ((maskSize - 1u) / 2u)) { result[id.z * count + i] = Scalar(0.0); return; }"}
  var x = f32(a[id.z * count + i]) * scale * f32(b[c]) + f32(b[channels + c]);
  if (param(6u) == 3u) { x /= 1.0 + exp(-x); }
  else if (param(6u) == 1u) { x = max(x, 0.0); }
  result[id.z * count + i] = Scalar(x);
}
`,
    rms:
      prefix +
      `
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;
` +
      params +
      `
var<workgroup> squares: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x; let channels = params.a.y;
  let offset = (group.z * points + group.x) * channels;
  var sum = 0.0;
  for (var c = lane; c < channels; c += 64u) { let x = f32(input[offset + c]); sum += x * x; }
  squares[lane] = sum;
  workgroupBarrier();
  for (var step = 32u; step > 0u; step /= 2u) {
    if (lane < step) { squares[lane] += squares[lane + step]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(squares[0] / f32(channels) + bitcast<f32>(params.a.z));
  for (var c = lane; c < channels; c += 64u) {
    output[offset + c] = Scalar(f32(input[offset + c]) * scale * weight[c]);
  }
}
`,
    attention:
      prefix +
      `
@group(0) @binding(0) var<storage, read> qkv: array<Scalar>;
@group(0) @binding(1) var<storage, read> rope: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;
@group(0) @binding(3) var<storage, read> mask: array<Scalar>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(4) var<uniform> params: Params;
` +
      `
var<workgroup> queries: array<f32, 256>;
var<workgroup> keys: array<f32, 1024>;
var<workgroup> values: array<f32, 1024>;
var<workgroup> scores: array<f32, 256>;
var<workgroup> rowMax: array<f32, 8>;
var<workgroup> rowSum: array<f32, 8>;
var<workgroup> rowScale: array<f32, 8>;
fn rotated(point: u32, head: u32, dim: u32, projection: u32, batch: u32) -> f32 {
  let points = params.a.x; let heads = params.a.y;
  let channels = heads * 32u;
  let index = (batch * points + point) * channels * 3u + projection * channels + head * 32u;
  ${
    graphAttention
      ? "return f32(qkv[index + dim]);"
      : `let table = ((head * 16u + dim / 2u) * points + point) * 2u;
  let x = f32(qkv[index + dim]); let y = f32(qkv[index + (dim ^ 1u)]);
  let sign = select(1.0, -1.0, dim % 2u == 0u);
  return f32(Scalar(x * rope[table] + sign * y * rope[table + 1u]));`
  }
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x; let heads = params.a.y;
  let q = lane / 32u; let dim = lane % 32u;

  let point = group.x * 8u + q; let head = group.y; let batch = group.z;
  var query = 0.0;
  if (point < points) { query = rotated(point, head, dim, 0u, batch); }
  queries[lane] = query;
  if (dim == 0u) { rowMax[q] = -1e30; rowSum[q] = 0.0; }
  var result = 0.0;
  workgroupBarrier();
  for (var base = 0u; base < points; base += 32u) {
    for (var i = lane; i < 1024u; i += 256u) {
      let keyPoint = base + i / 32u; let c = i % 32u;
      var k = 0.0; var v = 0.0;
      if (keyPoint < points) {
        k = rotated(keyPoint, head, c, 1u, batch);
        v = f32(qkv[(batch * points + keyPoint) * heads * 96u + heads * 64u + head * 32u + c]);
      }
      keys[i] = k; values[i] = v;
    }
    workgroupBarrier();
    var score = -1e30;
    if (base + dim < points${graphAttention ? "" : " && mask[(batch * points + base + dim) * params.a.z] != Scalar(0.0)"}) {
      score = 0.0;
      for (var c = 0u; c < 32u; c++) { score += queries[q * 32u + c] * keys[dim * 32u + c]; }
      score *= 0.1767766952966369;
      ${graphAttention ? "if (point < points) { score += rope[(head * points + point) * points + base + dim]; }" : ""}
    }
    scores[lane] = score;
    workgroupBarrier();
    if (dim == 0u) {
      var nextMax = rowMax[q];
      for (var k = 0u; k < 32u; k++) { nextMax = max(nextMax, scores[q * 32u + k]); }
      let scale = exp(rowMax[q] - nextMax);
      var sum = rowSum[q] * scale;
      for (var k = 0u; k < 32u; k++) {
        let probability = exp(scores[q * 32u + k] - nextMax);
        scores[q * 32u + k] = probability;
        sum += probability;
      }
      rowScale[q] = scale; rowMax[q] = nextMax; rowSum[q] = sum;
    }
    workgroupBarrier();
    result *= rowScale[q];
    for (var k = 0u; k < 32u; k++) { result += scores[q * 32u + k] * values[k * 32u + dim]; }
    workgroupBarrier();
  }
  if (point < points) {
    output[(batch * points + point) * heads * 32u + head * 32u + dim] = Scalar(result / rowSum[q]);
  }
}
`,
  }
}
