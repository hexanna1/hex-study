// Transformer kernels for 32-channel attention heads.
export const RMS_POINTS_PER_GROUP = 4

export function createTransformerKernels(half, game, graphAttention) {
  const yavalath = game === "yavalath"
  const prefix = `${half ? "enable f16;" : ""}\nalias Scalar = ${half ? "f16" : "f32"};\n`
  const params = `struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
`
  const kernels = {
    spatialRmsReduce: prefix + params + `
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
${yavalath ? "@group(0) @binding(1) var<storage, read> denseToPacked: array<u32>;" : ""}
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let count = params.a.x; let channels = params.a.y; let tiles = params.a.z; let reductionCount = params.a.w;
  var sum = 0.0;
  for (var j = 0u; j < 4u; j++) {
    let i = group.x * 1024u + lane + j * 256u;
    if (i < reductionCount) {
      ${yavalath ? `let packed = denseToPacked[i / channels];
      if (packed != 0xffffffffu) {
        let x = f32(input[group.z * count + packed * channels + i % channels]);
        sum += x * x;
      }` : `let x = f32(input[group.z * count + i]);
      sum += x * x;`}
    }
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step /= 2u) {
    if (lane < step) { sums[lane] += sums[lane + step]; }
    workgroupBarrier();
  }
  if (lane == 0u) { partials[group.z * tiles + group.x] = sums[0]; }
}
`,
    spatialRmsApply: prefix + `
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read> weights: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;
@group(0) @binding(3) var<storage, read> partials: array<f32>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> scale: f32;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(global_invocation_id) id: vec3<u32>) {
  let count = params.a.x; let channels = params.a.y; let tiles = params.a.z;
  if (lane == 0u) {
    var sum = 0.0;
    for (var t = 0u; t < tiles; t++) { sum += partials[id.z * tiles + t]; }
    let epsilon = bitcast<f32>(params.b.y);
    scale = inverseSqrt(sum / (f32(params.b.x * channels) + epsilon) + epsilon);
  }
  workgroupBarrier();
  let i = id.x;
  if (i >= count) { return; }
  let c = i % channels;
  var x = f32(input[id.z * count + i]) * scale * f32(weights[c]) + f32(weights[channels + c]);
  if (params.b.z == 3u) { x /= 1.0 + exp(-x); }
  else if (params.b.z == 1u) { x = max(x, 0.0); }
  output[id.z * count + i] = Scalar(x);
}
`,
    rms: prefix + params + `
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;
var<workgroup> squares: array<f32, ${32 * RMS_POINTS_PER_GROUP}>;
@compute @workgroup_size(32, ${RMS_POINTS_PER_GROUP})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) index: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x; let channels = params.a.y;
  let point = group.x * ${RMS_POINTS_PER_GROUP}u + local.y;
  let offset = (group.z * points + point) * channels;
  var low = 0.0; var high = 0.0;
  if (point < points) {
    for (var c = local.x; c < channels; c += 64u) { let x = f32(input[offset + c]); low += x * x; }
    for (var c = local.x + 32u; c < channels; c += 64u) { let x = f32(input[offset + c]); high += x * x; }
  }
  squares[index] = low + high;
  workgroupBarrier();
  for (var step = 16u; step > 0u; step /= 2u) {
    if (local.x < step) { squares[index] += squares[index + step]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(squares[local.y * 32u] / f32(channels) + bitcast<f32>(params.a.z));
  if (point < points) {
    for (var c = local.x; c < channels; c += 32u) {
      output[offset + c] = Scalar(f32(input[offset + c]) * scale * weight[c]);
    }
  }
}
`,
  }
  // Only small FP32 Y boards use staged attention.
  if (game !== "y" || half || graphAttention) return kernels
  return {
    ...kernels,
    attentionSoftmax,
    // Attention uses tiled matrix products with float32 scores and softmax reductions.
    attentionScores:
      prefix + params + `
@group(0) @binding(0) var<storage, read> qkv: array<Scalar>;
@group(0) @binding(1) var<storage, read> rope: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
var<workgroup> queries: array<f32, 512>;
var<workgroup> keys: array<vec4<f32>, 256>;
fn rotated(point: u32, head: u32, dim: u32, projection: u32, batch: u32) -> f32 {
  let points = params.a.x; let heads = params.a.y;
  let index = (batch * points + point) * heads * 96u + projection * heads * 32u + head * 32u;
  let table = ((head * 16u + dim / 2u) * points + point) * 2u;
  let x = f32(qkv[index + dim]); let y = f32(qkv[index + (dim ^ 1u)]);
  let sign = select(1.0, -1.0, dim % 2u == 0u);
  return f32(Scalar(x * rope[table] + sign * y * rope[table + 1u]));
}
@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x; let heads = params.a.y; let head = group.y % heads; let batch = group.z;
  let queryStart = (group.y / heads) * 16u; let keyStart = group.x * 32u;
  for (var i = lane; i < 512u; i += 64u) {
    let point = queryStart + i / 32u;
    var x = 0.0;
    if (point < points) { x = rotated(point, head, i % 32u, 0u, batch); }
    queries[i] = x;
  }
  for (var i = lane; i < 256u; i += 64u) {
    let dim = i / 8u; let key = keyStart + (i % 8u) * 4u;
    var x = vec4<f32>(0.0);
    for (var c = 0u; c < 4u; c++) {
      if (key + c < points) { x[c] = rotated(key + c, head, dim, 1u, batch); }
    }
    keys[i] = x;
  }
  workgroupBarrier();
  var sum0 = vec4<f32>(0.0); var sum1 = vec4<f32>(0.0);
  for (var c = 0u; c < 32u; c++) {
    let k = keys[c * 8u + local.x];
    sum0 += queries[(local.y * 2u) * 32u + c] * k;
    sum1 += queries[(local.y * 2u + 1u) * 32u + c] * k;
  }
  for (var r = 0u; r < 2u; r++) {
    let point = queryStart + local.y * 2u + r; let values = select(sum0, sum1, r == 1u);
    if (point < points) {
      for (var c = 0u; c < 4u; c++) {
        let key = keyStart + local.x * 4u + c;
        if (key < points) {
          let score = values[c] * 0.1767766952966369;
          scores[((batch * heads + head) * points + point) * points + key] = score;
        }
      }
    }
  }
}
`,
    attentionValues:
      prefix + params + `
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> qkv: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;
var<workgroup> probabilities: array<f32, 512>;
var<workgroup> values: array<vec4<f32>, 256>;
@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x; let heads = params.a.y; let head = group.y; let batch = group.z;
  let queryStart = group.x * 16u;
  var sum0 = vec4<f32>(0.0); var sum1 = vec4<f32>(0.0);
  for (var base = 0u; base < points; base += 32u) {
    for (var i = lane; i < 512u; i += 64u) {
      let point = queryStart + i / 32u; let key = base + i % 32u;
      var x = 0.0;
      if (point < points && key < points) { x = scores[((batch * heads + head) * points + point) * points + key]; }
      probabilities[i] = x;
    }
    for (var i = lane; i < 256u; i += 64u) {
      let key = base + i / 8u; let dim = (i % 8u) * 4u;
      var x = vec4<f32>(0.0);
      if (key < points) {
        let offset = (batch * points + key) * heads * 96u + heads * 64u + head * 32u + dim;
        x = vec4<f32>(f32(qkv[offset]), f32(qkv[offset + 1u]), f32(qkv[offset + 2u]), f32(qkv[offset + 3u]));
      }
      values[i] = x;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let v = values[k * 8u + local.x];
      sum0 += probabilities[(local.y * 2u) * 32u + k] * v;
      sum1 += probabilities[(local.y * 2u + 1u) * 32u + k] * v;
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < 2u; r++) {
    let point = queryStart + local.y * 2u + r; let x = select(sum0, sum1, r == 1u);
    if (point < points) {
      let offset = (batch * points + point) * heads * 32u + head * 32u + local.x * 4u;
      for (var c = 0u; c < 4u; c++) { output[offset + c] = Scalar(x[c]); }
    }
  }
}
`,
  }
}

const attentionSoftmax = `
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> scratch: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x;
  let offset = ((group.z * params.a.y + group.y) * points + group.x) * points;
  let lowKey = lane;
  let highKey = lane + 64u;
  var low = -1e30; var high = -1e30;
  if (lowKey < points) { low = scores[offset + lowKey]; }
  if (highKey < points) { high = scores[offset + highKey]; }
  scratch[lane] = max(max(-1e30, low), max(-1e30, high));
  workgroupBarrier();
  for (var step = 32u; step > 0u; step /= 2u) {
    if (lane < step) { scratch[lane] = max(scratch[lane], scratch[lane + step]); }
    workgroupBarrier();
  }
  let rowMax = scratch[0];
  workgroupBarrier();
  var sumLow = 0.0; var sumHigh = 0.0;
  if (lowKey < points) { low = exp(low - rowMax); sumLow += low; }
  if (highKey < points) { high = exp(high - rowMax); sumHigh += high; }
  scratch[lane] = sumLow + sumHigh;
  workgroupBarrier();
  for (var step = 32u; step > 0u; step /= 2u) {
    if (lane < step) { scratch[lane] += scratch[lane + step]; }
    workgroupBarrier();
  }
  let scale = 1.0 / scratch[0];
  if (lowKey < points) { scores[offset + lowKey] = low * scale; }
  if (highKey < points) { scores[offset + highKey] = high * scale; }
}
`
