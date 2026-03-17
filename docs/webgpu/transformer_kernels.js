import { kernel, READ, WRITE } from "./kernel.js"

export const RMS_POINTS_PER_GROUP = 4

export function createTransformerKernels(half) {
  const prefix = `${half ? "enable f16;" : ""}\nalias Scalar = ${half ? "f16" : "f32"};\n`
  const params = `struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
`
  return {
    spatialRmsReduce: kernel(prefix + `
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let count = params.a.x; let tiles = params.a.z;
  var sum = 0.0;
  for (var j = 0u; j < 4u; j++) {
    let i = group.x * 1024u + lane + j * 256u;
    if (i < count) {
      let x = f32(input[group.z * count + i]);
      sum += x * x;
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
`, [READ, WRITE]),
    spatialRmsApply: kernel(prefix + `
alias Vector = vec4<Scalar>;
@group(0) @binding(0) var<storage, read> input: array<Vector>;
@group(0) @binding(1) var<storage, read> weights: array<Vector>;
@group(0) @binding(2) var<storage, read_write> output: array<Vector>;
@group(0) @binding(3) var<storage, read> partials: array<f32>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> scale: f32;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(global_invocation_id) id: vec3<u32>) {
  let count = params.a.x / 4u; let channels = params.a.y / 4u; let tiles = params.a.z;
  if (lane == 0u) {
    var sum = 0.0;
    for (var t = 0u; t < tiles; t++) { sum += partials[id.z * tiles + t]; }
    let epsilon = bitcast<f32>(params.a.w);
    scale = inverseSqrt(sum / (f32(params.a.x) + epsilon) + epsilon);
  }
  workgroupBarrier();
  let i = id.x;
  if (i >= count) { return; }
  let c = i % channels;
  var x = vec4<f32>(input[id.z * count + i]) * scale * vec4<f32>(weights[c]) + vec4<f32>(weights[channels + c]);
  if (params.b.x == 3u) { x /= 1.0 + exp(-x); }
  else if (params.b.x == 1u) { x = max(x, vec4<f32>(0.0)); }
  output[id.z * count + i] = Vector(x);
}
`, [READ, READ, WRITE, READ]),
    rms: kernel(prefix + params + `
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
`),
  }
}
