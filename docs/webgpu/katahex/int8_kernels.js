// SPDX-License-Identifier: MIT
// Packed signed 8-bit convolutions with one activation scale per evaluation.

export function createInt8Kernels(half) {
  const scalar = half ? "f16" : "f32"
  const directive = half ? "enable f16;" : ""
  const convolution = `${directive}
requires packed_4x8_integer_dot_product;
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> inputScales: array<f32>;
@group(0) @binding(3) var<storage, read> weightScales: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<Scalar>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(5) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
var<workgroup> tileA: array<u32, 256>;
var<workgroup> tileB: array<u32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let width = param(0u); let height = param(1u); let ic = param(2u); let oc = param(3u);
  let ky = param(4u); let kx = param(5u); let dy = param(6u); let dx = param(7u);
  let row = group.y * 16u + local.y;
  let col = group.x * 16u + local.x;
  let icGroups = ic / 4u;
  let inner = ky * kx * icGroups;
  let inputOffset = group.z * width * height * icGroups;
  let outputOffset = group.z * width * height * oc;
  var sum = 0i;
  for (var base = 0u; base < inner; base += 16u) {
    let k = base + local.x;
    var av = 0u;
    if (row < width * height && k < inner) {
      let fy = k / (kx * icGroups); let fx = (k / icGroups) % kx;
      let y = i32(row / width) + (i32(fy) - i32(ky / 2u)) * i32(dy);
      let x = i32(row % width) + (i32(fx) - i32(kx / 2u)) * i32(dx);
      if (x >= 0 && y >= 0 && x < i32(width) && y < i32(height)) {
        av = input[inputOffset + (u32(y) * width + u32(x)) * icGroups + k % icGroups];
      }
    }
    tileA[local.y * 16u + local.x] = av;
    var bv = 0u;
    if (base + local.y < inner && col < oc) { bv = weights[(base + local.y) * oc + col]; }
    tileB[local.y * 16u + local.x] = bv;
    workgroupBarrier();
    for (var j = 0u; j < 16u; j++) {
      sum += dot4I8Packed(tileA[local.y * 16u + j], tileB[j * 16u + local.x]);
    }
    workgroupBarrier();
  }
  if (row < width * height && col < oc) {
    result[outputOffset + row * oc + col] = Scalar(f32(sum) * inputScales[group.z] * weightScales[col]);
  }
}
`
  return {
    max: `${directive}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
struct Params { values: vec4<u32> }
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> maxima: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let count = params.values.x;
  let offset = group.z * count;
  var peak = 0.0;
  for (var i = local; i < count; i += 256u) {
    peak = max(peak, abs(f32(input[offset + i])));
  }
  maxima[local] = peak;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step /= 2u) {
    if (local < step) { maxima[local] = max(maxima[local], maxima[local + step]); }
    workgroupBarrier();
  }
  if (local == 0u) { scales[group.z] = max(maxima[0] / 127.0, 1e-8); }
}
`,
    pack: `${directive}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> packed: array<u32>;
struct Params { values: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let points = params.values.x;
  let channels = params.values.y;
  let groups = channels / 4u;
  let slot = id.x;
  if (slot >= points * groups) { return; }
  let point = slot / groups;
  let group = slot % groups;
  let offset = id.z * points * channels + point * channels + group * 4u;
  var word = 0u;
  for (var lane = 0u; lane < 4u; lane++) {
    let code = i32(round(clamp(f32(input[offset + lane]) / scales[id.z], -127.0, 127.0)));
    word |= (u32(code) & 255u) << (lane * 8u);
  }
  packed[id.z * points * groups + slot] = word;
}
`,
    conv: convolution,
  }
}
