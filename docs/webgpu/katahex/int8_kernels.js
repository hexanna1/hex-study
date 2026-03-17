// SPDX-License-Identifier: MIT
// Packed signed 8-bit convolutions with one activation scale per evaluation.
// Each invocation accumulates a 4-by-4 output tile in registers.

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
var<workgroup> tileA: array<u32, 1024>;
var<workgroup> tileB: array<u32, 1024>;
@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let width = param(0u); let height = param(1u); let ic = param(2u); let oc = param(3u);
  let ky = param(4u); let kx = param(5u); let dy = param(6u); let dx = param(7u);
  let row = group.y * 32u + local.y * 4u;
  let col = group.x * 32u + local.x * 4u;
  let lane = local.y * 8u + local.x;
  let groups = ic / 4u; let inner = ky * kx * groups;
  let inputOffset = group.z * width * height * groups;
  let outputOffset = group.z * width * height * oc;
  var sum0 = vec4<i32>(0);
  var sum1 = vec4<i32>(0);
  var sum2 = vec4<i32>(0);
  var sum3 = vec4<i32>(0);
  for (var base = 0u; base < inner; base += 32u) {
    for (var i = lane; i < 1024u; i += 64u) {
      let r = group.y * 32u + i / 32u; let k = base + i % 32u;
      var a = 0u;
      if (r < width * height && k < inner) {
        let fy = k / (kx * groups); let fx = (k / groups) % kx;
        let y = i32(r / width) + (i32(fy) - i32(ky / 2u)) * i32(dy);
        let x = i32(r % width) + (i32(fx) - i32(kx / 2u)) * i32(dx);
        if (x >= 0 && y >= 0 && x < i32(width) && y < i32(height)) {
          a = input[inputOffset + (u32(y) * width + u32(x)) * groups + k % groups];
        }
      }
      tileA[i] = a;
    }
    for (var i = lane; i < 1024u; i += 64u) {
      let k = base + i / 32u; let c = group.x * 32u + i % 32u;
      var b = 0u;
      if (k < inner && c < oc) { b = weights[k * oc + c]; }
      tileB[i] = b;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let offset = k * 32u + local.x * 4u;
      let b0 = tileB[offset]; let b1 = tileB[offset + 1u];
      let b2 = tileB[offset + 2u]; let b3 = tileB[offset + 3u];
      let a0 = tileA[(local.y * 4u) * 32u + k];
      sum0 += vec4<i32>(dot4I8Packed(a0, b0), dot4I8Packed(a0, b1), dot4I8Packed(a0, b2), dot4I8Packed(a0, b3));
      let a1 = tileA[(local.y * 4u + 1u) * 32u + k];
      sum1 += vec4<i32>(dot4I8Packed(a1, b0), dot4I8Packed(a1, b1), dot4I8Packed(a1, b2), dot4I8Packed(a1, b3));
      let a2 = tileA[(local.y * 4u + 2u) * 32u + k];
      sum2 += vec4<i32>(dot4I8Packed(a2, b0), dot4I8Packed(a2, b1), dot4I8Packed(a2, b2), dot4I8Packed(a2, b3));
      let a3 = tileA[(local.y * 4u + 3u) * 32u + k];
      sum3 += vec4<i32>(dot4I8Packed(a3, b0), dot4I8Packed(a3, b1), dot4I8Packed(a3, b2), dot4I8Packed(a3, b3));
    }
    workgroupBarrier();
  }
  for (var c = 0u; c < 4u; c++) {
    if (col + c < oc) {
      if (row < width * height) { result[outputOffset + row * oc + col + c] = Scalar(f32(sum0[c]) * inputScales[group.z] * weightScales[col + c]); }
      if (row + 1u < width * height) { result[outputOffset + (row + 1u) * oc + col + c] = Scalar(f32(sum1[c]) * inputScales[group.z] * weightScales[col + c]); }
      if (row + 2u < width * height) { result[outputOffset + (row + 2u) * oc + col + c] = Scalar(f32(sum2[c]) * inputScales[group.z] * weightScales[col + c]); }
      if (row + 3u < width * height) { result[outputOffset + (row + 3u) * oc + col + c] = Scalar(f32(sum3[c]) * inputScales[group.z] * weightScales[col + c]); }
    }
  }
}
`
  return {
    maxpack: `${directive}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
struct Params { values: vec4<u32> }
@group(0) @binding(2) var<storage, read_write> packed: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
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
  let scale = max(maxima[0] / 127.0, 1e-8);
  if (local == 0u) { scales[group.z] = scale; }
  for (var slot = local; slot < count / 4u; slot += 256u) {
    var word = 0u;
    for (var lane = 0u; lane < 4u; lane++) {
      let code = i32(round(clamp(f32(input[offset + slot * 4u + lane]) / scale, -127.0, 127.0)));
      word |= (u32(code) & 255u) << (lane * 8u);
    }
    packed[group.z * count / 4u + slot] = word;
  }
}
`,
    conv: convolution,
  }
}
