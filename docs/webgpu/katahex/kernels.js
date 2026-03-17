// SPDX-License-Identifier: MIT
// WebGPU implementations of the operations in KataHex's neural-network backend.

export function createKernels(half) {
  const scalar = half ? "f16" : "f32"
  const bindings = (output = scalar) => `${half ? "enable f16;" : ""}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> a: array<Scalar>;
@group(0) @binding(1) var<storage, read> b: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> result: array<${output}>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`

  // Tiled convolution: rows are board points, columns are output channels.
  const convolution = (pointwise) => bindings() + `
var<workgroup> tileInput: array<Scalar, 1024>;
var<workgroup> tileWeight: array<Scalar, 1024>;
// Each invocation accumulates a 4-by-4 output tile in registers.
@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let width = param(0u); let height = param(1u); let points = width * height;
  let ic = param(2u); let oc = param(3u);
  let ky = param(4u); let kx = param(5u); let dy = param(6u); let dx = param(7u);
  let inner = ky * kx * ic;
  let row = group.y * 32u + local.y * 4u; let col = group.x * 32u + local.x * 4u;
  let lane = local.y * 8u + local.x;
  var sum0 = vec4<f32>(0.0); var sum1 = vec4<f32>(0.0);
  var sum2 = vec4<f32>(0.0); var sum3 = vec4<f32>(0.0);
  for (var base = 0u; base < inner; base += 32u) {
    for (var i = lane; i < 1024u; i += 64u) {
      let r = group.y * 32u + i / 32u; let k = base + i % 32u;
      var av = Scalar(0.0);
      if (r < points && k < inner) {
${pointwise ? "        av = a[(group.z * points + r) * ic + k];" : `        let fy = k / (kx * ic); let fx = (k / ic) % kx;
        let y = i32(r / width) + (i32(fy) - i32(ky / 2u)) * i32(dy);
        let x = i32(r % width) + (i32(fx) - i32(kx / 2u)) * i32(dx);
        if (x >= 0 && y >= 0 && x < i32(width) && y < i32(height)) {
          av = a[(group.z * points + u32(y) * width + u32(x)) * ic + k % ic];
        }`}
      }
      tileInput[i] = av;
      let wk = base + i / 32u; let c = group.x * 32u + i % 32u;
      var bv = Scalar(0.0);
      if (wk < inner && c < oc) { bv = b[wk * oc + c]; }
      tileWeight[i] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let offset = k * 32u + local.x * 4u;
      let b = vec4<f32>(f32(tileWeight[offset]), f32(tileWeight[offset + 1u]),
        f32(tileWeight[offset + 2u]), f32(tileWeight[offset + 3u]));
      sum0 += f32(tileInput[(local.y * 4u) * 32u + k]) * b;
      sum1 += f32(tileInput[(local.y * 4u + 1u) * 32u + k]) * b;
      sum2 += f32(tileInput[(local.y * 4u + 2u) * 32u + k]) * b;
      sum3 += f32(tileInput[(local.y * 4u + 3u) * 32u + k]) * b;
    }
    workgroupBarrier();
  }
  for (var c = 0u; c < 4u; c++) {
    if (col + c < oc) {
      if (row < points) { result[(group.z * points + row) * oc + col + c] = Scalar(sum0[c]); }
      if (row + 1u < points) { result[(group.z * points + row + 1u) * oc + col + c] = Scalar(sum1[c]); }
      if (row + 2u < points) { result[(group.z * points + row + 2u) * oc + col + c] = Scalar(sum2[c]); }
      if (row + 3u < points) { result[(group.z * points + row + 3u) * oc + col + c] = Scalar(sum3[c]); }
    }
  }
}
`

  return {
  conv: convolution(false),
  conv1x1: convolution(true),
  norm: bindings() + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= param(0u)) { return; }
  let c = i % param(1u);
  let offset = id.z * param(0u);
  let maskSize = param(3u);
  if (maskSize > 0u) {
    let point = i / param(1u);
    if (point % maskSize + point / maskSize >= maskSize) {
      result[offset + i] = Scalar(0.0);
      return;
    }
  }
  var x = f32(a[offset + i]) * f32(b[c]) + f32(b[param(1u) + c]);
  if (param(2u) == 1u) {
    x = max(x, 0.0);
  } else if (param(2u) == 2u) {
    let softplus = max(x, 0.0) + log(1.0 + exp(-abs(x)));
    x *= 2.0 / (1.0 + exp(-2.0 * softplus)) - 1.0;
  }

  result[offset + i] = Scalar(x);
}
`,
  add: bindings() + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < param(0u)) {
    let offset = id.z * param(0u);
    result[offset + i] = Scalar(f32(a[offset + i]) + f32(b[id.z * param(1u) + i % param(1u)]));
  }
}
`,
  // The spatial input's first channel is the mask.
  pool: bindings() + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x; let points = param(0u); let channels = param(1u);
  if (c >= channels) { return; }
  let inputOffset = id.z * points * channels;
  let outputOffset = id.z * channels * 3u;
  var sum = 0.0; var maximum = -1.0;
  for (var i = 0u; i < points; i++) {
    let x = f32(a[inputOffset + i * channels + c]);
    let mask = f32(b[(id.z * points + i) * param(4u)]);
    sum += x; maximum = max(maximum, x + mask - 1.0);
  }
  let mean = sum / f32(param(3u));
  let sizeScale = (sqrt(f32(param(3u))) - 14.0) * 0.1;
  result[outputOffset + c] = Scalar(mean);
  result[outputOffset + channels + c] = Scalar(mean * sizeScale);
  result[outputOffset + 2u * channels + c] = Scalar(select(maximum, mean * (sizeScale * sizeScale - 0.1), param(2u) == 1u));
}
`,
  readout: bindings("f32") + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; let points = param(0u);
  if (i < points) { result[id.z * (points + 3u) + i] = f32(a[id.z * points + i]); }
  else if (i < points + 3u) { result[id.z * (points + 3u) + i] = f32(b[id.z * 3u + i - points]); }
}
`,
  }
}
