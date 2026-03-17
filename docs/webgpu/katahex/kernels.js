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

  return {
  // Tiled implicit-im2col convolution. Rows are board points, columns are output channels.
  conv: bindings() + `
var<workgroup> tileA: array<Scalar, 256>;
var<workgroup> tileB: array<Scalar, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let width = param(0u); let height = param(1u); let ic = param(2u); let oc = param(3u);
  let ky = param(4u); let kx = param(5u); let dy = param(6u); let dx = param(7u);
  let row = group.y * 16u + local.y;
  let col = group.x * 16u + local.x;
  let inner = ky * kx * ic;
  let inputOffset = group.z * width * height * ic;
  let outputOffset = group.z * width * height * oc;
  var sum = 0.0;
  for (var base = 0u; base < inner; base += 16u) {
    let k = base + local.x;
    var av = 0.0;
    if (row < width * height && k < inner) {
      let fy = k / (kx * ic); let fx = (k / ic) % kx;
      let y = i32(row / width) + (i32(fy) - i32(ky / 2u)) * i32(dy);
      let x = i32(row % width) + (i32(fx) - i32(kx / 2u)) * i32(dx);
      if (x >= 0 && y >= 0 && x < i32(width) && y < i32(height)) {
        av = f32(a[inputOffset + (u32(y) * width + u32(x)) * ic + k % ic]);
      }
    }
    tileA[local.y * 16u + local.x] = Scalar(av);
    var bv = 0.0;
    if (base + local.y < inner && col < oc) { bv = f32(b[(base + local.y) * oc + col]); }
    tileB[local.y * 16u + local.x] = Scalar(bv);
    workgroupBarrier();
    for (var j = 0u; j < 16u; j++) {
      sum += f32(tileA[local.y * 16u + j]) * f32(tileB[j * 16u + local.x]);
    }
    workgroupBarrier();
  }
  if (row < width * height && col < oc) { result[outputOffset + row * oc + col] = Scalar(sum); }
}
`,
  norm: bindings() + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= param(0u)) { return; }
  let c = i % param(1u);
  let offset = id.z * param(0u);
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
  mask: bindings() + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; let count = param(0u); let channels = param(1u);
  if (i >= count) { return; }
  let point = i / channels;
  let onBoard = b[(id.z * (count / channels) + point) * param(2u)] != Scalar(0.0);
  result[id.z * count + i] = select(Scalar(0.0), a[id.z * count + i], onBoard);
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
  // Pool over playable cells; the spatial input's first channel is the mask.
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
