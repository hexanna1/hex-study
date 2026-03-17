// SPDX-License-Identifier: MIT
// WebGPU implementations of the operations in KataHex's neural-network backend.

export function createKernels(half, transformer = false, graphAttention = false) {
  const scalar = half ? "f16" : "f32"
  const bindings = (output = scalar, input = "Scalar") => `${half ? "enable f16;" : ""}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> a: array<${input}>;
@group(0) @binding(1) var<storage, read> b: array<${input}>;
@group(0) @binding(2) var<storage, read_write> result: array<${output}>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`

  const add = (vector) => {
    const value = vector ? "vec4<Scalar>" : "Scalar"
    const arithmetic = vector ? "vec4<f32>" : "f32"
    const stride = vector ? " / 4u" : ""
    return bindings(value, value) + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let count = param(0u)${stride};
  let other = param(1u)${stride};
  if (id.x < count) {
    let i = id.z * count + id.x;
    result[i] = ${value}(${arithmetic}(a[i]) + ${arithmetic}(b[id.z * other + id.x % other]));
  }
}
`
  }

  // Tiled convolution: rows are board points, columns are output channels.
  const rows = transformer ? 2 : 4
  const convolution = (pointwise) =>
    bindings() +
    `
var<workgroup> tileInput: array<Scalar, 1024>;
var<workgroup> tileWeight: array<Scalar, 1024>;
// Each invocation accumulates a tile in registers.
@compute @workgroup_size(8, ${32 / rows})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let width = param(0u); let height = param(1u); let points = width * height;
  let ic = param(2u); let oc = param(3u);
  let ky = param(4u); let kx = param(5u); let dy = param(6u); let dx = param(7u);
  let inner = ky * kx * ic;
  let row = group.y * 32u + local.y * ${rows}u; let col = group.x * 32u + local.x * 4u;
  let lane = local.y * 8u + local.x;
${Array.from({ length: rows }, (_, r) => `  var sum${r} = vec4<f32>(0.0);`).join("\n")}
  for (var base = 0u; base < inner; base += 32u) {
    for (var i = lane; i < 1024u; i += ${256 / rows}u) {
      let r = group.y * 32u + i / 32u; let k = base + i % 32u;
      var av = Scalar(0.0);
      if (r < points && k < inner) {
${
  pointwise
    ? "        av = a[(group.z * points + r) * ic + k];"
    : `        let fy = k / (kx * ic); let fx = (k / ic) % kx;
        let y = i32(r / width) + (i32(fy) - i32(ky / 2u)) * i32(dy);
        let x = i32(r % width) + (i32(fx) - i32(kx / 2u)) * i32(dx);
        if (x >= 0 && y >= 0 && x < i32(width) && y < i32(height)) {
          av = a[(group.z * points + u32(y) * width + u32(x)) * ic + k % ic];
        }`
}
      }
      tileInput[i] = av;
      let wk = base + i / 32u; let c = group.x * 32u + i % 32u;
      var bv = Scalar(0.0);
      if (wk < inner && c < oc) {
        bv = b[wk * oc + c];
      }
      tileWeight[i] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let offset = k * 32u + local.x * 4u;
      let b = vec4<f32>(f32(tileWeight[offset]), f32(tileWeight[offset + 1u]),
        f32(tileWeight[offset + 2u]), f32(tileWeight[offset + 3u]));
${Array.from({ length: rows }, (_, r) => `      sum${r} += f32(tileInput[(local.y * ${rows}u + ${r}u) * 32u + k]) * b;`).join("\n")}
    }
    workgroupBarrier();
  }
  for (var c = 0u; c < 4u; c++) {
    if (col + c < oc) {
${Array.from({ length: rows }, (_, r) => `      if (row + ${r}u < points) { result[(group.z * points + row + ${r}u) * oc + col + c] = Scalar(sum${r}[c]); }`).join("\n")}
    }
  }
}
`

  return {
    ...(!graphAttention && { sparseStem: sparseStem(half, transformer) }),
    conv: convolution(false),
    conv1x1: convolution(true),
    ...(!transformer ? {
      normVector:
        bindings("vec4<Scalar>", "vec4<Scalar>") +
        `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let count = param(0u) / 4u;
  let channels = param(1u) / 4u;
  if (id.x >= count) { return; }
  let c = id.x % channels;
  let i = id.z * count + id.x;
  var x = vec4<f32>(a[i]) * vec4<f32>(b[c]) + vec4<f32>(b[channels + c]);
  switch param(2u) {
    case 1u: { x = max(x, vec4<f32>(0.0)); }
    case 2u: {
      let softplus = max(x, vec4<f32>(0.0)) + log(1.0 + exp(-abs(x)));
      x *= 2.0 / (1.0 + exp(-2.0 * softplus)) - 1.0;
    }
    case 3u: { x /= 1.0 + exp(-x); }
    default: {}
  }
  result[i] = vec4<Scalar>(x);
}
`,
      addVector: add(true),
    } : {}),
    norm:
      bindings() +
      `
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
  let diameter = param(4u);
  if (diameter > 0u) {
    let point = i / param(1u); let sum = point % diameter + point / diameter;
    let radius = (diameter - 1u) / 2u;
    if (sum < radius || sum > 3u * radius) { result[offset + i] = Scalar(0.0); return; }
  }
  var x = f32(a[offset + i]) * f32(b[c]) + f32(b[param(1u) + c]);
  if (param(2u) == 1u) {
    x = max(x, 0.0);
  } else if (param(2u) == 2u) {
    let softplus = max(x, 0.0) + log(1.0 + exp(-abs(x)));
    x *= 2.0 / (1.0 + exp(-2.0 * softplus)) - 1.0;
  }

  if (param(2u) == 3u) { x /= 1.0 + exp(-x); }
  result[offset + i] = Scalar(x);
}
`,
    add: add(false),
    // The spatial input's first channel is the mask.
    pool:
      bindings() +
      `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x; let points = param(0u); let channels = param(1u);
  if (c >= channels) { return; }
  let inputOffset = id.z * points * channels;
  let outputOffset = id.z * channels * 3u;
  var sum = 0.0; var maximum = -1.0;
  for (var i = 0u; i < points; i++) {
    let x = f32(a[inputOffset + i * channels + c]);
    var mask = 1.0;
    if (param(5u) == 0u) { mask = f32(b[(id.z * points + i) * param(4u)]); }
    sum += x; maximum = max(maximum, x + mask - 1.0);
  }
  let mean = sum / f32(param(3u));
  let sizeScale = (sqrt(f32(param(3u))) - 14.0) * 0.1;
  result[outputOffset + c] = Scalar(mean);
  result[outputOffset + channels + c] = Scalar(mean * sizeScale);
  result[outputOffset + 2u * channels + c] = Scalar(select(maximum, mean * (sizeScale * sizeScale - 0.1), param(2u) == 1u));
}
`,
    readout:
      bindings("f32") +
      `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; let points = param(0u);
  if (i < points) { result[id.z * (points + 3u) + i] = f32(a[id.z * points + i]); }
  else if (i < points + 3u) { result[id.z * (points + 3u) + i] = f32(b[id.z * 3u + i - points]); }
}
`,
  }
}

function sparseStem(half, packed) {
  return `${half ? "enable f16;" : ""}
alias Scalar = ${half ? "f16" : "f32"};
alias Vector = vec4<Scalar>;
@group(0) @binding(0) var<storage, read> spatial: array<Scalar>;
@group(0) @binding(1) var<storage, read> weights: array<Vector>;
@group(0) @binding(2) var<storage, read_write> trunk: array<Vector>;
${packed ? "@group(0) @binding(3) var<storage, read> playable: array<u32>;" : ""}
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(${packed ? 4 : 3}) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = params.a.x; let points = params.a.y;
  let inputChannels = params.a.z; let outputVectors = params.a.w / 4u;
  if (id.x >= points * outputVectors) { return; }
  let position = ${packed ? "playable[id.x / outputVectors]" : "id.x / outputVectors"};
  let col = id.x % outputVectors;
  let x = i32(position % size); let y = i32(position / size);
  var sum = vec4<f32>(0.0);
  for (var dy = 0u; dy < 3u; dy++) {
    let iy = y + i32(dy) - 1;
    for (var dx = 0u; dx < 3u; dx++) {
      let ix = x + i32(dx) - 1;
      if (ix >= 0 && ix < i32(size) && iy >= 0 && iy < i32(size)) {
        let feature = ((id.z * size + u32(iy)) * size + u32(ix)) * inputChannels;
        let weight = (dy * 3u + dx) * inputChannels * outputVectors + col;
        if (spatial[feature] != Scalar(0)) { sum += vec4<f32>(weights[weight]); }
        if (spatial[feature + 1u] != Scalar(0)) { sum += vec4<f32>(weights[weight + outputVectors]); }
        if (spatial[feature + 2u] != Scalar(0)) { sum += vec4<f32>(weights[weight + 2u * outputVectors]); }
      }
    }
  }
  trunk[id.z * points * outputVectors + id.x] = Vector(sum);
}
`
}
