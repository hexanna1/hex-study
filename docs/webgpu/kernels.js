// SPDX-License-Identifier: MIT
// WebGPU implementations of the operations in KataHex's neural-network backend.
import { kernel, READ, WRITE } from "./kernel.js"

export const activationCases = (activation) => `  switch ${activation} {
    case 1u: { x = max(x, vec4<f32>(0.0)); }
    case 2u: {
      let softplus = max(x, vec4<f32>(0.0)) + log(1.0 + exp(-abs(x)));
      x *= 2.0 / (1.0 + exp(-2.0 * softplus)) - 1.0;
    }
    case 3u: { x /= 1.0 + exp(-x); }
    default: {}
  }`

export function createKernels(half, transformer = false, graphAttention = false) {
  const scalar = half ? "f16" : "f32"
  const bindings = (output = scalar, input = "Scalar", binary = true) => `${half ? "enable f16;" : ""}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> a: array<${input}>;
${binary ? `@group(0) @binding(1) var<storage, read> b: array<${input}>;` : ""}
@group(0) @binding(${binary ? 2 : 1}) var<storage, read_write> result: array<${output}>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(${binary ? 3 : 2}) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`

  const add = bindings("vec4<Scalar>", "vec4<Scalar>") + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let count = param(0u) / 4u;
  let other = param(1u) / 4u;
  if (id.x < count) {
    let i = id.z * count + id.x;
    result[i] = vec4<Scalar>(vec4<f32>(a[i]) + vec4<f32>(b[id.z * other + id.x % other]));
  }
}
`

  const rows = transformer ? 2 : 4
  const conv1x1 = bindings() + `
var<workgroup> tileInput: array<Scalar, 1024>;
var<workgroup> tileWeight: array<Scalar, 1024>;
@compute @workgroup_size(8, ${32 / rows})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let points = param(0u); let ic = param(1u); let oc = param(2u);
  let row = group.y * 32u + local.y * ${rows}u; let col = group.x * 32u + local.x * 4u;
  let lane = local.y * 8u + local.x;
${Array.from({ length: rows }, (_, r) => `  var sum${r} = vec4<f32>(0.0);`).join("\n")}
  for (var base = 0u; base < ic; base += 32u) {
    for (var i = lane; i < 1024u; i += ${256 / rows}u) {
      let r = group.y * 32u + i / 32u; let k = base + i % 32u;
      var av = Scalar(0.0);
      if (r < points && k < ic) {
        av = a[(group.z * points + r) * ic + k];
      }
      tileInput[i] = av;
      let wk = base + i / 32u; let c = group.x * 32u + i % 32u;
      var bv = Scalar(0.0);
      if (wk < ic && c < oc) {
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
    ...(!graphAttention && {
      sparseStem: kernel(sparseStem(half, transformer), transformer ? [READ, READ, WRITE, READ] : undefined),
    }),
    conv1x1: kernel(conv1x1),
    normVector:
      kernel(bindings("vec4<Scalar>", "vec4<Scalar>") +
      `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let count = param(0u) / 4u;
  let channels = param(1u) / 4u;
  if (id.x >= count) { return; }
  let c = id.x % channels;
  let i = id.z * count + id.x;
  var x = vec4<f32>(a[i]) * vec4<f32>(b[c]) + vec4<f32>(b[channels + c]);
${activationCases("param(2u)")}
  result[i] = vec4<Scalar>(x);
}
`),
    add: kernel(add),
    pool:
      kernel(bindings(scalar, "Scalar", false) +
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
    sum += x; maximum = max(maximum, x);
  }
  let mean = sum / f32(points);
  let sizeScale = (sqrt(f32(points)) - 14.0) * 0.1;
  result[outputOffset + c] = Scalar(mean);
  result[outputOffset + channels + c] = Scalar(mean * sizeScale);
  result[outputOffset + 2u * channels + c] = Scalar(select(maximum, mean * (sizeScale * sizeScale - 0.1), param(2u) == 1u));
}
`, [READ, WRITE]),
    readout: kernel(`${half ? "enable f16;" : ""}
alias Scalar = ${scalar};
@group(0) @binding(0) var<storage, read> policy: array<Scalar>;
@group(0) @binding(1) var<storage, read> hidden: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<Scalar>;
@group(0) @binding(4) var<storage, read> affine: array<Scalar>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(5) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; let points = params.a.x; let channels = params.a.y;
  let offset = id.z * (points + 3u) + i;
  if (i < points) {
    result[offset] = f32(policy[id.z * points + i]);
  } else if (i < points + 3u) {
    let channel = i - points;
    var sum = 0.0;
    for (var c = 0u; c < channels; c++) {
      sum += f32(hidden[id.z * channels + c]) * f32(projection[c * 3u + channel]);
    }
    // The FP16 path rounds the projection before the affine.
    let rounded = f32(Scalar(sum));
    result[offset] = f32(Scalar(rounded * f32(affine[channel]) + f32(affine[3u + channel])));
  }
}
`, [READ, READ, WRITE, READ, READ]),
  }
}

function sparseStem(half, packed) {
  return `${half ? "enable f16;" : ""}
alias Scalar = ${half ? "f16" : "f32"};
alias Vector = vec4<Scalar>;
@group(0) @binding(0) var<storage, read> spatial: array<u32>;
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
        let stone = spatial[(id.z * size + u32(iy)) * size + u32(ix)];
        let weight = (dy * 3u + dx) * inputChannels * outputVectors + col;
        if (stone != 3u) { sum += vec4<f32>(weights[weight]); }
        if (stone == 1u) { sum += vec4<f32>(weights[weight + outputVectors]); }
        if (stone == 2u) { sum += vec4<f32>(weights[weight + 2u * outputVectors]); }
      }
    }
  }
  trunk[id.z * points * outputVectors + id.x] = Vector(sum);
}
`
}
