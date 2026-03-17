// SPDX-License-Identifier: MIT
// F(2,3): 2-by-2 output tiles from 4-by-4 input tiles.
export function createWinogradKernels() {
  const bindings = (input, output) => `enable f16;
@group(0) @binding(0) var<storage, read> a: array<${input}>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> result: array<${output}>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
fn param(i: u32) -> u32 {
  if (i < 4u) { return params.a[i]; }
  return params.b[i - 4u];
}
`;
  return {
  winogradInput: bindings("f16", "f16") + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ic = param(2u); let tiles = param(4u) * param(5u);
  if (id.x >= tiles * ic) { return; }
  let tile = id.x / ic; let channel = id.x % ic;
  let ox = i32(tile % param(4u)) * 2 - 1;
  let oy = i32(tile / param(4u)) * 2 - 1;
  var d: array<f32, 16>; var t: array<f32, 16>;
  for (var y = 0u; y < 4u; y++) {
    for (var x = 0u; x < 4u; x++) {
      let ix = ox + i32(x); let iy = oy + i32(y);
      if (ix >= 0 && iy >= 0 && ix < i32(param(0u)) && iy < i32(param(1u))) {
        d[y * 4u + x] = f32(a[((id.z * param(1u) + u32(iy)) * param(0u) + u32(ix)) * ic + channel]);
      }
    }
  }
  for (var x = 0u; x < 4u; x++) {
    t[x] = d[x] - d[8u + x];
    t[4u + x] = d[4u + x] + d[8u + x];
    t[8u + x] = d[8u + x] - d[4u + x];
    t[12u + x] = d[4u + x] - d[12u + x];
  }
  for (var y = 0u; y < 4u; y++) {
    let i = y * 4u;
    let v = vec4<f32>(t[i] - t[i + 2u], t[i + 1u] + t[i + 2u],
      t[i + 2u] - t[i + 1u], t[i + 1u] - t[i + 3u]);
    for (var x = 0u; x < 4u; x++) {
      result[((i + x) * param(6u) * tiles + id.z * tiles + tile) * ic + channel] = f16(v[x]);
    }
  }
}
`,
  winogradMultiply: bindings("f16", "f32") + `
var<workgroup> tileInput: array<f16, 1024>;
var<workgroup> tileWeight: array<f16, 1024>;
@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let rows = param(4u) * param(5u) * param(6u);
  let ic = param(2u); let oc = param(3u);
  let row = group.y * 32u + local.y * 4u; let col = group.x * 32u + local.x * 4u;
  let lane = local.y * 8u + local.x;
  var sum0 = vec4<f32>(0.0); var sum1 = vec4<f32>(0.0);
  var sum2 = vec4<f32>(0.0); var sum3 = vec4<f32>(0.0);
  for (var base = 0u; base < ic; base += 32u) {
    for (var i = lane; i < 1024u; i += 64u) {
      let r = group.y * 32u + i / 32u; let k = base + i % 32u;
      var av = f16(0.0);
      if (r < rows && k < ic) { av = a[(group.z * rows + r) * ic + k]; }
      tileInput[i] = av;
      let wk = base + i / 32u; let c = group.x * 32u + i % 32u;
      var bv = f16(0.0);
      if (wk < ic && c < oc) { bv = b[(group.z * ic + wk) * oc + c]; }
      tileWeight[i] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let offset = k * 32u + local.x * 4u;
      let weight = vec4<f32>(f32(tileWeight[offset]), f32(tileWeight[offset + 1u]),
        f32(tileWeight[offset + 2u]), f32(tileWeight[offset + 3u]));
      sum0 += f32(tileInput[(local.y * 4u) * 32u + k]) * weight;
      sum1 += f32(tileInput[(local.y * 4u + 1u) * 32u + k]) * weight;
      sum2 += f32(tileInput[(local.y * 4u + 2u) * 32u + k]) * weight;
      sum3 += f32(tileInput[(local.y * 4u + 3u) * 32u + k]) * weight;
    }
    workgroupBarrier();
  }
  for (var c = 0u; c < 4u; c++) {
    if (col + c < oc) {
      if (row < rows) { result[(group.z * rows + row) * oc + col + c] = sum0[c]; }
      if (row + 1u < rows) { result[(group.z * rows + row + 1u) * oc + col + c] = sum1[c]; }
      if (row + 2u < rows) { result[(group.z * rows + row + 2u) * oc + col + c] = sum2[c]; }
      if (row + 3u < rows) { result[(group.z * rows + row + 3u) * oc + col + c] = sum3[c]; }
    }
  }
}
`,
  winogradOutput: bindings("f32", "f16") + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let oc = param(3u); let tiles = param(4u) * param(5u);
  if (id.x >= tiles * oc) { return; }
  let tile = id.x / oc; let channel = id.x % oc;
  let ox = tile % param(4u) * 2u; let oy = tile / param(4u) * 2u;
  let stride = param(6u) * tiles * oc; let index = (id.z * tiles + tile) * oc + channel;
  var d: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { d[i] = a[i * stride + index]; }
  var t: array<f32, 8>;
  for (var x = 0u; x < 4u; x++) {
    t[x] = d[x] + d[4u + x] + d[8u + x];
    t[4u + x] = d[4u + x] - d[8u + x] - d[12u + x];
  }
  for (var y = 0u; y < 2u; y++) {
    let i = y * 4u;
    let v = vec2<f32>(t[i] + t[i + 1u] + t[i + 2u], t[i + 1u] - t[i + 2u] - t[i + 3u]);
    for (var x = 0u; x < 2u; x++) {
      if (ox + x < param(0u) && oy + y < param(1u)) {
        result[((id.z * param(1u) + oy + y) * param(0u) + ox + x) * oc + channel] = f16(v[x]);
      }
    }
  }
}
`,
  };
}

export function transformWinogradWeights(layer, weights) {
  const G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]];
  for (let i = 0; i < layer.input; i++) {
    for (let o = 0; o < layer.output; o++) {
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
          let value = 0;
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              value += G[y][ky] * layer.weights[((ky * 3 + kx) * layer.input + i) * layer.output + o] * G[x][kx];
            }
          }
          weights[((y * 4 + x) * layer.input + i) * layer.output + o] = value;
        }
      }
    }
  }
  return weights;
}
