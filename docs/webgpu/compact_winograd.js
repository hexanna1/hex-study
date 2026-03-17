// SPDX-License-Identifier: MIT
// Input transform follows KataGo's F(2,3) kernels in cpp/neuralnet/openclkernels.cpp. See LICENSE.
import { winogradOutputKernels } from "./winograd_kernels.js"
import { kernel, READ, WRITE } from "./kernel.js"

export const COMPACT_OUTPUT_TILE = 2
export const COMPACT_COEFFICIENTS = 16

export function unpackSignedBytes(word) {
  return `bitcast<vec4<i32>>(vec4<u32>(${word}) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u)`
}

const lines = (count, emit) => Array.from({ length: count }, (_, i) => emit(i)).join("\n")
const argumentsFor = (count, emit) => Array.from({ length: count }, (_, i) => emit(i)).join(", ")

const bindings = (weighted) => `
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
${weighted ? "@group(0) @binding(1) var<storage, read> b: array<u32>;" : ""}
@group(0) @binding(${weighted ? 2 : 1}) var<storage, read_write> result: array<vec4<f32>>;
struct Params { shape: vec4<u32>, tiles: vec4<u32> }
@group(0) @binding(${weighted ? 3 : 2}) var<uniform> params: Params;
`

function multiply(winograd, tileRows, pair = false) {
  // Two planes share one decode and split the input-channel tile in half.
  // The total workgroup storage is the same as for a single plane.
  const planes = (emit) => lines(pair ? 2 : 1, emit)
  const perPlane = pair ? 4 : 8
  const filterTileVectors = perPlane * 32
  const inputTileVectors = tileRows * perPlane
  const guardFilterLoad = inputTileVectors > filterTileVectors
  const threads = tileRows * 2
  const store = (p, r) => {
    const index = `(start${p} + row + ${r}u) * oc + col`
    if (winograd) return `      result[${index}] = sum${p}_${r} * scale;`
    return `      let index${p} = ${index};
      let projected${p} = sum${p}_${r} * scale;
      if (params.tiles.w != 0u) {
        result[index${p}] = projected${p} + result[index${p}];
      } else {
        result[index${p}] = projected${p};
      }`
  }
  return bindings(true) + `
fn codes(index: u32) -> vec4<i32> {
  let word = b[index];
  return ${unpackSignedBytes("word")};
}
${pair ? `
fn filterPair(index: u32, stride: u32, inner: bool) -> array<vec4<i32>, 2> {
  let left = codes(index);
  let right = codes(index + 2u * stride);
  if (!inner) { return array<vec4<i32>, 2>(2 * left, 2 * right); }
  let outer = left + right;
  let center = codes(index + stride);
  return array<vec4<i32>, 2>(outer + center, outer - center);
}
fn loadFilterPair(index: u32, stride: u32, y: u32, inner: bool) -> array<vec4<f32>, 2> {
  let rowStride = 3u * stride;
  if (y == 0u || y == 3u) {
    let row = filterPair(index + select(0u, 2u * rowStride, y == 3u), stride, inner);
    return array<vec4<f32>, 2>(vec4<f32>(2 * row[0]), vec4<f32>(2 * row[1]));
  }
  let top = filterPair(index, stride, inner);
  let center = filterPair(index + rowStride, stride, inner);
  let bottom = filterPair(index + 2u * rowStride, stride, inner);
  return array<vec4<f32>, 2>(
    vec4<f32>(top[0] + bottom[0] + select(-center[0], center[0], y == 1u)),
    vec4<f32>(top[1] + bottom[1] + select(-center[1], center[1], y == 1u)),
  );
}
` : winograd ? `
fn filterRow(index: u32, stride: u32, plane: u32) -> vec4<i32> {
  if (plane == 0u) { return 2 * codes(index); }
  if (plane == 3u) { return 2 * codes(index + 2u * stride); }
  let outer = codes(index) + codes(index + 2u * stride);
  let center = codes(index + stride);
  return outer + select(-center, center, plane == 1u);
}

fn loadFilter(index: u32, stride: u32, plane: u32) -> vec4<f32> {
  let x = plane % 4u;
  let y = plane / 4u;
  let rowStride = 3u * stride;
  if (y == 0u) { return vec4<f32>(2 * filterRow(index, stride, x)); }
  if (y == 3u) { return vec4<f32>(2 * filterRow(index + 2u * rowStride, stride, x)); }
  let outer = filterRow(index, stride, x) + filterRow(index + 2u * rowStride, stride, x);
  let center = filterRow(index + rowStride, stride, x);
  return vec4<f32>(outer + select(-center, center, y == 1u));
}
` : ""}
${planes((p) => `var<workgroup> inputTile${p}: array<vec4<f32>, ${inputTileVectors}>;
var<workgroup> filterTile${p}: array<vec4<f32>, ${filterTileVectors}>;`)}

@compute @workgroup_size(8, ${tileRows / 4})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let rows = ${winograd ? "params.tiles.x * params.tiles.y" : "params.shape.x * params.shape.y"} * params.tiles.z;
  let ic = params.shape.z / 4u;
  let oc = params.shape.w / 4u;
  let row = group.y * ${tileRows}u + local.y * 4u;
  let col = group.x * 8u + local.x;
  let lane = local.y * 8u + local.x;
  ${pair ? `// Four middle pairs and four edge pairs cover the 16 coefficient planes.
  let y = group.z % 4u;
  let middle = group.z < 4u;
  let start0 = (4u * y + select(0u, 1u, middle)) * rows;
  let start1 = (4u * y + select(3u, 2u, middle)) * rows;` : "let start0 = group.z * rows;"}
  ${planes((p) => lines(4, (r) => `var sum${p}_${r} = vec4<f32>(0.0);`))}
  for (var base = 0u; base < ic; base += ${perPlane}u) {
    for (var i = lane; i < ${inputTileVectors}u; i += ${threads}u) {
      let r = group.y * ${tileRows}u + i / ${perPlane}u;
      let k = base + i % ${perPlane}u;
${planes((p) => `      var value${p} = vec4<f32>(0.0);
      if (r < rows && k < ic) { value${p} = a[(start${p} + r) * ic + k]; }
      inputTile${p}[i] = value${p};`)}
      ${guardFilterLoad ? `if (i < ${filterTileVectors}u) {` : ""}
      let wk = base * 4u + i / 8u;
      let wc = group.x * 8u + i % 8u;
${planes((p) => `      var weight${p} = vec4<f32>(0.0);`)}
      if (wk < params.shape.z && wc < oc) {
        let index = wk * oc + wc;
        ${pair ? `let pair = loadFilterPair(index, params.shape.z * oc, y, middle);
        weight0 = pair[0];
        weight1 = pair[1];` : `weight0 = ${winograd ? "loadFilter(index, params.shape.z * oc, group.z)" : "vec4<f32>(codes(index))"};`}
      }
${planes((p) => `      filterTile${p}[i] = weight${p};`)}
      ${guardFilterLoad ? "}" : ""}
    }
    workgroupBarrier();
    for (var k = 0u; k < ${perPlane}u; k++) {
      let weight = k * 32u + local.x;
${planes((p) => lines(4, (i) => `      let w${p}_${i} = filterTile${p}[weight + ${i * 8}u];`))}
${planes((p) => lines(4, (r) => `      let v${p}_${r} = inputTile${p}[(local.y * 4u + ${r}u) * ${perPlane}u + k];
${lines(4, (i) => `      sum${p}_${r} += v${p}_${r}.${"xyzw"[i]} * w${p}_${i};`)}`))}
    }
    workgroupBarrier();
  }
  if (col < oc) {
    let offset = ${winograd ? "9u * " : ""}params.shape.z * oc + col * 4u;
    let scale = bitcast<vec4<f32>>(vec4<u32>(b[offset], b[offset + 1u], b[offset + 2u], b[offset + 3u]));
${lines(4, (r) => `    if (row + ${r}u < rows) {
${planes((p) => store(p, r))}
    }`)}
  }
}
`
}

export function createCompactWinogradKernels(tileRows, pairPlanes) {
  return {
    compactWinogradInput: kernel(bindings(false) + `
fn load(x: i32, y: i32, batch: u32, channel: u32, ic: u32) -> vec4<f32> {
  let width = params.shape.x;
  let height = params.shape.y;
  if (x < 0 || y < 0 || x >= i32(width) || y >= i32(height)) { return vec4<f32>(0.0); }
  return a[((batch * height + u32(y)) * width + u32(x)) * ic + channel];
}

fn transform(d0: vec4<f32>, d1: vec4<f32>, d2: vec4<f32>, d3: vec4<f32>) -> array<vec4<f32>, 4> {
  return array<vec4<f32>, 4>(d0 - d2, d1 + d2, d2 - d1, d1 - d3);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ic = params.shape.z / 4u;
  let tiles = params.tiles.x * params.tiles.y;
  if (id.x >= tiles * ic) { return; }
  let tile = id.x / ic;
  let channel = id.x % ic;
  let ox = i32(tile % params.tiles.x) * 2 - 1;
  let oy = i32(tile / params.tiles.x) * 2 - 1;
  let stride = params.tiles.z * tiles * ic;
  let base = (id.z * tiles + tile) * ic + channel;
${lines(4, (x) => `  let col${x} = transform(${argumentsFor(4, (y) => `load(ox + ${x}, oy + ${y}, id.z, channel, ic)`)});`)}
${lines(4, (y) => `  let row${y} = transform(${argumentsFor(4, (x) => `col${x}[${y}]`)});
${lines(4, (x) => `  result[${y * 4 + x}u * stride + base] = row${y}[${x}];`)}`)}
}
`, [READ, WRITE]),
    compactWinogradMultiply: kernel(multiply(true, tileRows, pairPlanes)),
    compactPointwise: kernel(multiply(false, tileRows)),
    ...winogradOutputKernels("compactWinogradOutput", false, COMPACT_OUTPUT_TILE, false),
  }
}
