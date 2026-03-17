export const COMPACT_OUTPUT_TILE = 2
export const COMPACT_COEFFICIENTS = 16

const lines = (count, emit) => Array.from({ length: count }, (_, i) => emit(i)).join("\n")
const argumentsFor = (count, emit) => Array.from({ length: count }, (_, i) => emit(i)).join(", ")

const bindings = `
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;
struct Params { shape: vec4<u32>, tiles: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
`

function multiply(winograd) {
  return bindings + `
fn codes(index: u32) -> vec4<i32> {
  let word = b[index];
  return bitcast<vec4<i32>>(vec4<u32>(word) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u);
}
${winograd ? `
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
var<workgroup> inputTile: array<vec4<f32>, 256>;
var<workgroup> filterTile: array<vec4<f32>, 256>;

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let rows = ${winograd ? "params.tiles.x * params.tiles.y" : "params.shape.x * params.shape.y"} * params.tiles.z;
  let ic = params.shape.z / 4u;
  let oc = params.shape.w / 4u;
  let row = group.y * 32u + local.y * 4u;
  let col = group.x * 8u + local.x;
  let lane = local.y * 8u + local.x;
  let start = group.z * rows;
  ${lines(4, (r) => `var sum${r} = vec4<f32>(0.0);`)}
  for (var base = 0u; base < ic; base += 8u) {
    for (var i = lane; i < 256u; i += 64u) {
      let r = group.y * 32u + i / 8u;
      let k = base + i % 8u;
      var value = vec4<f32>(0.0);
      if (r < rows && k < ic) { value = a[(start + r) * ic + k]; }
      inputTile[i] = value;
      let wk = base * 4u + i / 8u;
      let wc = group.x * 8u + i % 8u;
      var weight = vec4<f32>(0.0);
      if (wk < params.shape.z && wc < oc) {
        let index = wk * oc + wc;
        weight = ${winograd ? "loadFilter(index, params.shape.z * oc, group.z)" : "vec4<f32>(codes(index))"};
      }
      filterTile[i] = weight;
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k++) {
      let weight = k * 32u + local.x;
${lines(4, (i) => `      let w${i} = filterTile[weight + ${i * 8}u];`)}
${lines(4, (r) => `      let v${r} = inputTile[(local.y * 4u + ${r}u) * 8u + k];
${lines(4, (i) => `      sum${r} += v${r}.${"xyzw"[i]} * w${i};`)}`)}
    }
    workgroupBarrier();
  }
  if (col < oc) {
    let offset = ${winograd ? "9u * " : ""}params.shape.z * oc + col * 4u;
    let scale = bitcast<vec4<f32>>(vec4<u32>(b[offset], b[offset + 1u], b[offset + 2u], b[offset + 3u]));
${lines(4, (r) => `    if (row + ${r}u < rows) { result[(start + row + ${r}u) * oc + col] = sum${r} * scale; }`)}
  }
}
`
}

export function createCompactWinogradKernels() {
  return {
    compactWinogradInput: bindings + `
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
`,
    compactWinogradMultiply: multiply(true),
    compactPointwise: multiply(false),
    compactWinogradOutput: bindings + `
fn transform(d0: vec4<f32>, d1: vec4<f32>, d2: vec4<f32>, d3: vec4<f32>) -> array<vec4<f32>, 2> {
  return array<vec4<f32>, 2>(d0 + d1 + d2, d1 - d2 - d3);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let oc = params.shape.w / 4u;
  let tiles = params.tiles.x * params.tiles.y;
  if (id.x >= tiles * oc) { return; }
  let tile = id.x / oc;
  let channel = id.x % oc;
  let ox = tile % params.tiles.x * 2u;
  let oy = tile / params.tiles.x * 2u;
  let stride = params.tiles.z * tiles * oc;
  let base = (id.z * tiles + tile) * oc + channel;
${lines(4, (x) => `  let col${x} = transform(${argumentsFor(4, (y) => `a[${y * 4 + x}u * stride + base]`)});`)}
${lines(2, (y) => `  let row${y} = transform(${argumentsFor(4, (x) => `col${x}[${y}]`)});
${lines(2, (x) => `  if (ox + ${x}u < params.shape.x && oy + ${y}u < params.shape.y) {
    result[((id.z * params.shape.y + oy + ${y}u) * params.shape.x + ox + ${x}u) * oc + channel] = row${y}[${x}];
  }`)}`)}
}
`,
  }
}
