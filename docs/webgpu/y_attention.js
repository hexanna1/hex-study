import { FUSED_Y_MAX_POINTS } from "./fused_attention.js"
import { kernel, READ, WRITE } from "./kernel.js"

export const ONLINE_Y_MIN_POINTS = FUSED_Y_MAX_POINTS + 1
export const ONLINE_Y_QUERY_ROWS = 16
export const ONLINE_Y_KEYS_PER_TILE = 32

export function rotaryPhases(size, points, pairs, frequencies, pointMajor) {
  const table = new Float32Array(points.length * pairs * 2)
  for (let pair = 0; pair < pairs; pair++) {
    const fx = frequencies[pair * 2]
    const fy = frequencies[pair * 2 + 1]
    for (let row = 0; row < points.length; row++) {
      const coordinate = points[row]
      const angle = Math.fround(Math.fround((coordinate % size) * fx) +
        Math.fround(Math.floor(coordinate / size) * fy))
      const offset = (pointMajor ? row * pairs + pair : pair * points.length + row) * 2
      table[offset] = Math.cos(angle)
      table[offset + 1] = Math.sin(angle)
    }
  }
  return table
}

export function createOnlineYAttention(half, { graphBias, indexed = false }) {
  const rows = ONLINE_Y_QUERY_ROWS
  const lanes = rows * 8
  const scalar = half ? "f16" : "f32"
  const prefix = half ? "enable f16;" : ""
  const yRotateQK = half || graphBias ? null : `
alias Pair = vec2<f32>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(0) var<storage, read> rope: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> qkv: array<Pair>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let points = params.a.x;
  let span = params.a.y * 16u;
  if (id.x >= points * span * 2u) { return; }
  let point = id.x / (span * 2u);
  let pair = id.x % span;
  let projection = id.x / span % 2u;
  let index = (id.z * points + point) * span * 3u + projection * span + pair;
  let phase = rope[pair * points + point];
  let xy = vec2<f32>(qkv[index]);
  qkv[index] = Pair(xy * phase.x + vec2<f32>(-xy.y, xy.x) * phase.y);
}
`
  // Register-resident queries and unrolled values improve Y but can slow graph-biased attention.
  const unroll = !graphBias
  const qkvRow = (point) => indexed ? `patternByPoint[batch * points + ${point}]` : `(batch * points + ${point})`
  const repeat = (emit) => Array.from({ length: 8 }, (_, i) => emit(i)).join("\n    ")
  const yAttentionOnline = `${prefix}
alias Scalar = ${scalar};
alias Vector = vec4<Scalar>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(0) var<storage, read> qkv: array<Vector>;
${graphBias ? "@group(0) @binding(1) var<storage, read> bias: array<f32>;" : ""}
@group(0) @binding(${graphBias ? 2 : 1}) var<storage, read_write> output: array<Vector>;
${indexed ? "@group(0) @binding(3) var<storage, read> patternByPoint: array<u32>;" : ""}
${graphBias ? `@group(0) @binding(${3 + Number(indexed)}) var<storage, read> biasIndices: array<u32>;` : ""}
@group(0) @binding(${2 + Number(graphBias) * 2 + Number(indexed)}) var<uniform> params: Params;
${unroll ? "" : `var<workgroup> queries: array<Vector, ${rows * 8}>;`}
var<workgroup> keys: array<Vector, 256>;
var<workgroup> values: array<Vector, 256>;
var<workgroup> weights: array<vec4<f32>, ${lanes}>;
var<workgroup> maxima: array<f32, ${lanes}>;
var<workgroup> sums: array<f32, ${lanes}>;

@compute @workgroup_size(8, ${rows})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x;
  let heads = params.a.y;
  let head = group.y;
  let batch = group.z;
  let firstQuery = group.x * ${rows}u;
  let query = firstQuery + local.y;
  let rowLanes = local.y * 8u;
  ${unroll ? `let queryOffset = ((batch * points + query) * heads * 3u + head) * 8u;
  ${Array.from({ length: 8 }, (_, dim) => `var q${dim} = vec4<f32>(0);
  if (query < points) { q${dim} = vec4<f32>(qkv[queryOffset + ${dim}u]); }`).join("\n  ")}` : `for (var i = lane; i < ${rows * 8}u; i += ${lanes}u) {
    let point = firstQuery + i / 8u;
    var x = Vector(0);
    if (point < points) { x = qkv[(${qkvRow("point")} * heads * 3u + head) * 8u + i % 8u]; }
    queries[i] = x;
  }`}
  var maximum = -1e30;
  var normalizer = 0.0;
  var weighted = vec4<f32>(0);
  for (var base = 0u; base < points; base += ${ONLINE_Y_KEYS_PER_TILE}u) {
    // Load four neighboring keys as vectors and transpose their components into the tile.
    for (var i = lane; i < 64u; i += ${lanes}u) {
      let dim = i / 8u;
      let firstKey = base + i % 8u * 4u;
      let pointStride = heads * 24u;
      var a = Vector(0);
      var b = Vector(0);
      var c = Vector(0);
      var d = Vector(0);
      ${indexed ? `let keyHead = (heads + head) * 8u + dim;
      if (firstKey < points) { a = qkv[${qkvRow("firstKey")} * pointStride + keyHead]; }
      if (firstKey + 1u < points) { b = qkv[${qkvRow("firstKey + 1u")} * pointStride + keyHead]; }
      if (firstKey + 2u < points) { c = qkv[${qkvRow("firstKey + 2u")} * pointStride + keyHead]; }
      if (firstKey + 3u < points) { d = qkv[${qkvRow("firstKey + 3u")} * pointStride + keyHead]; }`
      : `let offset = ((batch * points + firstKey) * heads * 3u + heads + head) * 8u + dim;
      if (firstKey < points) { a = qkv[offset]; }
      if (firstKey + 1u < points) { b = qkv[offset + pointStride]; }
      if (firstKey + 2u < points) { c = qkv[offset + pointStride * 2u]; }
      if (firstKey + 3u < points) { d = qkv[offset + pointStride * 3u]; }`}
      let keyIndex = dim * 32u + i % 8u;
      keys[keyIndex] = Vector(a.x, b.x, c.x, d.x);
      keys[keyIndex + 8u] = Vector(a.y, b.y, c.y, d.y);
      keys[keyIndex + 16u] = Vector(a.z, b.z, c.z, d.z);
      keys[keyIndex + 24u] = Vector(a.w, b.w, c.w, d.w);
    }
    for (var i = lane; i < 256u; i += ${lanes}u) {
      let key = base + i / 8u;
      var v = Vector(0);
      if (key < points) { v = qkv[(${qkvRow("key")} * heads * 3u + heads * 2u + head) * 8u + i % 8u]; }
      values[i] = v;
    }
    workgroupBarrier();
    var dot = vec4<f32>(0);
    ${unroll ? repeat((dim) => `let k${dim} = ${dim * 32}u + local.x;
    dot += q${dim}.x * vec4<f32>(keys[k${dim}]);
    dot += q${dim}.y * vec4<f32>(keys[k${dim} + 8u]);
    dot += q${dim}.z * vec4<f32>(keys[k${dim} + 16u]);
    dot += q${dim}.w * vec4<f32>(keys[k${dim} + 24u]);`) : `for (var dim = 0u; dim < 8u; dim++) {
      let q = vec4<f32>(queries[local.y * 8u + dim]);
      let k = dim * 32u + local.x;
      dot += q.x * vec4<f32>(keys[k]);
      dot += q.y * vec4<f32>(keys[k + 8u]);
      dot += q.z * vec4<f32>(keys[k + 16u]);
      dot += q.w * vec4<f32>(keys[k + 24u]);
    }`}
    let keyPositions = vec4<u32>(base + local.x * 4u) + vec4<u32>(0, 1, 2, 3);
    let valid = keyPositions < vec4<u32>(points);
    ${graphBias ? `let index = query * params.a.w + keyPositions.x / 2u;
    let pair0 = biasIndices[index];
    let pair1 = biasIndices[index + 1u];
    let headStart = head * params.a.z;
    let positionalBias = vec4<f32>(
      bias[headStart + (pair0 & 65535u)], bias[headStart + (pair0 >> 16u)],
      bias[headStart + (pair1 & 65535u)], bias[headStart + (pair1 >> 16u)],
    );` : ""}
    let scores = select(vec4<f32>(-1e30), dot * 0.1767766952966369${graphBias ? " + positionalBias" : ""}, valid);
    maxima[lane] = max(max(scores.x, scores.y), max(scores.z, scores.w));
    workgroupBarrier();
    var nextMax = maximum;
    for (var j = 0u; j < 8u; j++) { nextMax = max(nextMax, maxima[rowLanes + j]); }
    let w = select(vec4<f32>(0), exp(scores - vec4<f32>(nextMax)), valid);
    weights[lane] = w;
    sums[lane] = ((w.x + w.y) + w.z) + w.w;
    workgroupBarrier();
    var sum = 0.0;
    for (var j = 0u; j < 8u; j++) { sum += sums[rowLanes + j]; }
    // Express both the old and new tile contributions relative to the new maximum.
    let rescale = exp(maximum - nextMax);
    maximum = nextMax;
    normalizer = normalizer * rescale + sum;
    weighted *= rescale;
    ${unroll ? repeat((j) => `let w${j} = weights[rowLanes + ${j}u];
    let v${j} = ${j * 32}u + local.x;
    weighted += w${j}.x * vec4<f32>(values[v${j}]);
    weighted += w${j}.y * vec4<f32>(values[v${j} + 8u]);
    weighted += w${j}.z * vec4<f32>(values[v${j} + 16u]);
    weighted += w${j}.w * vec4<f32>(values[v${j} + 24u]);`) : `for (var j = 0u; j < 8u; j++) {
      let tileWeights = weights[rowLanes + j];
      let firstValue = j * 32u + local.x;
      weighted += tileWeights.x * vec4<f32>(values[firstValue]);
      weighted += tileWeights.y * vec4<f32>(values[firstValue + 8u]);
      weighted += tileWeights.z * vec4<f32>(values[firstValue + 16u]);
      weighted += tileWeights.w * vec4<f32>(values[firstValue + 24u]);
    }`}
    workgroupBarrier();
  }
  if (query < points) {
    output[((batch * points + query) * heads + head) * 8u + local.x] = Vector(weighted * (1.0 / normalizer));
  }
}
`
  const buffers = graphBias ? [READ, READ, WRITE] : [READ, WRITE]
  if (indexed) buffers.push(READ)
  if (graphBias) buffers.push(READ)
  return {
    ...(yRotateQK && { yRotateQK: kernel(yRotateQK, [READ, WRITE]) }),
    yAttentionOnline: kernel(yAttentionOnline, buffers),
  }
}

export function createYAttentionKeys() {
  return kernel(`enable f16;
alias Scalar = f16;
@group(0) @binding(0) var<storage, read> qkv: array<Scalar>;
@group(0) @binding(1) var<storage, read> rope: array<f32>;
@group(0) @binding(2) var<storage, read_write> transposed: array<vec4<Scalar>>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let points = params.a.x; let heads = params.a.y; let columns = params.a.z;
  if (id.x >= columns * heads * 32u) { return; }
  let column = id.x % columns;
  let dim = id.x / columns % 32u;
  let head = id.x / (columns * 32u);
  var row = vec4<Scalar>(0.0);
  for (var p = 0u; p < 4u; p++) {
    let point = column * 4u + p;
    if (point < points) {
      let index = (id.z * points + point) * heads * 96u + heads * 32u + head * 32u + dim;
      let phase = ((head * 16u + dim / 2u) * points + point) * 2u;
      let x = f32(qkv[index]); let y = f32(qkv[(index - dim) + (dim ^ 1u)]);
      let sign = select(1.0, -1.0, dim % 2u == 0u);
      row[p] = Scalar(x * rope[phase] + sign * y * rope[phase + 1u]);
    }
  }
  transposed[id.z * columns * heads * 32u + id.x] = row;
}
`)
}
