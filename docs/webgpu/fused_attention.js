import { kernel, READ, WRITE } from "./kernel.js"

export const FUSED_ATTENTION_QUERY_ROWS = 8
export const FUSED_Y_MAX_POINTS = 78
const FUSED_YAVALATH_MAX_POINTS = 61

export function createFusedAttention(half, { yavalath = false } = {}) {
  const points = yavalath ? FUSED_YAVALATH_MAX_POINTS : FUSED_Y_MAX_POINTS
  const rows = FUSED_ATTENTION_QUERY_ROWS
  const querySize = rows * 32
  const threads = rows * 8
  const transposedKeys = half && !yavalath
  const scalar = half ? "f16" : "f32"
  return kernel(`${half ? "enable f16;" : ""}
alias S = ${scalar};
alias V = vec4<S>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(0) var<storage, read> input: array<V>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<V>;
${transposedKeys ? "@group(0) @binding(3) var<storage, read> transposed: array<V>;" : ""}
@group(0) @binding(${transposedKeys ? 4 : 3}) var<uniform> params: Params;
var<workgroup> queries: array<S, ${querySize}>;
var<workgroup> tile: array<V, 256>;
var<workgroup> scores: array<f32, ${rows * points}>;
var<workgroup> reductions: array<f32, ${rows * 64}>;

fn rotated(point: u32, head: u32, dim: u32, projection: u32, batch: u32) -> S {
  let points = params.a.x; let heads = params.a.y;
  let offset = (batch * points + point) * heads * 24u + projection * heads * 8u + head * 8u;
  let values = input[offset + dim / 4u];
  let x = f32(values[dim % 4u]); let y = f32(values[(dim % 4u) ^ 1u]);
  let phase = ((head * 16u + dim / 2u) * points + point) * 2u;
  let sign = select(1.0, -1.0, dim % 2u == 0u);
  return S(x * table[phase] + sign * y * table[phase + 1u]);
}
@compute @workgroup_size(8, ${rows})
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
  let points = params.a.x; let heads = params.a.y; let head = group.y; let batch = group.z;
  let queryStart = group.x * ${rows}u;
  let query = queryStart + local.y;
  let row = local.y * ${points}u;
  let reduction = local.y * 64u;
  for (var i = lane; i < ${querySize}u; i += ${threads}u) {
    let point = queryStart + i / 32u;
    var q = S(0);
    if (point < points) { q = rotated(point, head, i % 32u, 0u, batch); }
    queries[i] = q;
  }
  for (var base = 0u; base < points; base += 32u) {
    for (var i = lane; i < 256u; i += ${threads}u) {
      let dim = i / 8u;
      var k = V(0);
      ${transposedKeys ? `let column = base / 4u + i % 8u;
      if (column < params.a.z) { k = transposed[((batch * heads + head) * 32u + dim) * params.a.z + column]; }` : `
      let key = base + (i % 8u) * 4u;
      ${Array.from({ length: 4 }, (_, j) =>
        `if (key + ${j}u < points) { k[${j}u] = rotated(key + ${j}u, head, dim, 1u, batch); }`,
      ).join("\n      ")}`}
      tile[i] = k;
    }
    workgroupBarrier();
    var dot = vec4<f32>(0);
    for (var dim = 0u; dim < 32u; dim++) {
      dot += f32(queries[local.y * 32u + dim]) * vec4<f32>(tile[dim * 8u + local.x]);
    }
    for (var i = 0u; i < 4u; i++) {
      let key = base + local.x * 4u + i;
      if (key < points) { scores[row + key] = dot[i] * 0.1767766952966369; }
    }
    workgroupBarrier();
  }

  var localMax = -1e30;
  for (var j = local.x; j < points; j += 8u) { localMax = max(localMax, scores[row + j]); }
  reductions[reduction + local.x] = localMax;
  workgroupBarrier();
  var rowMax = -1e30;
  for (var j = 0u; j < 8u; j++) { rowMax = max(rowMax, reductions[reduction + j]); }
  workgroupBarrier();
  // Pair the two groups of 64 packed keys before reducing lane sums.
  for (var j = local.x; j < 64u; j += 8u) {
    var low = 0.0;
    var high = 0.0;
    let lowKey = j;
    let highKey = j + 64u;
    if (lowKey < points) {
      let weight = exp(scores[row + lowKey] - rowMax);
      scores[row + lowKey] = weight;
      low += weight;
    }
    if (highKey < points) {
      let weight = exp(scores[row + highKey] - rowMax);
      scores[row + highKey] = weight;
      high += weight;
    }
    reductions[reduction + j] = low + high;
  }
  // A lane owns both entries in each pair until the final three steps.
  for (var step = 32u; step >= 8u; step /= 2u) {
    for (var j = local.x; j < step; j += 8u) {
      reductions[reduction + j] += reductions[reduction + j + step];
    }
  }
  workgroupBarrier();
  for (var step = 4u; step > 0u; step /= 2u) {
    if (local.x < step) { reductions[reduction + local.x] += reductions[reduction + local.x + step]; }
    workgroupBarrier();
  }

  let scale = 1.0 / reductions[reduction];
  var sum = vec4<f32>(0);
  for (var base = 0u; base < points; base += 32u) {
    for (var i = lane; i < 256u; i += ${threads}u) {
      let key = base + i / 8u;
      var v = V(0);
      if (key < points) { v = input[(batch * points + key) * heads * 24u + heads * 16u + head * 8u + i % 8u]; }
      tile[i] = v;
    }
    workgroupBarrier();
    for (var i = 0u; i < 32u; i++) {
      let key = base + i;
      var p = 0.0;
      if (key < points) { p = scores[row + key] * scale; }
      sum += p * vec4<f32>(tile[i * 8u + local.x]);
    }
    workgroupBarrier();
  }
  if (query < points) { output[((batch * points + query) * heads + head) * 8u + local.x] = V(sum); }
}
`, transposedKeys ? [READ, READ, WRITE, READ] : undefined)
}
