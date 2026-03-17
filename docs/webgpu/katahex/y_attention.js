// Below Y13, small boards keep complete score rows in workgroup memory.
export const Y_ATTENTION_QUERY_ROWS = 8
const FUSED_Y_MAX_POINTS = 78
export const ONLINE_Y_MIN_POINTS = FUSED_Y_MAX_POINTS + 1
export const ONLINE_Y_QUERY_ROWS = 16

export function createOnlineYAttention(half, graphBias) {
  const rows = ONLINE_Y_QUERY_ROWS
  const lanes = rows * 8
  const scalar = half ? "f16" : "f32"
  const prefix = half ? "enable f16;" : ""
  const yRotateQK = half || graphBias ? null : `
alias Pair = vec2<f32>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(1) var<storage, read> rope: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> qkv: array<Pair>;
@group(0) @binding(3) var<uniform> params: Params;
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
  const repeat = (emit) => Array.from({ length: 8 }, (_, i) => emit(i)).join("\n    ")
  const yAttentionOnline = `${prefix}
alias Scalar = ${scalar};
alias Vector = vec4<Scalar>;
struct Params { a: vec4<u32>, b: vec4<u32> }
@group(0) @binding(0) var<storage, read> qkv: array<Vector>;
${graphBias ? "@group(0) @binding(1) var<storage, read> bias: array<f32>;" : ""}
@group(0) @binding(2) var<storage, read_write> output: array<Vector>;
@group(0) @binding(3) var<uniform> params: Params;
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
    if (point < points) { x = qkv[((batch * points + point) * heads * 3u + head) * 8u + i % 8u]; }
    queries[i] = x;
  }`}
  var maximum = -1e30;
  var normalizer = 0.0;
  var weighted = vec4<f32>(0);
  // The key tile bounds workgroup storage independently of the board size.
  for (var base = 0u; base < points; base += 32u) {
    // Load four neighboring keys as vectors and transpose their components into the tile.
    for (var i = lane; i < 64u; i += ${lanes}u) {
      let dim = i / 8u;
      let firstKey = base + i % 8u * 4u;
      let offset = ((batch * points + firstKey) * heads * 3u + heads + head) * 8u + dim;
      let pointStride = heads * 24u;
      var a = Vector(0);
      var b = Vector(0);
      var c = Vector(0);
      var d = Vector(0);
      if (firstKey < points) { a = qkv[offset]; }
      if (firstKey + 1u < points) { b = qkv[offset + pointStride]; }
      if (firstKey + 2u < points) { c = qkv[offset + pointStride * 2u]; }
      if (firstKey + 3u < points) { d = qkv[offset + pointStride * 3u]; }
      let keyIndex = dim * 32u + i % 8u;
      keys[keyIndex] = Vector(a.x, b.x, c.x, d.x);
      keys[keyIndex + 8u] = Vector(a.y, b.y, c.y, d.y);
      keys[keyIndex + 16u] = Vector(a.z, b.z, c.z, d.z);
      keys[keyIndex + 24u] = Vector(a.w, b.w, c.w, d.w);
    }
    for (var i = lane; i < 256u; i += ${lanes}u) {
      let key = base + i / 8u;
      var v = Vector(0);
      if (key < points) { v = qkv[((batch * points + key) * heads * 3u + heads * 2u + head) * 8u + i % 8u]; }
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
    ${graphBias ? `var positionalBias = vec4<f32>(0);
    if (query < points) {
      let biasRow = (head * points + query) * points;
      for (var j = 0u; j < 4u; j++) {
        if (valid[j]) { positionalBias[j] = bias[biasRow + keyPositions[j]]; }
      }
    }` : ""}
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
  return yRotateQK ? { yRotateQK, yAttentionOnline } : { yAttentionOnline }
}

// Rotate each key once and arrange four consecutive keys for the score tile.
export function createYAttentionKeys() {
  return `enable f16;
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
`
}

export function createFusedYAttention() {
  const rows = Y_ATTENTION_QUERY_ROWS
  const points = FUSED_Y_MAX_POINTS
  const threads = rows * 8
  const repeat = (count, emit) => Array.from({ length: count }, (_, i) => emit(i)).join("\n")
  const softmaxBlocks = Math.ceil(points / 128)
  const querySize = rows * 32
  return `enable f16;
alias S = f16;
alias V = vec4<S>;
struct Params { a:vec4<u32>, b:vec4<u32> }
@group(0) @binding(0) var<storage,read> input:array<V>;
@group(0) @binding(1) var<storage,read> table:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<V>;
@group(0) @binding(3) var<storage, read> transposed:array<V>;
@group(0) @binding(4) var<uniform> params:Params;
var<workgroup> queries:array<S,${querySize}>;
var<workgroup> tile:array<V,256>;
var<workgroup> scores:array<f32, ${rows * points}>;
var<workgroup> reductions:array<f32, ${rows * 64}>;
fn rotatedQuery(point:u32, head:u32, dim:u32, batch:u32)->S {
  let points = params.a.x; let heads = params.a.y;
  let offset = (batch*points+point)*heads*24u + head*8u;
  let values = input[offset + dim/4u];
  let x = f32(values[dim % 4u]); let y = f32(values[(dim % 4u) ^ 1u]);
  let phase = ((head * 16u + dim / 2u) * points + point) * 2u;
  let sign = select(1.0, -1.0, dim % 2u == 0u);
  return S(x * table[phase] + sign * y * table[phase + 1u]);
}
@compute @workgroup_size(8,${rows})
fn main(@builtin(local_invocation_id) local:vec3<u32>, @builtin(local_invocation_index) lane:u32, @builtin(workgroup_id) group:vec3<u32>) {
  let points=params.a.x; let heads=params.a.y; let head=group.y; let batch=group.z;
  let queryStart=group.x*${rows}u;
  let query=queryStart+local.y;
  let row=local.y*${points}u;
  let reduction=local.y*64u;
  for(var i=lane;i<${querySize}u;i+=${threads}u) {
    let point=queryStart+i/32u;
    var q=S(0);
    if(point<points) {q=rotatedQuery(point,head,i%32u,batch);}
    queries[i]=q;
  }
  for(var base=0u;base<points;base+=32u) {
    for(var i=lane;i<256u;i+=${threads}u) {
      let dim=i/8u;
      let column=base/4u+i%8u;
      var k=V(0);
      if(column<params.a.z) {
        k=transposed[((batch*heads+head)*32u+dim)*params.a.z+column];
      }
      tile[i]=k;
    }
    workgroupBarrier();
    var dot=vec4<f32>(0);
    for(var dim=0u;dim<32u;dim++) {
      dot += f32(queries[local.y*32u+dim])*vec4<f32>(tile[dim*8u+local.x]);
    }
    for(var i=0u;i<4u;i++) {
      let key=base+local.x*4u+i;
      if(key<points){
        let score = dot[i]*0.1767766952966369;
        scores[row+key] = score;
      }
    }
    workgroupBarrier();
  }
  var localMax = -1e30;
  for(var j=local.x;j<points;j+=8u) {localMax=max(localMax,scores[row+j]);}
  reductions[reduction+local.x]=localMax;
  workgroupBarrier();
  var rowMax=-1e30;
  for(var j=0u;j<8u;j++) {rowMax=max(rowMax,reductions[reduction+j]);}
  workgroupBarrier();
  for(var j=local.x;j<64u;j+=8u) {
    var low=0.0;
    var high=0.0;
    // Keep the same low/high sums and reduction order as the separate softmax.
    ${repeat(softmaxBlocks, (i) => `let low${i}=j + ${128 * i}u;
    let high${i}=j + ${128 * i + 64}u;
    if(low${i}<points) {let w=exp(scores[row+low${i}]-rowMax); scores[row+low${i}]=w; low+=w;}
    if(high${i}<points) {let w=exp(scores[row+high${i}]-rowMax); scores[row+high${i}]=w; high+=w;}`)}
    reductions[reduction+j] = low+high;
  }
  // Each lane owns the entries paired by the first three reduction steps.
  for(var step=32u;step>=8u;step/=2u) {
    for(var j=local.x;j<step;j+=8u) {
      reductions[reduction+j]+=reductions[reduction+j+step];
    }
  }
  workgroupBarrier();
  for(var step=4u;step>0u;step/=2u) {
    if(local.x<step) {
      reductions[reduction+local.x]+=reductions[reduction+local.x+step];
    }
    workgroupBarrier();
  }
  let scale = 1.0 / reductions[reduction];
  var sum=vec4<f32>(0);
  for(var base=0u;base<points;base+=32u) {
    for(var i=lane;i<256u;i+=${threads}u) {
      let key=base+i/8u;
      var v=V(0);
      if(key<points) {v=input[(batch*points+key)*heads*24u+heads*16u+head*8u+i%8u];}
      tile[i]=v;
    }
    workgroupBarrier();
    for(var i=0u;i<32u;i++){
      let key=base+i;
      var p=0.0;
      if(key<points) {p = scores[row+key]*scale;}
      sum += p*vec4<f32>(tile[i*8u+local.x]);
    }
    workgroupBarrier();
  }
  if(query<points) {output[((batch*points+query)*heads+head)*8u+local.x]=V(sum);}
}
`
}
