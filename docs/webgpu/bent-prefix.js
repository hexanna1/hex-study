// The Bent-Y stem and first transformer projections are local to each feature row.
import { bentYFeatureRows, BENT_SIDE_BIT } from "./benty.js"
import { RMS_POINTS_PER_GROUP } from "./transformer_kernels.js"
import { kernel, READ, WRITE } from "./kernel.js"

// Each two-bit tier encodes zero, one, or both thresholds reached.
const rows = []
const lookup = new Uint16Array(1 << (BENT_SIDE_BIT + 1))
const tiers = [0, 1, 3]
for (let side = 0; side < 2; side++) {
  for (let degree = 0; degree < 2; degree++) {
    const common = 1 | (degree << 4) | (side << BENT_SIDE_BIT)
    rows.push(common | 2, common | 4)
    for (const own of tiers) {
      for (const opponent of tiers) {
        for (const ownSides of tiers) {
          for (const opponentSides of tiers) {
            rows.push(common | (own << 5) | (opponent << 7) |
              ((ownSides & 1) << 9) | ((opponentSides & 1) << 10) |
              ((ownSides >> 1) << 11) | ((opponentSides >> 1) << 12))
          }
        }
      }
    }
  }
}
rows.forEach((row, index) => { lookup[row] = index })

export function packBentYRows(states, size, maps, points) {
  const result = new Uint32Array(states.length * points.length)
  const orientations = states.map(({ toPlay, symmetry = 0 }) => maps[toPlay - 1][symmetry])
  bentYFeatureRows(states, size, points, (batch, point, features) => {
    result[batch * points.length + orientations[batch][point]] = lookup[features]
  })
  return result
}

export function createBentPrefix(engine) {
  const { device, model, bytesPerValue } = engine
  const { pre, blocks: [attention] } = model.blocks[0]
  const temporary = []
  const bufferFor = (size, usage, owner, mappedAtCreation = false) => {
    const buffer = device.createBuffer({ size, usage, mappedAtCreation })
    owner.push(buffer)
    return buffer
  }
  const count = rows.length
  const tensor = (channels, owner = temporary) => ({
    buffer: bufferFor(count * channels * bytesPerValue, GPUBufferUsage.STORAGE, owner),
    channels,
  })
  const uniform = (params) => {
    const buffer = bufferFor(32, GPUBufferUsage.UNIFORM, temporary, true)
    new Uint32Array(buffer.getMappedRange()).set(params)
    buffer.unmap()
    return buffer
  }
  const freeTemporary = () => {
    for (const buffer of temporary) buffer.destroy()
  }
  try {
    const keys = bufferFor(count * Uint32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE, temporary, true)
    new Uint32Array(keys.getMappedRange()).set(rows)
    keys.unmap()
    const initial = tensor(model.initial.output, engine.weights)
    const affine = tensor(initial.channels)
    const inner = tensor(pre.conv.output, engine.weights)
    const normalized = tensor(inner.channels)
    const qkv = tensor(attention.qkv.output, engine.weights)

    const tile = engine.pointwiseTiles[0]
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    const dispatch = (kind, resources, parameters, [x, y = 1]) => {
      const { pipeline, layout } = engine.pipelines[kind]
      const bindGroup = device.createBindGroup({
        layout,
        entries: [...resources, uniform(parameters)].map((buffer, binding) => ({ binding, resource: { buffer } })),
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(x, y)
    }
    const linear = (input, layer, output) => dispatch(
      "pointwise0", [input.buffer, layer.weights.buffer, output.buffer],
      [count, input.channels, output.channels, 1],
      [Math.ceil(output.channels / tile.columns), Math.ceil(count / tile.points)],
    )
    const epsilon = new Uint32Array(new Float32Array([attention.norm.epsilon]).buffer)[0]
    dispatch(
      "bentPrefixStem", [keys, model.initial.weights.buffer, initial.buffer, model.global.weights.buffer],
      [model.initial.input, initial.channels, count], [Math.ceil(count * initial.channels / 256)],
    )
    dispatch(
      "normVector", [initial.buffer, pre.norm.weights.buffer, affine.buffer],
      [count * initial.channels, initial.channels, pre.norm.activation], [Math.ceil(count * initial.channels / 256)],
    )
    linear(affine, pre.conv, inner)
    dispatch(
      "rms", [inner.buffer, attention.norm.weights.buffer, normalized.buffer],
      [count, inner.channels, epsilon], [Math.ceil(count / RMS_POINTS_PER_GROUP)],
    )
    linear(normalized, attention.qkv, qkv)
    pass.end()
    device.queue.submit([encoder.finish()])
    // Queue order makes the tables ready before any graph can consume them.
    device.queue.onSubmittedWorkDone().then(freeTemporary, freeTemporary)
    return { initial, inner, qkv }
  } catch (error) {
    freeTemporary()
    throw error
  }
}

export function bentPrefixKernels(half) {
  const header = `${half ? "enable f16;" : ""}
alias Vector = vec4<${half ? "f16" : "f32"}>;
struct Params { a: vec4<u32>, b: vec4<u32> }
`
  const bentPrefixStem = header + `
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<Vector>;
@group(0) @binding(2) var<storage, read_write> output: array<Vector>;
@group(0) @binding(3) var<storage, read> globalWeights: array<Vector>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let columns = params.a.y / 4u;
  if (id.x >= params.a.z * columns) { return; }
  let column = id.x % columns;
  let features = keys[id.x / columns];
  var sum = vec4<f32>(0);
  for (var k = 0u; k < params.a.x; k++) {
    let x = select(0.0, f32((features >> k) & 1u), k < ${BENT_SIDE_BIT}u);
    sum += x * vec4<f32>(weights[k * columns + column]);
  }
  let side = f32((features >> ${BENT_SIDE_BIT}u) & 1u);
  // Keep the stem's tensor rounding before the global bias is added.
  output[id.x] = Vector(vec4<f32>(Vector(sum)) + side * vec4<f32>(globalWeights[column]));
}
`
  const scatterBentPrefix = header + `
@group(0) @binding(0) var<storage, read> initialTable: array<Vector>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> initial: array<Vector>;
@group(0) @binding(3) var<storage, read> innerTable: array<Vector>;
@group(0) @binding(4) var<storage, read_write> inner: array<Vector>;
@group(0) @binding(5) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let points = params.a.x;
  let initialColumns = params.a.y / 4u;
  let innerColumns = params.a.z / 4u;
  // The stem is wider than the inner projection, so one grid covers both outputs.
  let point = id.x / initialColumns;
  if (point >= points) { return; }
  let row = id.z * points + point;
  let key = indices[row];
  let column = id.x % initialColumns;
  initial[row * initialColumns + column] = initialTable[key * initialColumns + column];
  if (column < innerColumns) {
    inner[row * innerColumns + column] = innerTable[key * innerColumns + column];
  }
}
`
  return {
    bentPrefixStem: kernel(bentPrefixStem, [READ, READ, WRITE, READ]),
    scatterBentPrefix: kernel(scatterBentPrefix, [READ, READ, WRITE, READ, WRITE]),
  }
}
