import { toHalf } from "./precision.js"
import { tensorScratch } from "./tensor_scratch.js"
import { isWinogradLayer, WINOGRAD_COEFFICIENTS } from "./winograd_kernels.js"
import { createWinogradFilter, transformWinogradWeights } from "./winograd_filters.js"

export function mappedBuffer(device, size, usage, fill) {
  let buffer
  try {
    buffer = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage, mappedAtCreation: true })
  } catch (error) {
    // The size is aligned. Failure to allocate the initial mapping throws synchronously.
    if (error instanceof RangeError) throw new GPUOutOfMemoryError(error.message)
    throw error
  }
  try {
    fill(buffer.getMappedRange())
    buffer.unmap()
    return buffer
  } catch (error) {
    buffer.destroy()
    throw error
  }
}

export function bufferWithData(device, data, usage, half = false) {
  return mappedBuffer(device, data.length * (half ? 2 : data.BYTES_PER_ELEMENT), usage, (range) => {
    if (!half) {
      new Uint8Array(range).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
    } else if (typeof Float16Array === "function") {
      new Float16Array(range, 0, data.length).set(data)
    } else {
      toHalf(data, new Uint16Array(range, 0, data.length))
    }
  })
}

// Submit groups as the archive arrives. Each dispatch borrows its temporary
// input buffers until submission, or until model loading is cancelled.
async function prepareUploadBatch(device, code) {
  device.pushErrorScope("validation")
  let pipeline
  try {
    pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    }).catch(() => null)
  } finally {
    await device.popErrorScope()
  }
  if (!pipeline) return null
  const layout = pipeline.getBindGroupLayout(0)
  const inputs = []
  let encoder, pass, layers = 0
  function temporary(buffer) {
    inputs.push(buffer)
    return buffer
  }
  function release() {
    for (const buffer of inputs) buffer.destroy()
    inputs.length = 0
  }
  function flush() {
    if (!pass) return
    pass.end()
    device.queue.submit([encoder.finish()])
    pass = null
    encoder = null
    layers = 0
    release()
  }
  function dispatch(buffers, groups) {
    if (!pass) {
      encoder = device.createCommandEncoder()
      pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
    }
    const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
    pass.setBindGroup(0, device.createBindGroup({ layout, entries }))
    pass.dispatchWorkgroups(groups)
    if (++layers === 12) flush()
  }
  return { temporary, dispatch, flush, release }
}

// Four packed words per invocation share one aligned scale lookup.
const HALF_UPLOAD_SHADER = `enable f16;
@group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f16>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let groups = input[0].x;
  if (id.x >= groups) { return; }
  let code = vec4<i32>(input[1u + id.x]);
  let scales = 1u + groups + id.x % input[0].y * 4u;
  for (var i = 0u; i < 4u; i++) {
    let word = code[i];
    let decoded = vec4<f32>(
      f32(extractBits(word, 0u, 8u)), f32(extractBits(word, 8u, 8u)),
      f32(extractBits(word, 16u, 8u)), f32(extractBits(word, 24u, 8u)),
    ) * bitcast<vec4<f32>>(input[scales + i]);
    output[id.x * 4u + i] = vec4<f16>(decoded);
  }
}
`

export async function prepareHalfUpload(device) {
  const batch = await prepareUploadBatch(device, HALF_UPLOAD_SHADER)
  if (!batch) return null
  function encode({ codes, scales }) {
    const size = codes.length * Uint16Array.BYTES_PER_ELEMENT
    const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE })
    try {
      const scaleOffset = 16 + codes.length
      const input = mappedBuffer(device, scaleOffset + scales.byteLength, GPUBufferUsage.STORAGE, (range) => {
        new Uint32Array(range, 0, 2).set([codes.length / 16, scales.length / 16])
        new Int8Array(range, 16, codes.length).set(codes)
        new Float32Array(range, scaleOffset, scales.length).set(scales)
      })
      batch.temporary(input)
      batch.dispatch([input, output], Math.ceil(codes.length / 1024))
      return output
    } catch (error) {
      output.destroy()
      throw error
    }
  }
  return { encode, flush: batch.flush, release: batch.release }
}

function packedWeights(device, { codes, scales }, scale) {
  return mappedBuffer(device, codes.byteLength + scales.byteLength, GPUBufferUsage.STORAGE, (range) => {
    new Int8Array(range, 0, codes.length).set(codes)
    const outputScales = new Float32Array(range, codes.byteLength, scales.length)
    for (let i = 0; i < scales.length; i++) outputScales[i] = scales[i] * scale
  })
}

async function prepareHalfWinograd(device) {
  const batch = await prepareUploadBatch(device, createWinogradFilter(true).code)
  if (!batch) return null
  const encode = (tensor) => {
    const filters = tensor.codes.length / 9
    const buffer = device.createBuffer({
      size: filters * WINOGRAD_COEFFICIENTS * Uint16Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    })
    try {
      const weights = batch.temporary(packedWeights(device, tensor, 1))
      const shape = Uint32Array.of(filters / 4, tensor.scales.length / 4)
      const params = batch.temporary(bufferWithData(device, shape, GPUBufferUsage.UNIFORM))
      batch.dispatch([weights, buffer, params], Math.ceil(filters / (4 * 64)))
      return buffer
    } catch (error) {
      buffer.destroy()
      throw error
    }
  }
  return { encode, flush: batch.flush, release: batch.release }
}

export async function uploadModel(engine, archive, signal, halfUpload) {
  const { model, half, device } = engine
  const scratch = tensorScratch()
  const preparation = half && !model.transformer ? await prepareHalfWinograd(device) : null
  let nextYield = performance.now() + 32
  const upload = async (object) => {
    // Yield to worker messages even when the entire download is buffered.
    if (performance.now() >= nextYield) {
      await new Promise((resolve) => setTimeout(resolve, 0))
      signal.throwIfAborted()
      nextYield = performance.now() + 32
    }
    const winograd = !model.transformer && isWinogradLayer(object)
    const quantized = object.weights?.type === "q7"
    const pointwise = object.ky === 1 && object.kx === 1 && object.input % 4 === 0 && object.output % 4 === 0
    const compact = !half && !model.transformer && quantized && (winograd || pointwise)
    if (compact) {
      const buffer = packedWeights(device, await archive.quantizedTensor(object.weights), winograd ? 0.25 : 1)
      engine.weights.push(buffer)
      if (winograd) {
        object.winograd = { buffer }
        delete object.weights
      } else object.weights = { buffer, compact: true }
      return
    }
    if (preparation && winograd && quantized) {
      const buffer = preparation.encode(await archive.quantizedTensor(object.weights))
      engine.weights.push(buffer)
      object.winograd = { buffer }
      delete object.weights
      return
    }
    if (half && winograd && object.weights?.type) object.weights = await archive.decodeTensor(object.weights)
    if (half && winograd && object.weights instanceof Float32Array) {
      const count = WINOGRAD_COEFFICIENTS * object.input * object.output
      const buffer = mappedBuffer(device, count * 2, GPUBufferUsage.STORAGE, (range) => {
        const values = typeof Float16Array === "function" ? new Float16Array(range) : null
        const bits = values ? null : new Uint16Array(range)
        transformWinogradWeights(object, scratch, (data, offset) => {
          if (values) values.set(data, offset)
          else toHalf(data, bits.subarray(offset, offset + data.length))
        })
      })
      engine.weights.push(buffer)
      object.winograd = { buffer }
      delete object.weights
      return
    }
    for (let [key, value] of Object.entries(object)) {
      if (key === "biasInput" || key === "biasOutput") {
        value.weights = (await archive.decodeTensor(value.weights)).slice()
        continue
      }
      if (key === "ropeFreqs") {
        object[key] = (await archive.decodeTensor(value)).slice()
        continue
      }
      const storeHalf = half && object.kind !== "rms"
      if (halfUpload && storeHalf && value?.type === "q7" && value.length >= 4096 && value.channels % 16 === 0) {
        const buffer = halfUpload.encode(await archive.quantizedTensor(value))
        engine.weights.push(buffer)
        object[key] = { buffer }
        continue
      }
      const WeightArray = storeHalf ? globalThis.Float16Array : Float32Array
      if (value?.type === "q7" && WeightArray) {
        const quantized = await archive.quantizedTensor(value)
        const bytes = WeightArray.BYTES_PER_ELEMENT * value.length
        const buffer = mappedBuffer(device, bytes, GPUBufferUsage.STORAGE, (range) => {
          archive.dequantize(quantized, new WeightArray(range, 0, value.length))
        })
        engine.weights.push(buffer)
        object[key] = { buffer }
        continue
      }
      if (value?.type) object[key] = value = await archive.decodeTensor(value)
      if (value instanceof Float32Array) {
        const buffer = bufferWithData(device, value, GPUBufferUsage.STORAGE, storeHalf)
        engine.weights.push(buffer)
        object[key] = { buffer }
      } else if (value && typeof value === "object") await upload(value)
    }
  }
  try {
    signal.throwIfAborted()
    await upload(model)
    preparation?.flush()
    halfUpload?.flush()
  } finally {
    preparation?.release()
    halfUpload?.release()
  }
}
