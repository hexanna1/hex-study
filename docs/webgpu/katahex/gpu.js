// SPDX-License-Identifier: MIT
// Network execution adapted from cpp/neuralnet/eigenbackend.cpp.
import { createKernels } from "./kernels.js"
import { createInt8Kernels } from "./int8_kernels.js"
import { toHalf } from "./precision.js"
import { symmetryPoint } from "./symmetry.js"

const MAX_BATCH_SIZE = 128
const MAX_SUBMISSION_SIZE = 64
const GPU_YIELD_MS = 8
// Keep graph scratch storage around 54 MiB at the largest board sizes.
const BATCH_POINT_BUDGET = 32 * 19 * 19
// Limit the larger model's GPU work per submission for pointer responsiveness.
const MAIN_BATCH_POINT_BUDGET = 8 * 11 * 11
const Y_BATCH_POINT_BUDGET = 8 * 19 * 19

function batchSizeFor(size, bytesPerValue, modelName) {
  if (modelName !== "b5") {
    const pointBudget = modelName === "b10nbt-24" ? Y_BATCH_POINT_BUDGET : MAIN_BATCH_POINT_BUDGET
    return Math.min(16, Math.max(1, Math.floor(pointBudget / (size * size))))
  }
  return Math.min(MAX_BATCH_SIZE,
    Math.max(1, Math.floor(BATCH_POINT_BUDGET * 4 / (size * size * bytesPerValue))))
}

function bufferWithData(device, data, usage) {
  const buffer = device.createBuffer({ size: Math.max(4, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true })
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
  buffer.unmap()
  return buffer
}

function largestPackedInput(value) {
  if (!value || typeof value !== "object" || value.buffer) return 0
  let channels = value.weights?.packed ? value.input : 0
  for (const child of Object.values(value)) channels = Math.max(channels, largestPackedInput(child))
  return channels
}

class Graph {
  constructor(engine, size) {
    this.engine = engine
    this.device = engine.device
    this.size = size
    this.searchBatchSize = batchSizeFor(size, engine.bytesPerValue, engine.modelName)
    this.batchSize = Math.min(this.searchBatchSize, MAX_SUBMISSION_SIZE)
    this.buffers = []
    this.free = new Map()
    this.ops = []
    const model = engine.model
    const packedChannels = largestPackedInput(model)
    if (packedChannels) {
      this.packedInput = this.device.createBuffer({
        size: size * size * packedChannels * this.batchSize,
        usage: GPUBufferUsage.STORAGE,
      })
      this.inputScales = this.device.createBuffer({
        size: this.batchSize * 4,
        usage: GPUBufferUsage.STORAGE,
      })
      this.buffers.push(this.packedInput, this.inputScales)
    }
    this.spatial = this.tensor(size * size, model.spatialChannels)
    this.global = this.tensor(1, model.globalChannels)

    let trunk = this.conv(this.spatial, model.initial)
    const globalBias = this.conv(this.global, model.global)
    const biased = this.add(trunk, globalBias)
    this.release(trunk, globalBias)
    trunk = biased
    for (const block of model.blocks) trunk = this.block(trunk, block)
    const tip = this.norm(trunk, model.tip)
    this.release(trunk)

    const policy = model.policy
    let local = this.conv(tip, policy.local)
    const poolConv = this.conv(tip, policy.pool)
    const poolAct = this.norm(poolConv, policy.poolNorm)
    this.release(poolConv)
    const pooled = this.pool(poolAct, false)
    this.release(poolAct)
    const policyBias = this.conv(pooled, policy.poolBias)
    this.release(pooled)
    const localBiased = this.add(local, policyBias)
    this.release(local, policyBias)
    local = this.norm(localBiased, policy.norm)
    this.release(localBiased)
    this.policy = this.conv(local, policy.output)
    this.release(local)

    const value = model.value
    const vConv = this.conv(tip, value.conv)
    this.release(tip)
    const vAct = this.norm(vConv, value.norm)
    this.release(vConv)
    const vPool = this.pool(vAct, true)
    this.release(vAct)
    const hidden = this.conv(vPool, value.hidden)
    this.release(vPool)
    const hiddenAct = this.norm(hidden, value.hiddenNorm)
    this.release(hidden)
    const output = this.conv(hiddenAct, value.output)
    this.release(hiddenAct)
    this.value = this.norm(output, value.outputNorm)
    this.release(output)
    const outputBytes = (size * size + 3) * this.batchSize * 4
    const readout = this.device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    this.buffers.push(readout)
    this.output = { buffer: readout }
    this.operation("readout", this.policy, this.value, this.output, [size * size], [Math.ceil((size * size + 3) / 64)])
    this.readback = this.device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    this.buffers.push(this.readback)
  }

  tensor(points, channels) {
    const count = points * channels
    let buffer = this.free.get(count)?.pop()
    if (!buffer) {
      buffer = this.device.createBuffer({
        size: Math.ceil(count * this.batchSize * this.engine.bytesPerValue / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
      this.buffers.push(buffer)
    }
    return { buffer, points, channels, count }
  }

  release(...tensors) {
    for (const tensor of tensors) {
      if (!this.free.has(tensor.count)) this.free.set(tensor.count, [])
      this.free.get(tensor.count).push(tensor.buffer)
    }
  }

  operation(kind, a, b, output, parameters, dispatch) {
    const data = new Uint32Array(8)
    data.set(parameters)
    const params = bufferWithData(this.device, data, GPUBufferUsage.UNIFORM)
    this.buffers.push(params)
    const bindGroup = this.device.createBindGroup({
      layout: this.engine.layout,
      entries: [a.buffer, b.buffer, output.buffer, params].map((buffer, binding) => ({ binding, resource: { buffer } })),
    })
    this.ops.push({ pipeline: this.engine.pipelines[kind], bindGroup, dispatch })
    return output
  }

  int8Operation(kind, buffers, parameters, dispatch) {
    const data = new Uint32Array(8)
    data.set(parameters)
    const params = bufferWithData(this.device, data, GPUBufferUsage.UNIFORM)
    this.buffers.push(params)
    const bindGroup = this.device.createBindGroup({
      layout: this.engine.int8Layouts[kind],
      entries: [...buffers, params].map((buffer, binding) => ({ binding, resource: { buffer } })),
    })
    this.ops.push({ pipeline: this.engine.int8Pipelines[kind], bindGroup, dispatch })
  }

  conv(input, layer) {
    const output = this.tensor(input.points, layer.output)
    const width = input.points === 1 ? 1 : this.size
    if (layer.weights.packed) {
      this.int8Operation("max", [input.buffer, this.inputScales], [input.count], [1])
      this.int8Operation("pack", [input.buffer, this.inputScales, this.packedInput],
        [input.points, input.channels], [Math.ceil(input.count / 4 / 64)])
      this.int8Operation("conv",
        [this.packedInput, layer.weights.packed.buffer, this.inputScales, layer.weights.scales.buffer, output.buffer],
        [width, input.points / width, layer.input, layer.output, layer.ky, layer.kx, layer.dy, layer.dx],
        [Math.ceil(layer.output / 16), Math.ceil(input.points / 16)])
      return output
    }
    return this.operation("conv", input, layer.weights, output,
      [width, input.points / width, layer.input, layer.output, layer.ky, layer.kx, layer.dy, layer.dx],
      [Math.ceil(layer.output / 16), Math.ceil(input.points / 16)])
  }

  norm(input, layer) {
    const normalized = this.operation("norm", input, layer.weights, this.tensor(input.points, input.channels),
      [input.count, input.channels, layer.activation], [Math.ceil(input.count / 64)])
    if (this.engine.game !== "y" || input.points === 1) return normalized
    const masked = this.operation("mask", normalized, this.spatial, this.tensor(input.points, input.channels),
      [input.count, input.channels, this.engine.model.spatialChannels], [Math.ceil(input.count / 64)])
    this.release(normalized)
    return masked
  }

  add(input, bias) {
    return this.operation("add", input, bias, this.tensor(input.points, input.channels),
      [input.count, bias.count], [Math.ceil(input.count / 64)])
  }

  pool(input, valueHead) {
    const playable = this.engine.game === "y" ? this.size * (this.size + 1) / 2 : input.points
    return this.operation("pool", input, this.spatial, this.tensor(1, input.channels * 3),
      [input.points, input.channels, Number(valueHead), playable, this.engine.model.spatialChannels],
      [Math.ceil(input.channels / 64)])
  }

  normConv(input, layer) {
    const norm = this.norm(input, layer.norm)
    const output = this.conv(norm, layer.conv)
    this.release(norm)
    return output
  }

  block(input, block) {
    let mid
    if (block.pool) {
      const pre = this.norm(input, block.pre.norm)
      const local = this.conv(pre, block.pre.conv)
      const poolConv = this.conv(pre, block.pool.conv)
      this.release(pre)
      const poolAct = this.norm(poolConv, block.pool.norm)
      this.release(poolConv)
      const pooled = this.pool(poolAct, false)
      this.release(poolAct)
      const bias = this.conv(pooled, block.pool.matrix)
      this.release(pooled)
      mid = this.add(local, bias)
      this.release(local, bias)
    } else {
      mid = this.normConv(input, block.pre)
    }
    if (block.blocks) {
      for (const child of block.blocks) mid = this.block(mid, child)
    }
    const post = this.normConv(mid, block.post)
    this.release(mid)
    const output = this.add(post, input)
    this.release(post, input)
    return output
  }

  async evaluate(states) {
    const model = this.engine.model
    const spatial = new Float32Array(this.spatial.count * states.length)
    const global = new Float32Array(model.globalChannels * states.length)
    states.forEach(({ board, toPlay, symmetry = 0 }, batch) => {
      const transpose = this.engine.game === "hex" && toPlay === 2
      for (let i = 0; i < board.length; i++) {
        // Hex evaluates White in transposed coordinates; Y uses one shared
        // three-side goal for both colors.
        const canonical = transpose ? (i % this.size) * this.size + Math.floor(i / this.size) : i
        const point = symmetry ? symmetryPoint(canonical, this.size, this.engine.game, symmetry) : canonical
        const offset = batch * this.spatial.count + point * model.spatialChannels
        if (board[i] !== 3) spatial[offset] = 1
        if (board[i] === 1 || board[i] === 2) spatial[offset + (board[i] === toPlay ? 1 : 2)] = 1
      }
      global[batch * model.globalChannels] = Number(toPlay === 2)
    })
    const write = (buffer, values) => {
      const data = this.engine.half ? toHalf(values) : values
      // Queue writes are four-byte aligned, including odd-sized half arrays.
      const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4)
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
      this.device.queue.writeBuffer(buffer, 0, padded)
    }
    write(this.spatial.buffer, spatial)
    write(this.global.buffer, global)
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    for (const op of this.ops) {
      pass.setPipeline(op.pipeline)
      pass.setBindGroup(0, op.bindGroup)
      pass.dispatchWorkgroups(op.dispatch[0], op.dispatch[1] || 1, states.length)
    }
    pass.end()
    const stride = this.size * this.size + 3
    const byteLength = states.length * stride * 4
    encoder.copyBufferToBuffer(this.output.buffer, 0, this.readback, 0, byteLength)
    this.device.queue.submit([encoder.finish()])
    await this.readback.mapAsync(GPUMapMode.READ, 0, byteLength)
    const values = new Float32Array(this.readback.getMappedRange(0, byteLength).slice(0))
    this.readback.unmap()
    if (values.some((value) => !Number.isFinite(value))) {
      throw new Error("The GPU returned a non-finite evaluation. Analysis stopped.")
    }
    return states.map(({ toPlay, symmetry = 0 }, i) => {
      const rawPolicy = values.subarray(i * stride, (i + 1) * stride - 3)
      const transpose = this.engine.game === "hex" && toPlay === 2
      let policy = rawPolicy
      if (symmetry) {
        policy = Float32Array.from(rawPolicy, (_, point) => {
          const canonical = transpose ? (point % this.size) * this.size + Math.floor(point / this.size) : point
          return rawPolicy[symmetryPoint(canonical, this.size, this.engine.game, symmetry)]
        })
      } else if (transpose) {
        policy = Float32Array.from(rawPolicy, (_, point) => rawPolicy[(point % this.size) * this.size + Math.floor(point / this.size)])
      }
      return { policy, value: values.subarray((i + 1) * stride - 3, (i + 1) * stride) }
    })
  }

  destroy() {
    for (const buffer of this.buffers) buffer.destroy()
  }
}

export class Network {
  static async create(model, onLost, modelName, packedInt8, game) {
    if (!navigator.gpu) throw new Error("Analysis needs a browser with WebGPU support.")
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error("No WebGPU adapter is available on this device.")
    const half = adapter.features.has("shader-f16")
    const device = await adapter.requestDevice({ requiredFeatures: half ? ["shader-f16"] : [] })
    const engine = new Network(device, model, half, modelName, packedInt8, game)
    try {
      engine.layout = device.createBindGroupLayout({
        entries: ["read-only-storage", "read-only-storage", "storage", "uniform"].map((type, binding) => ({
          binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
        })),
      })
      const layout = device.createPipelineLayout({ bindGroupLayouts: [engine.layout] })
      engine.pipelines = Object.fromEntries(await Promise.all(Object.entries(createKernels(half)).map(async ([name, code]) => [name,
        await device.createComputePipelineAsync({ layout, compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }),
      ])))
      if (packedInt8) {
        const types = {
          max: ["read-only-storage", "storage", "uniform"],
          pack: ["read-only-storage", "read-only-storage", "storage", "uniform"],
          conv: ["read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"],
        }
        const kernels = createInt8Kernels(half)
        engine.int8Layouts = Object.fromEntries(Object.entries(types).map(([name, buffers]) => [name,
          device.createBindGroupLayout({ entries: buffers.map((type, binding) => ({
            binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
          })) }),
        ]))
        engine.int8Pipelines = Object.fromEntries(await Promise.all(Object.entries(kernels).map(async ([name, code]) => [name,
          await device.createComputePipelineAsync({
            layout: device.createPipelineLayout({ bindGroupLayouts: [engine.int8Layouts[name]] }),
            compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
          }),
        ])))
      }
      const upload = (object) => {
        for (const [key, value] of Object.entries(object)) {
          if (value instanceof Float32Array) {
            const data = key === "scales" ? value : half ? toHalf(value) : value
            const buffer = bufferWithData(device, data, GPUBufferUsage.STORAGE)
            engine.weights.push(buffer)
            object[key] = { buffer, count: value.length }
          } else if (value instanceof Uint32Array) {
            const buffer = bufferWithData(device, value, GPUBufferUsage.STORAGE)
            engine.weights.push(buffer)
            object[key] = { buffer, count: value.length }
          } else if (value && typeof value === "object") upload(value)
        }
      }
      upload(model)
      device.lost.then(({ reason, message }) => {
        if (reason !== "destroyed") onLost(new Error(`GPU connection lost: ${message}. Reload the page to retry.`))
      })
      device.addEventListener("uncapturederror", (event) => onLost(event.error))
      return engine
    } catch (error) {
      engine.destroy()
      throw error
    }
  }

  constructor(device, model, half, modelName, packedInt8, game) {
    this.half = half
    this.modelName = modelName
    this.game = game
    this.packedInt8 = packedInt8
    this.runtime = packedInt8
      ? `int8 conv + f${half ? 16 : 32}`
      : `f${half ? 16 : 32} conv`
    this.runtimeHelp = packedInt8
      ? `8-bit integer convolutions; ${half ? 16 : 32}-bit floating-point tensors.`
      : `${half ? 16 : 32}-bit floating-point convolutions and tensors.`
    this.bytesPerValue = half ? 2 : 4
    this.device = device
    this.model = model
    this.weights = []
    this.graph = null
  }

  prepare(size) {
    if (this.graph?.size !== size) {
      this.graph?.destroy()
      this.graph = null
      this.graph = new Graph(this, size)
    }
    return this.graph
  }

  async evaluate(states, cancelled, yieldMs = GPU_YIELD_MS) {
    this.prepare(states[0].size)
    const results = []
    const submissionSize = this.graph.batchSize
    for (let offset = 0; offset < states.length; offset += submissionSize) {
      if (cancelled()) return null
      results.push(...await this.graph.evaluate(states.slice(offset, offset + submissionSize)))
      await new Promise((resolve) => setTimeout(resolve, yieldMs))
    }
    return results
  }

  destroy() {
    this.graph?.destroy()
    for (const buffer of this.weights) buffer.destroy()
    this.device.destroy()
  }
}
