// SPDX-License-Identifier: MIT
// Network execution adapted from cpp/neuralnet/eigenbackend.cpp.
import { bentYInput, graphBiasFeatures, graphAttentionBias } from "./benty.js"
import "../../board_geometry.js"
import { createTransformerKernels, RMS_POINTS_PER_GROUP } from "./transformer_kernels.js"
import {
  createFusedYAttention, createOnlineYAttention, createYAttentionKeys,
  ONLINE_Y_MIN_POINTS, ONLINE_Y_QUERY_ROWS, Y_ATTENTION_QUERY_ROWS,
} from "./y_attention.js"
import { createYavalathAttention, YAVALATH_ATTENTION_QUERY_ROWS } from "./yavalath_attention.js"
import { createPointwiseKernels, POINTWISE_ROWS_FOR_WIDE_TILE, pointwiseTiles } from "./pointwise_kernels.js"
import { createKernels } from "./kernels.js"
import {
  createWinogradKernels,
  createWinogradWeightTransform,
  transformWinogradWeights,
  WINOGRAD_OUTPUT_TILE,
  WINOGRAD_COEFFICIENTS,
} from "./winograd_kernels.js"
import { COMPACT_COEFFICIENTS, COMPACT_OUTPUT_TILE, createCompactWinogradKernels } from "./compact_winograd.js"
import { toHalf } from "./precision.js"
import { tensorScratch } from "./tensor_scratch.js"
import { symmetriesFor, symmetryPoint } from "./symmetry.js"

const MAX_SUBMISSION_SIZE = 64
// Bound submitted batches for scratch memory and response time.
const BATCH_POINT_BUDGET = 32 * 19 * 19
const ATTENTION_SCORE_BUDGET = 64 * 1024 * 1024
// Limit the larger model's GPU work per submission for pointer responsiveness.
const MAIN_BATCH_POINT_BUDGET = 7 * 19 * 19
const MAIN_FP32_BATCH_POINT_BUDGET = 16 * 11 * 11
const SMALL_FP32_BATCH_TILE_BUDGET = 32 * 7 * 7
const LARGE_Y_READBACK_POINTS = 640

function batchSizeFor(size, bytesPerValue, modelName) {
  if (modelName === "katahex-20240812") {
    const budget = bytesPerValue === 2 ? MAIN_BATCH_POINT_BUDGET : MAIN_FP32_BATCH_POINT_BUDGET
    return Math.min(32, Math.max(1, Math.floor(budget / (size * size))))
  }
  const areaCap = Math.max(1, Math.floor((BATCH_POINT_BUDGET * 4) / (size * size * bytesPerValue)))
  if (modelName === "hex-b5-16") {
    if (bytesPerValue === 2) return Math.min(48, areaCap)
    const tiles = Math.ceil(size / COMPACT_OUTPUT_TILE) ** 2
    return Math.min(48, Math.max(1, Math.floor(SMALL_FP32_BATCH_TILE_BUDGET / tiles)))
  }
  return Math.min(MAX_SUBMISSION_SIZE, areaCap)
}

function bufferWithData(device, data, usage, half = false) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(data.length * (half ? 2 : data.BYTES_PER_ELEMENT) / 4) * 4),
    usage,
    mappedAtCreation: true,
  })
  try {
    const range = buffer.getMappedRange()
    if (!half) {
      new Uint8Array(range).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
    } else if (typeof Float16Array === "function") {
      new Float16Array(range, 0, data.length).set(data)
    } else {
      toHalf(data, new Uint16Array(range, 0, data.length))
    }
    buffer.unmap()
    return buffer
  } catch (error) {
    buffer.destroy()
    throw error
  }
}

function largestAttentionHeadCount(blocks) {
  let heads = 0
  for (const block of blocks) {
    heads = Math.max(heads, block.heads || 0, largestAttentionHeadCount(block.blocks || []))
  }
  return heads
}

function boardPointMaps(size, game) {
  const forPlayer = (transpose) => symmetriesFor(game).map((symmetry) =>
    Uint32Array.from({ length: size * size }, (_, point) => {
      const canonical = transpose ? (point % size) * size + Math.floor(point / size) : point
      return symmetryPoint(canonical, size, game, symmetry)
    }),
  )
  const first = forPlayer(false)
  return [first, game === "hex" ? forPlayer(true) : first]
}

class Graph {
  constructor(engine, size) {
    this.engine = engine
    this.device = engine.device
    this.size = size
    this.buffers = []
    this.tensors = []
    this.ops = []
    try {
      this.build()
    } catch (error) {
      this.destroy()
      throw error
    }
  }

  build() {
    const { engine, size } = this
    this.playablePoints = globalThis.BoardGeometry.playablePoints(engine.game, size)
    const model = engine.model
    this.packed = model.transformer
    this.points = this.packed ? this.playablePoints.length : size * size
    this.searchBatchSize = batchSizeFor(size, engine.bytesPerValue, engine.modelName)
    const attentionHeads = model.transformer ? largestAttentionHeadCount(model.blocks) : 0
    if (model.transformer) {
      // Bound the work per submission even when attention keeps scores in workgroup memory.
      const scoreBytes = this.points * this.points * attentionHeads * 4
      const budget = Math.min(ATTENTION_SCORE_BUDGET, this.device.limits.maxStorageBufferBindingSize)
      this.searchBatchSize = Math.min(this.searchBatchSize, Math.max(1, Math.floor(budget / scoreBytes)))
    }
    this.batchSize = Math.min(this.searchBatchSize, MAX_SUBMISSION_SIZE)
    this.useOnlineAttention = model.graphAttention || (engine.game === "y" && this.points >= ONLINE_Y_MIN_POINTS)
    this.useFusedYAttention = engine.half && engine.game === "y" && !this.useOnlineAttention
    if (engine.game === "y" && !engine.half && !this.useOnlineAttention) {
      this.attentionScores = this.tensor(this.points, this.points * attentionHeads, 4)
    }
    this.outputStride = this.points + 3
    const maps = boardPointMaps(size, engine.game)
    if (this.packed) {
      const indices = new Int32Array(size * size).fill(-1)
      this.playablePoints.forEach((point, index) => { indices[point] = index })
      this.policyMaps = maps.map((bySymmetry) => bySymmetry.map((map) => Int32Array.from(map, (point) => indices[point])))
      if (engine.game === "yavalath") {
        this.denseToPacked = { buffer: bufferWithData(this.device, indices, GPUBufferUsage.STORAGE) }
        this.buffers.push(this.denseToPacked.buffer)
      }
    } else this.policyMaps = maps
    this.inputMaps = model.graphAttention ? this.policyMaps : maps
    if (model.graphAttention) this.graphFeatures = graphBiasFeatures(size)
    const inputPoints = model.graphAttention ? this.points : size * size
    this.spatial = this.tensor(inputPoints, model.spatialChannels)
    this.global = this.tensor(1, model.globalChannels)

    let trunk
    // The sparse stem uses binary board features. Packed models also map the output to playable cells.
    if (!model.graphAttention) {
      const playable = this.packed
        ? { buffer: bufferWithData(this.device, Uint32Array.from(this.playablePoints), GPUBufferUsage.STORAGE) }
        : null
      if (playable) this.buffers.push(playable.buffer)
      trunk = this.operation(
        "sparseStem", this.spatial, model.initial.weights, this.tensor(this.points, model.initial.output),
        [size, this.points, model.initial.input, model.initial.output],
        [Math.ceil(this.points * model.initial.output / 256)], playable,
      )
    } else {
      trunk = this.conv(this.spatial, model.initial, model.graphAttention ? this.points : size)
    }
    const globalBias = this.conv(this.global, model.global)
    trunk = this.add(trunk, globalBias)
    for (const block of model.blocks) trunk = this.block(trunk, block)
    const tip = this.norm(trunk, model.tip)

    const policy = model.policy
    let local = this.conv(tip, policy.local)
    const poolConv = this.conv(tip, policy.pool)
    const poolAct = this.norm(poolConv, policy.poolNorm)
    const pooled = this.pool(poolAct, false)
    const policyBias = this.conv(pooled, policy.poolBias)

    local = this.norm(this.add(local, policyBias), policy.norm)
    const policyOut = this.conv(local, policy.output)

    const value = model.value
    const vConv = this.conv(tip, value.conv)
    const vAct = this.norm(vConv, value.norm)
    const vPool = this.pool(vAct, true)
    const hidden = this.conv(vPool, value.hidden)
    const hiddenAct = this.norm(hidden, value.hiddenNorm)
    const output = this.conv(hiddenAct, value.output)
    const valueOut = this.norm(output, value.outputNorm)

    const outputBytes = this.outputStride * this.batchSize * 4
    const readout = this.device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    this.buffers.push(readout)
    this.output = { buffer: readout }
    this.operation("readout", policyOut, valueOut, this.output, [this.points], [Math.ceil(this.outputStride / 64)])
    this.readback = this.device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    this.buffers.push(this.readback)
    this.allocateTensors()
  }

  tensor(points, channels, bytes = this.engine.bytesPerValue) {
    const count = points * channels
    const size = Math.ceil((count * bytes * this.batchSize) / 4) * 4
    const buffer = { size, first: Infinity, last: -1 }
    this.tensors.push(buffer)
    return { buffer, points, channels, count }
  }

  allocateTensors() {
    // Inputs are populated before the first dispatch.
    this.spatial.buffer.first = this.global.buffer.first = 0
    const slots = []
    const candidates = this.tensors.filter((tensor) => tensor.last >= 0)
      .sort((a, b) => b.size - a.size || a.first - b.first)
    for (const tensor of candidates) {
      // Reuse the smallest fitting slot. Tensors used by the same dispatch overlap.
      let slot
      for (const candidate of slots) {
        if (slot && slot.size <= candidate.size) continue
        if (candidate.tensors.every((other) => tensor.last < other.first || tensor.first > other.last)) slot = candidate
      }
      if (!slot) {
        slot = { size: tensor.size, tensors: [] }
        slots.push(slot)
      }
      slot.tensors.push(tensor)
    }
    for (const slot of slots) {
      const gpu = this.device.createBuffer({
        size: slot.size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
      this.buffers.push(gpu)
      for (const tensor of slot.tensors) tensor.gpu = gpu
    }
    this.bindOperations()
    this.spatial.buffer = this.spatial.buffer.gpu
    this.global.buffer = this.global.buffer.gpu
    this.tensors = null
  }

  bindOperations() {
    const paramBytes = 8 * Uint32Array.BYTES_PER_ELEMENT
    const alignment = this.device.limits.minUniformBufferOffsetAlignment
    const stride = Math.ceil(paramBytes / alignment) * alignment
    const params = this.device.createBuffer({
      size: (this.ops.length - 1) * stride + paramBytes,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    })
    this.buffers.push(params)
    const words = new Uint32Array(params.getMappedRange())
    this.ops.forEach((op, index) => words.set(op.data, index * stride / Uint32Array.BYTES_PER_ELEMENT))
    params.unmap()
    this.ops = this.ops.map(({ layout, resources, data, ...op }, index) => ({
      ...op,
      bindGroup: this.device.createBindGroup({
        layout,
        entries: [
          ...resources.map((buffer, binding) => ({ binding, resource: { buffer: buffer.gpu ?? buffer } })),
          { binding: resources.length, resource: { buffer: params, offset: index * stride, size: paramBytes } },
        ],
      }),
    }))
  }

  operation(kind, a, b, output, parameters, dispatch, extraInput = null) {
    const { pipeline, layout } = this.engine.pipelines[kind]
    const data = new Uint32Array(8)
    data.set(parameters)
    const resources = [a.buffer, b.buffer, output.buffer, ...(extraInput ? [extraInput.buffer] : [])]
    for (const buffer of resources) {
      if (buffer.first !== undefined) {
        buffer.first = Math.min(buffer.first, this.ops.length)
        buffer.last = this.ops.length
      }
    }
    this.ops.push({
      pipeline,
      layout,
      resources,
      dispatch,
      data,
    })
    return output
  }

  conv(input, layer, width = input.points === 1 ? 1 : this.packed ? input.points : this.size, residual = null) {
    const pointwise = layer.ky === 1 && layer.kx === 1
    const transformer = this.engine.model.transformer
    if (residual && transformer && pointwise && !layer.winograd && !layer.weights.compact &&
        input.buffer !== residual.buffer && input.points === residual.points && layer.output === residual.channels) {
      // Each invocation reads and updates its own skip vector, saving a graph tensor.
      const projected = this.pointwise(input, layer, residual, "pointwiseInPlace")
      if (projected) return projected
    }
    const output = this.tensor(input.points, layer.output)
    if (layer.winograd) {
      const compact = !this.engine.half
      const tileSize = compact ? COMPACT_OUTPUT_TILE : WINOGRAD_OUTPUT_TILE
      const coefficients = compact ? COMPACT_COEFFICIENTS : WINOGRAD_COEFFICIENTS
      const tileColumns = Math.ceil(width / tileSize)
      const tileRows = Math.ceil(input.points / width / tileSize)
      const tiles = tileColumns * tileRows
      const parameters = [width, input.points / width, layer.input, layer.output, tileColumns, tileRows, this.batchSize]
      const transformed = this.tensor(coefficients * tiles, layer.input)
      const multiplied = this.tensor(coefficients * tiles, layer.output, 4)
      this.operation(
        compact ? "compactWinogradInput" : "winogradInput",
        input, layer.winograd, transformed, parameters, [Math.ceil((tiles * layer.input) / (compact ? 256 : 64))],
      )
      const aligned = layer.input % 32 === 0 && layer.output % 32 === 0
      const multiply = compact ? "compactWinogradMultiply" : aligned ? "winogradMultiplyAligned" : "winogradMultiply"
      this.operation(multiply, transformed, layer.winograd, multiplied, parameters, [Math.ceil(layer.output / 32)])
      Object.assign(this.ops.at(-1), { rowsPerPosition: tiles, tileRows: 32, planes: coefficients })
      const inverse = compact ? "compactWinogradOutput" : residual ? "winogradOutputResidual" : "winogradOutput"
      this.operation(
        inverse, multiplied, compact ? layer.winograd : residual ?? layer.winograd, output, parameters,
        [Math.ceil((tiles * layer.output) / 256)],
      )
      if (!compact) return output
    } else if (layer.weights.compact) {
      this.operation(
        "compactPointwise", input, layer.weights, output,
        [width, input.points / width, layer.input, layer.output, 0, 0, this.batchSize],
        [Math.ceil(layer.output / 32)],
      )
      Object.assign(this.ops.at(-1), { rowsPerPosition: input.points, tileRows: 32, planes: 1 })
    } else {
      const separateSkip = transformer ? null : residual
      const kind = separateSkip ? "pointwiseWithSkip" : "pointwise"
      if (pointwise && (transformer || input.points >= 32) &&
          this.pointwise(input, layer, output, kind, separateSkip)) {
        if (residual && !separateSkip) return this.add(output, residual)
        return output
      }
      this.operation(
        pointwise ? "conv1x1" : "conv", input, layer.weights, output,
        [width, input.points / width, layer.input, layer.output, layer.ky, layer.kx, layer.dy, layer.dx],
        [Math.ceil(layer.output / 32), Math.ceil(input.points / 32)],
      )
    }
    if (!residual) return output
    return this.add(output, residual)
  }

  pointwise(input, layer, output, kind = "pointwise", extraInput = null, ropeHeads = 0) {
    if (layer.input % 4 || output.channels % 4) return null
    const [narrow, wide] = this.engine.pointwiseTiles
    this.operation(kind + "0", input, layer.weights, output, [input.points, layer.input, output.channels, this.batchSize, ropeHeads],
      [Math.ceil(output.channels / narrow.columns)], extraInput)
    const op = this.ops.at(-1)
    Object.assign(op, { rowsPerPosition: input.points, tileRows: narrow.points, planes: 1 })
    if (wide) {
      op.wide = {
        pipeline: this.engine.pipelines[kind + "1"].pipeline,
        columns: Math.ceil(output.channels / wide.columns),
        points: wide.points,
      }
    }
    return output
  }

  norm(input, layer) {
    if (layer.kind === "spatialRms") {
      const reductionCount = this.denseToPacked ? this.size * this.size * input.channels : input.count
      const tiles = Math.ceil(reductionCount / 1024)
      const partials = this.tensor(tiles, 1, 4)
      const epsilon = new Uint32Array(new Float32Array([layer.epsilon]).buffer)[0]
      const parameters = [
        input.count,
        input.channels,
        tiles,
        reductionCount,
        this.playablePoints.length,
        epsilon,
        layer.activation,
      ]
      this.operation("spatialRmsReduce", input, this.denseToPacked ?? layer.weights, partials, parameters, [tiles])
      const output = this.operation(
        "spatialRmsApply",
        input,
        layer.weights,
        this.tensor(input.points, input.channels),
        parameters,
        [Math.ceil(input.count / 64)],
        partials,
      )
      return output
    }
    const maskSize = this.engine.game === "y" && !this.packed && input.points !== 1 ? this.size : 0
    const diameter = this.engine.game === "yavalath" && !this.packed && input.points !== 1 ? this.size : 0
    const vector = !this.engine.model.transformer && input.channels % 4 === 0 && !maskSize && !diameter
    return this.operation(
      vector ? "normVector" : "norm",
      input,
      layer.weights,
      this.tensor(input.points, input.channels),
      [input.count, input.channels, layer.activation, maskSize, diameter],
      [Math.ceil(input.count / (vector ? 256 : 64))],
    )
  }

  add(input, bias) {
    const vector = !this.engine.model.transformer && input.count % 4 === 0 && bias.count % 4 === 0
    return this.operation(
      vector ? "addVector" : "add",
      input,
      bias,
      this.tensor(input.points, input.channels),
      [input.count, bias.count],
      [Math.ceil(input.count / (vector ? 256 : 64))],
    )
  }

  pool(input, valueHead) {
    const playable = this.playablePoints.length
    return this.operation(
      "pool",
      input,
      this.spatial,
      this.tensor(1, input.channels * 3),
      [input.points, input.channels, Number(valueHead), playable, this.engine.model.spatialChannels, Number(this.packed)],
      [Math.ceil(input.channels / 64)],
    )
  }

  normConv(input, layer) {
    const norm = this.norm(input, layer.norm)
    return this.conv(norm, layer.conv)
  }

  rms(input, norm) {
    const epsilon = new Uint32Array(new Float32Array([norm.epsilon]).buffer)[0]
    return this.operation(
      "rms",
      input,
      norm.weights,
      this.tensor(input.points, input.channels),
      [input.points, input.channels, epsilon],
      [Math.ceil(input.points / RMS_POINTS_PER_GROUP)],
    )
  }

  attention(input, block) {
    const normalized = this.rms(input, block.norm)
    const rotateQKV = this.useOnlineAttention && this.engine.half && !this.engine.model.graphAttention
    let table
    if (this.engine.model.graphAttention) { table = graphAttentionBias(this.graphFeatures, block) } else {
      table = new Float32Array(block.heads * 16 * input.points * 2)
      for (let head = 0; head < block.heads; head++) {
        for (let pair = 0; pair < 16; pair++) {
          const headPair = head * 16 + pair
          const fx = block.ropeFreqs[headPair * 2]
          const fy = block.ropeFreqs[headPair * 2 + 1]
          for (let point = 0; point < input.points; point++) {
            const coordinate = this.packed ? this.playablePoints[point] : point
            const angle = Math.fround(
              Math.fround((coordinate % this.size) * fx) + Math.fround(Math.floor(coordinate / this.size) * fy),
            )
            const offset = (rotateQKV
              ? point * block.heads * 16 + headPair
              : headPair * input.points + point) * 2
            table[offset] = Math.cos(angle)
            table[offset + 1] = Math.sin(angle)
          }
        }
      }
    }
    const rope = { buffer: bufferWithData(this.device, table, GPUBufferUsage.STORAGE) }
    this.buffers.push(rope.buffer)
    const joined = rotateQKV
      ? this.pointwise(
          normalized, block.qkv, this.tensor(input.points, block.qkv.output), "pointwiseRoPE", rope, block.heads,
        )
      : this.conv(normalized, block.qkv)
    const attended = this.tensor(input.points, input.channels)
    if (this.engine.game === "yavalath") {
      this.operation(
        "fusedYavalathAttention", joined, rope, attended, [input.points, block.heads, this.size * this.size],
        [Math.ceil(input.points / YAVALATH_ATTENTION_QUERY_ROWS), block.heads], this.denseToPacked,
      )
    } else if (this.useOnlineAttention) {
      if (!this.engine.half && !this.engine.model.graphAttention) {
        this.operation(
          "yRotateQK", this.spatial, rope, joined,
          [input.points, block.heads], [Math.ceil(input.points * block.heads / 4)],
        )
      }
      this.operation(
        "yAttentionOnline", joined, rope, attended, [input.points, block.heads],
        [Math.ceil(input.points / ONLINE_Y_QUERY_ROWS), block.heads],
      )
    } else if (this.useFusedYAttention) {
      const columns = Math.ceil(input.points / 4)
      const keys = this.operation(
        "yAttentionKeys", joined, rope, this.tensor(columns * 4, input.channels),
        [input.points, block.heads, columns], [Math.ceil(columns * input.channels / 64)],
      )
      this.operation(
        "fusedYAttention", joined, rope, attended, [input.points, block.heads, columns],
        [Math.ceil(input.points / Y_ATTENTION_QUERY_ROWS), block.heads], keys,
      )
    } else {
      const scores = this.operation(
        "attentionScores",
        joined,
        rope,
        this.attentionScores,
        [input.points, block.heads],
        [Math.ceil(input.points / 32), Math.ceil(input.points / 16) * block.heads],
      )
      this.operation(
        "attentionSoftmax", this.spatial, this.global, scores,
        [input.points, block.heads], [input.points, block.heads],
      )
      this.operation(
        "attentionValues", scores, joined, attended,
        [input.points, block.heads], [Math.ceil(input.points / 16), block.heads],
      )
    }
    return this.conv(attended, block.output, undefined, input)
  }

  ffn(input, block) {
    const normalized = this.rms(input, block.norm)
    const activated = this.pointwise(normalized, block.up, this.tensor(input.points, block.hidden), "pointwiseSwiGLU")
    return this.conv(activated, block.output, undefined, input)
  }

  block(input, block) {
    if (block.kind === "attention") return this.attention(input, block)
    if (block.kind === "ffn") return this.ffn(input, block)
    let mid
    if (block.pool) {
      const pre = this.norm(input, block.pre.norm)
      const local = this.conv(pre, block.pre.conv)
      const poolConv = this.conv(pre, block.pool.conv)
      const poolAct = this.norm(poolConv, block.pool.norm)
      const pooled = this.pool(poolAct, false)
      const bias = this.conv(pooled, block.pool.matrix)
      mid = this.add(local, bias)
    } else { mid = this.normConv(input, block.pre) }
    if (block.blocks) {
      for (const child of block.blocks) mid = this.block(mid, child)
    }
    const post = this.norm(mid, block.post.norm)
    return this.conv(post, block.post.conv, undefined, input)
  }

  async evaluate(states) {
    const model = this.engine.model
    // Board features are binary, so encode 1.0 directly in the input format.
    const InputArray = this.engine.half ? Uint16Array : Float32Array
    const one = this.engine.half ? 0x3c00 : 1
    const spatial = new InputArray(this.spatial.count * states.length)
    const global = new InputArray(model.globalChannels * states.length)
    states.forEach(({ board, toPlay, symmetry = 0 }, batch) => {
      const map = this.inputMaps[toPlay - 1][symmetry]
      const start = batch * this.spatial.count
      if (model.graphAttention) {
        const features = bentYInput(board, this.size, toPlay)
        this.playablePoints.forEach((point, index) => {
          const offset = start + map[point] * model.spatialChannels
          for (let channel = 0; channel < model.spatialChannels; channel++) {
            spatial[offset + channel] = features[index * model.spatialChannels + channel] * one
          }
        })
      } else {
        for (let i = 0; i < board.length; i++) {
          const offset = start + map[i] * model.spatialChannels
          if (board[i] !== 3) spatial[offset] = one
          if (board[i] === 1 || board[i] === 2) spatial[offset + (board[i] === toPlay ? 1 : 2)] = one
        }
      }
      global[batch * model.globalChannels] = Number(toPlay === 2) * one
    })
    const write = (buffer, data) => {
      // writeBuffer requires four-byte alignment.
      if (data.byteLength % 4 === 0) {
        this.device.queue.writeBuffer(buffer, 0, data)
        return
      }
      const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4)
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
      this.device.queue.writeBuffer(buffer, 0, padded)
    }
    write(this.spatial.buffer, spatial)
    write(this.global.buffer, global)
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    for (const op of this.ops) {
      const rows = (op.rowsPerPosition ?? 0) * states.length
      const wide = rows >= POINTWISE_ROWS_FOR_WIDE_TILE ? op.wide : null
      pass.setPipeline(wide?.pipeline ?? op.pipeline)
      pass.setBindGroup(0, op.bindGroup)
      pass.dispatchWorkgroups(
        wide?.columns ?? op.dispatch[0],
        rows ? Math.ceil(rows / (wide?.points ?? op.tileRows)) : op.dispatch[1] ?? 1,
        op.planes ?? states.length,
      )
    }
    pass.end()
    const stride = this.outputStride
    const byteLength = states.length * stride * 4
    encoder.copyBufferToBuffer(this.output.buffer, 0, this.readback, 0, byteLength)
    this.device.queue.submit([encoder.finish()])
    const mapping = this.readback.mapAsync(GPUMapMode.READ, 0, byteLength)
    // Submissions advance completion on implementations that poll pending mappings.
    // B18 and large Y boards spend longer on the GPU and favor fewer empty submissions.
    const longSubmission = this.engine.modelName === "katahex-20240812" ||
      (this.engine.game === "y" && this.points >= LARGE_Y_READBACK_POINTS)
    const pollMs = longSubmission ? 8 : 4
    const timer = setInterval(() => this.device.queue.submit([]), pollMs)
    try {
      await mapping
    } finally {
      clearInterval(timer)
    }
    const values = new Float32Array(this.readback.getMappedRange(0, byteLength).slice(0))
    this.readback.unmap()
    if (values.some((value) => !Number.isFinite(value))) {
      throw new Error("The GPU returned a non-finite evaluation. Analysis stopped.")
    }
    return states.map(({ toPlay, symmetry = 0 }, i) => {
      const rawPolicy = values.subarray(i * stride, (i + 1) * stride - 3)
      const remap = this.packed || symmetry || (this.engine.game === "hex" && toPlay === 2)
      let policy = rawPolicy
      if (remap) {
        const map = this.policyMaps[toPlay - 1][symmetry]
        policy = new Float32Array(map.length)
        for (let i = 0; i < map.length; i++) policy[i] = map[i] < 0 ? -5000 : rawPolicy[map[i]]
      }
      return { policy, value: values.subarray((i + 1) * stride - 3, (i + 1) * stride) }
    })
  }

  destroy() {
    for (const buffer of this.buffers) buffer.destroy()
  }
}

async function uploadModel(engine, archive, signal) {
  const { model, half, device } = engine
  const scratch = tensorScratch()
  const useWasm = half && engine.modelName === "katahex-20240812"
  let transform
  let nextYield = performance.now() + 32
  const upload = async (object) => {
    // Yield to worker messages even when the entire download is buffered.
    if (performance.now() >= nextYield) {
      await new Promise((resolve) => setTimeout(resolve, 0))
      signal.throwIfAborted()
      nextYield = performance.now() + 32
    }
    const winograd =
      !model.transformer &&
      object.ky === 3 &&
      object.kx === 3 &&
      object.dy === 1 &&
      object.dx === 1 &&
      object.input >= 32 &&
      object.output >= 32 &&
      object.input % 4 === 0 &&
      object.output % 4 === 0
    const compact = !half && !model.transformer && object.weights?.type === "q7" &&
      (winograd || (object.ky === 1 && object.kx === 1 && object.input % 4 === 0 && object.output % 4 === 0))
    if (compact) {
      const { codes, scales } = await archive.quantizedTensor(object.weights)
      const data = scratch("compact", Uint32Array, codes.length / 4 + scales.length)
      new Uint8Array(data.buffer, data.byteOffset, codes.length).set(codes)
      const outputScales = new Float32Array(data.buffer, data.byteOffset + codes.length, scales.length)
      const scale = winograd ? 0.25 : 1
      for (let i = 0; i < scales.length; i++) outputScales[i] = scales[i] * scale
      const buffer = bufferWithData(device, data, GPUBufferUsage.STORAGE)
      engine.weights.push(buffer)
      if (winograd) {
        object.winograd = { buffer }
        delete object.weights
      } else object.weights = { buffer, compact: true }
      return
    }
    if (useWasm && transform !== null && winograd && object.weights?.type === "q7") {
      const quantized = await archive.quantizedTensor(object.weights)
      // Choose once per model to use a single workspace. A cold module never delays upload.
      transform ??= createWinogradWeightTransform()
      if (transform) {
        let weights
        try {
          weights = transform(quantized.codes, quantized.scales)
        } catch {
          // The workspace can fail to grow on memory-constrained devices.
          transform = null
        }
        if (weights) {
          const buffer = bufferWithData(device, weights, GPUBufferUsage.STORAGE)
          engine.weights.push(buffer)
          object.winograd = { buffer, count: weights.length }
          delete object.weights
          return
        }
      }
      object.weights = await archive.decodeTensor(object.weights, quantized)
    }
    if (half && winograd && object.weights?.type) object.weights = await archive.decodeTensor(object.weights)
    if (half && winograd && object.weights instanceof Float32Array) {
      object.winograd = transformWinogradWeights(
        object,
        scratch("winograd", Float32Array, WINOGRAD_COEFFICIENTS * object.input * object.output),
        scratch("winogradColumns", Float64Array, 3 * object.input * object.output),
        scratch("winogradOtherColumns", Float64Array, 3 * object.input * object.output),
      )
      delete object.weights
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
      if (value?.type) object[key] = value = await archive.decodeTensor(value)
      if (value instanceof Float32Array) {
        const storeHalf = half && key !== "scales" && object.kind !== "rms"
        const buffer = bufferWithData(device, value, GPUBufferUsage.STORAGE, storeHalf)
        engine.weights.push(buffer)
        object[key] = { buffer, count: value.length }
      } else if (value instanceof Uint32Array) {
        const buffer = bufferWithData(device, value, GPUBufferUsage.STORAGE)
        engine.weights.push(buffer)
        object[key] = { buffer, count: value.length }
      } else if (value && typeof value === "object") await upload(value)
    }
  }
  await upload(model)
}

export class Network {
  static async create(adapter, archive, onError, modelName, game, signal) {
    const model = archive.model
    const half = adapter.features.has("shader-f16")
    const device = await adapter.requestDevice({ requiredFeatures: half ? ["shader-f16"] : [] })
    const engine = new Network(device, model, half, modelName, game)
    try {
      signal.throwIfAborted()
      device.lost.then(({ reason, message }) => {
        if (reason !== "destroyed") onError(new Error(`GPU connection lost: ${message}. Press Start to retry.`))
      })
      device.addEventListener("uncapturederror", (event) => onError(event.error))
      const bindGroupLayout = device.createBindGroupLayout({
        entries: ["read-only-storage", "read-only-storage", "storage", "uniform"].map((type, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type },
        })),
      })
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] })
      const extraInputBindGroupLayout = device.createBindGroupLayout({
        entries: ["read-only-storage", "read-only-storage", "storage", "read-only-storage", "uniform"].map(
          (type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } }),
        ),
      })
      const extraInputPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [extraInputBindGroupLayout] })
      const halfY = half && game === "y"
      const kernels = {
        ...createKernels(half, model.transformer, model.graphAttention),
        ...createPointwiseKernels(half, engine.pointwiseTiles, model.transformer, halfY),
        ...(!model.transformer
          ? (half ? createWinogradKernels(modelName === "katahex-20240812") : createCompactWinogradKernels())
          : {}),
        ...(model.transformer ? createTransformerKernels(half, game, model.graphAttention) : {}),
        ...(game === "y" || model.graphAttention ? createOnlineYAttention(half, model.graphAttention) : {}),
        ...(halfY ? { yAttentionKeys: createYAttentionKeys(), fusedYAttention: createFusedYAttention() } : {}),
        ...(game === "yavalath" ? { fusedYavalathAttention: createYavalathAttention(half) } : {}),
      }
      const extraInputKernels = new Set([
        "pointwiseWithSkip0", "pointwiseRoPE0", "pointwiseRoPE1", "spatialRmsApply", "fusedYAttention",
        "fusedYavalathAttention",
      ])
      if (model.transformer) extraInputKernels.add("sparseStem")
      engine.pipelines = Object.fromEntries(
        await Promise.all(
          Object.entries(kernels).map(async ([name, code]) => {
            const extraInput = extraInputKernels.has(name)
            const pipeline = await device.createComputePipelineAsync({
              layout: extraInput ? extraInputPipelineLayout : pipelineLayout,
              compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
            })
            return [name, { pipeline, layout: extraInput ? extraInputBindGroupLayout : bindGroupLayout }]
          }),
        ),
      )
      signal.throwIfAborted()
      await uploadModel(engine, archive, signal)
      await archive.finish()
      signal.throwIfAborted()
      return engine
    } catch (error) {
      engine.destroy()
      throw error
    }
  }

  constructor(device, model, half, modelName, game) {
    this.half = half
    this.modelName = modelName
    this.game = game
    this.pointwiseTiles = pointwiseTiles(half, model.transformer)
    this.runtime = half ? "f16" : "f32"
    this.runtimeHelp = model.transformer
      ? `${half ? 16 : 32}-bit floating-point tensors and projections; 32-bit attention and normalization reductions.`
      : half
        ? "16-bit floating-point tensors and Winograd convolutions."
        : "32-bit floating-point tensors and Winograd convolutions with packed weights."
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

  async evaluate(states, cancelled) {
    this.prepare(states[0].size)
    const results = []
    const submissionSize = this.graph.batchSize
    for (let offset = 0; offset < states.length; offset += submissionSize) {
      if (cancelled()) return null
      results.push(...(await this.graph.evaluate(states.slice(offset, offset + submissionSize))))
      // Let worker messages run before submitting another GPU batch.
      if (offset + submissionSize < states.length) await new Promise((resolve) => setTimeout(resolve, 0))
    }
    return results
  }

  destroy() {
    this.graph?.destroy()
    for (const buffer of this.weights) buffer.destroy()
    this.device.destroy()
  }
}
