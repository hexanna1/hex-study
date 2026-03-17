// SPDX-License-Identifier: MIT
// Network execution adapted from cpp/neuralnet/eigenbackend.cpp.
import { graphBiasFeatures, graphAttentionBias } from "./benty.js"
import { bentPrefixKernels, createBentPrefix, packBentYRows } from "./bent-prefix.js"
import "../board_geometry.js"
import { createTransformerKernels, RMS_POINTS_PER_GROUP } from "./transformer_kernels.js"
import {
  createOnlineYAttention, createYAttentionKeys, ONLINE_Y_MIN_POINTS, ONLINE_Y_QUERY_ROWS, rotaryPhases,
} from "./y_attention.js"
import { createFusedAttention, FUSED_ATTENTION_QUERY_ROWS } from "./fused_attention.js"
import { createPointwiseKernels, POINTWISE_ROWS_FOR_WIDE_TILE, pointwiseTiles } from "./pointwise_kernels.js"
import { createKernels } from "./kernels.js"
import { WRITE } from "./kernel.js"
import {
  createWinogradKernels,
  countWinogradFilters,
  WINOGRAD_OUTPUT_TILE,
  WINOGRAD_COEFFICIENTS,
} from "./winograd_kernels.js"
import { COMPACT_COEFFICIENTS, COMPACT_OUTPUT_TILE, createCompactWinogradKernels } from "./compact_winograd.js"
import { createWinogradFilter } from "./winograd_filters.js"
import { bufferWithData, mappedBuffer, prepareHalfUpload, uploadModel } from "./weight_upload.js"
import { symmetriesFor, symmetryPoint } from "./symmetry.js"

const MAX_SUBMISSION_SIZE = 64
const BATCH_POINT_BUDGET = 32 * 19 * 19
const ATTENTION_COMPARISON_BUDGET = 16 * 1024 * 1024
const LARGE_CNN_WEIGHT_BYTES = 64 * 1024 * 1024
const MIN_FLOAT_WINOGRAD_MACS = 128 * 1024 * 1024
// Limit the larger model's GPU work per submission for pointer responsiveness.
const MAIN_BATCH_POINT_BUDGET = 7 * 19 * 19
const MAIN_FP32_BATCH_POINT_BUDGET = 16 * 11 * 11
const SMALL_FP32_BATCH_TILE_BUDGET = 32 * 7 * 7
const LARGE_Y_READBACK_POINTS = 640

function useFloatWinograd(size, engine) {
  if (engine.half || engine.model.transformer) return false
  const smallTiles = Math.ceil(size / COMPACT_OUTPUT_TILE) ** 2
  const largeTiles = Math.ceil(size / WINOGRAD_OUTPUT_TILE) ** 2
  const products = WINOGRAD_COEFFICIENTS * largeTiles
  // Count edge padding when requiring a one-third reduction in products.
  return largeTiles >= 9 && products * engine.winogradFilterCount >= MIN_FLOAT_WINOGRAD_MACS &&
    products * 3 <= COMPACT_COEFFICIENTS * smallTiles * 2
}

function batchSizeFor(size, engine) {
  if (engine.largeCNN) {
    const budget = engine.half ? MAIN_BATCH_POINT_BUDGET : MAIN_FP32_BATCH_POINT_BUDGET
    return Math.min(32, Math.max(1, Math.floor(budget / (size * size))))
  }
  const areaCap = Math.max(1, Math.floor((BATCH_POINT_BUDGET * 4) / (size * size * engine.bytesPerValue)))
  if (!engine.model.transformer) {
    if (engine.half) return Math.min(48, areaCap)
    const tiles = Math.ceil(size / COMPACT_OUTPUT_TILE) ** 2
    return Math.min(48, Math.max(1, Math.floor(SMALL_FP32_BATCH_TILE_BUDGET / tiles)))
  }
  return Math.min(MAX_SUBMISSION_SIZE, areaCap)
}

function scratchSlots(tensors) {
  const slots = []
  for (const tensor of tensors) {
    // Tensors used by the same dispatch overlap.
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
  return slots
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
    Uint16Array.from({ length: size * size }, (_, point) => {
      const canonical = transpose ? (point % size) * size + Math.floor(point / size) : point
      return symmetryPoint(canonical, size, game, symmetry)
    }),
  )
  const first = forPlayer(false)
  return [first, game === "hex" ? forPlayer(true) : first]
}

class Graph {
  constructor(engine, size, largeBatch, batchWidth, floatWinograd) {
    this.engine = engine
    this.device = engine.device
    this.size = size
    this.largeBatch = largeBatch
    this.floatWinograd = floatWinograd
    this.buffers = []
    this.tensors = []
    this.ops = []
    try {
      this.build(batchWidth)
    } catch (error) {
      this.destroy()
      throw error
    }
  }

  build(batchWidth) {
    const { engine, size } = this
    const model = engine.model
    this.packed = model.transformer
    this.playablePoints = this.packed ? globalThis.BoardGeometry.playablePoints(engine.game, size) : null
    this.points = this.playablePoints?.length ?? size * size
    this.searchBatchSize = batchWidth
    const longSubmission = engine.largeCNN || (engine.game === "y" && this.points >= LARGE_Y_READBACK_POINTS)
    this.pollMs = longSubmission ? 8 : 4
    const attentionHeads = model.transformer ? largestAttentionHeadCount(model.blocks) : 0
    if (model.transformer) {
      const comparisons = this.points * this.points * attentionHeads
      this.searchBatchSize = Math.min(this.searchBatchSize, Math.max(1, Math.floor(ATTENTION_COMPARISON_BUDGET / comparisons)))
    }
    this.batchSize = Math.min(this.searchBatchSize * (this.largeBatch ? 2 : 1), MAX_SUBMISSION_SIZE)
    this.useOnlineAttention = model.graphAttention || (engine.game === "y" && this.points >= ONLINE_Y_MIN_POINTS)
    this.outputStride = this.points + 3
    const maps = boardPointMaps(size, engine.game)
    if (this.packed) {
      const indices = new Int32Array(size * size).fill(-1)
      this.playablePoints.forEach((point, index) => { indices[point] = index })
      const packed = maps[0].map((map) => Int16Array.from(map, (point) => indices[point]))
      this.policyMaps = [packed, packed]
    } else this.policyMaps = maps
    this.inputMaps = model.graphAttention ? this.policyMaps : maps
    let trunk
    let firstBlock = 0
    if (model.graphAttention) {
      this.graphFeatures = graphBiasFeatures(size)
      const biasIndices = bufferWithData(this.device, this.graphFeatures.packedPairs, GPUBufferUsage.STORAGE)
      this.biasIndices = { buffer: biasIndices }
      this.buffers.push(biasIndices)
      this.spatial = this.tensor(this.points, 1, Uint32Array.BYTES_PER_ELEMENT)
      const { initial, inner, qkv } = engine.bentPrefix
      const scatterInitial = this.tensor(this.points, initial.channels)
      const scatterInner = this.tensor(this.points, inner.channels)
      this.operation(
        "scatterBentPrefix", [initial, this.spatial, scatterInitial, inner, scatterInner],
        [this.points, initial.channels, inner.channels], [Math.ceil(this.points * initial.channels / 256)],
      )
      trunk = this.block(scatterInitial, model.blocks[0], { inner: scatterInner, qkv })
      firstBlock = 1
    } else {
      this.spatial = this.tensor(size * size, 1, Uint32Array.BYTES_PER_ELEMENT)
      this.global = this.tensor(1, model.globalChannels)
      const playable = this.packed
        ? { buffer: bufferWithData(this.device, Uint32Array.from(this.playablePoints), GPUBufferUsage.STORAGE) }
        : null
      const stemBindings = [this.spatial, model.initial.weights, this.tensor(this.points, model.initial.output)]
      if (playable) {
        this.buffers.push(playable.buffer)
        stemBindings.push(playable)
      }
      trunk = this.operation(
        "sparseStem", stemBindings,
        [size, this.points, model.initial.input, model.initial.output],
        [Math.ceil(this.points * model.initial.output / 256)],
      )
      trunk = this.add(trunk, this.conv(this.global, model.global))
    }
    for (let index = firstBlock; index < model.blocks.length; index++) {
      trunk = this.block(trunk, model.blocks[index])
    }
    const tip = this.norm(trunk, model.tip, false)

    const policyStart = this.ops.length
    const policy = model.policy
    let local = this.conv(tip, policy.local)
    const poolConv = this.conv(tip, policy.pool)
    const poolAct = this.norm(poolConv, policy.poolNorm)
    const pooled = this.pool(poolAct, false)
    const policyBias = this.conv(pooled, policy.poolBias)

    local = this.norm(this.add(local, policyBias), policy.norm)
    const policyOut = this.conv(local, policy.output)

    const valueStart = this.ops.length
    const value = model.value
    const vConv = this.conv(tip, value.conv)
    const vAct = this.norm(vConv, value.norm)
    const vPool = this.pool(vAct, true)
    const hidden = this.conv(vPool, value.hidden)
    const hiddenAct = this.norm(hidden, value.hiddenNorm)
    const valueEnd = this.ops.length

    const outputBytes = this.outputStride * this.batchSize * 4
    const readout = this.device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    this.buffers.push(readout)
    this.output = { buffer: readout }
    const readoutBindings = [hiddenAct, this.output, value.output.weights, value.outputNorm.weights]
    this.operation(
      "readout", [policyOut, ...readoutBindings], [this.points, hiddenAct.channels], [Math.ceil(this.outputStride / 64)],
    )
    // A zero point count skips policy output.
    this.operation("readout", [hiddenAct, ...readoutBindings], [0, hiddenAct.channels], [1])
    this.readback = this.device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    this.buffers.push(this.readback)
    this.allocateTensors()
    const valueReadout = this.ops.pop()
    this.valueOps = [...this.ops.slice(0, policyStart), ...this.ops.slice(valueStart, valueEnd), valueReadout]
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
    this.spatial.buffer.first = 0
    if (this.global) this.global.buffer.first = 0
    const tensors = this.tensors.filter((tensor) => tensor.last >= 0)
    const bySize = (a, b) => b.size - a.size
    // Equal-size tensors can favor either first use or longer lifetimes.
    const earliest = scratchSlots([...tensors].sort((a, b) => bySize(a, b) || a.first - b.first))
    const longest = scratchSlots(tensors.sort((a, b) => bySize(a, b) || (b.last - b.first) - (a.last - a.first)))
    const bytes = (slots) => slots.reduce((total, slot) => total + slot.size, 0)
    const slots = bytes(longest) < bytes(earliest) ? longest : earliest
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
    if (this.global) this.global.buffer = this.global.buffer.gpu
    this.tensors = null
  }

  bindOperations() {
    const paramBytes = 8 * Uint32Array.BYTES_PER_ELEMENT
    const alignment = this.device.limits.minUniformBufferOffsetAlignment
    const stride = Math.ceil(paramBytes / alignment) * alignment
    const size = (this.ops.length - 1) * stride + paramBytes
    const params = mappedBuffer(this.device, size, GPUBufferUsage.UNIFORM, (range) => {
      const words = new Uint32Array(range)
      this.ops.forEach((op, index) => words.set(op.data, index * stride / Uint32Array.BYTES_PER_ELEMENT))
    })
    this.buffers.push(params)
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

  operation(kind, bindings, parameters, dispatch) {
    this.pendingWinogradOutput = null
    this.pendingPointwiseOutput = null
    const { pipeline, layout, output } = this.engine.pipelines[kind]
    const data = new Uint32Array(8)
    data.set(parameters)
    const resources = bindings.map(({ buffer }) => buffer)
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
    return bindings[output]
  }

  conv(input, layer, residual = null) {
    const transformer = this.engine.model.transformer
    const canWriteSkip = residual && input.buffer !== residual.buffer && input.points === residual.points &&
      layer.output === residual.channels
    if (canWriteSkip && transformer && !layer.winograd && !layer.weights.compact) {
      const projected = this.pointwise(input, layer, residual, "pointwiseInPlace")
      if (projected) return projected
    }
    const compactResidual = Boolean(canWriteSkip && layer.weights?.compact)
    const output = compactResidual ? residual : this.tensor(input.points, layer.output)
    if (layer.winograd) {
      const floatTransform = this.floatWinograd && layer.input % 32 === 0 && layer.output % 32 === 0
      const compact = !this.engine.half && !floatTransform
      const width = this.size
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
        [input, transformed], parameters, [Math.ceil((tiles * layer.input) / (compact ? 256 : 64))],
      )
      let weights = layer.winograd
      if (floatTransform) {
        const filters = layer.input * layer.output
        const allocation = { size: filters * WINOGRAD_COEFFICIENTS * 4, first: Infinity, last: -1 }
        this.tensors.push(allocation)
        weights = this.operation("restoreWinograd32", [weights, { buffer: allocation }],
          [filters / 4, layer.output / 4], [Math.ceil(filters / 256)])
        this.ops.at(-1).planes = 1
      }
      const multiply = compact ? "compactWinogradMultiply" : "winogradMultiply"
      this.operation(multiply, [transformed, weights, multiplied], parameters, [Math.ceil(layer.output / 32)])
      const rowsPerWorkgroup = compact ? this.engine.compactTileRows : 32
      Object.assign(this.ops.at(-1), {
        rowsPerPosition: tiles,
        tileRows: rowsPerWorkgroup,
        planes: compact && this.engine.pairCompactPlanes ? coefficients / 2 : coefficients,
      })
      const inverse = (compact ? "compactWinogradOutput" : "winogradOutput") + (residual ? "Residual" : "")
      const bindings = residual ? [multiplied, residual, output] : [multiplied, output]
      const dispatch = [Math.ceil((tiles * layer.output) / 256)]
      this.operation(inverse, bindings, parameters, dispatch)
      this.pendingWinogradOutput = {
        inverse, multiplied, residual, output, parameters, dispatch,
        keepResidual: !compact && this.engine.largeCNN && Boolean(residual),
      }
      return output
    } else if (layer.weights.compact) {
      this.operation(
        "compactPointwise", [input, layer.weights, output],
        [input.points, 1, layer.input, layer.output, 0, 0, this.batchSize, Number(compactResidual)],
        [Math.ceil(layer.output / 32)],
      )
      Object.assign(this.ops.at(-1), { rowsPerPosition: input.points, tileRows: this.engine.compactTileRows, planes: 1 })
      if (compactResidual) return output
    } else {
      const separateSkip = !transformer && this.engine.half ? residual : null
      const kind = separateSkip ? "pointwiseWithSkip" : "pointwise"
      if ((transformer || input.points >= 32) &&
          this.pointwise(input, layer, output, kind, { extraInput: separateSkip })) {
        if (residual && !separateSkip) return this.add(output, residual)
        if (this.engine.largeCNN && this.engine.half) {
          this.pendingPointwiseOutput = { input, layer, output, kind, separateSkip }
        }
        return output
      }
      this.operation(
        "conv1x1", [input, layer.weights, output],
        [input.points, layer.input, layer.output],
        [Math.ceil(layer.output / 32), Math.ceil(input.points / 32)],
      )
    }
    if (!residual) return output
    return this.add(output, residual)
  }

  pointwise(input, layer, output, kind = "pointwise", {
    extraInput, ropeHeads = 0, affine, unnormalized,
  } = {}) {
    if (layer.input % 4 || output.channels % 4) return null
    const [narrow, wide] = this.engine.pointwiseTiles
    const bindings = [input, layer.weights, output]
    if (extraInput) bindings.push(extraInput)
    if (affine) bindings.push(affine.weights)
    if (unnormalized) bindings.push(unnormalized)
    const parameter = affine?.activation ?? ropeHeads
    this.operation(kind + "0", bindings, [input.points, layer.input, output.channels, this.batchSize, parameter],
      [Math.ceil(output.channels / narrow.columns)])
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

  norm(input, layer, keepInput = true) {
    if (layer.kind === "spatialRms") {
      const tiles = Math.ceil(input.count / 1024)
      const partials = this.tensor(tiles, 1, 4)
      const epsilon = new Uint32Array(new Float32Array([layer.epsilon]).buffer)[0]
      const parameters = [
        input.count,
        input.channels,
        tiles,
        epsilon,
        layer.activation,
      ]
      this.operation("spatialRmsReduce", [input, partials], parameters, [tiles])
      return this.operation(
        "spatialRmsApply",
        [input, layer.weights, this.tensor(input.points, input.channels), partials],
        parameters,
        [Math.ceil(input.count / 256)],
      )
    }
    const pointwise = this.pendingPointwiseOutput
    if (keepInput && pointwise?.output === input) {
      this.ops.pop()
      const normalized = this.tensor(input.points, input.channels)
      return this.pointwise(pointwise.input, pointwise.layer, normalized, pointwise.kind + "NormKeep", {
        extraInput: pointwise.separateSkip,
        affine: layer,
        unnormalized: input,
      })
    }
    const pending = this.pendingWinogradOutput
    if (pending?.output === input && (!keepInput || pending.keepResidual)) {
      // FP16 narrows both the convolution and the residual sum before the affine.
      this.ops.pop()
      const normalized = this.tensor(input.points, input.channels)
      const { inverse, multiplied, residual, parameters, dispatch } = pending
      const bindings = residual ? [multiplied, residual, normalized] : [multiplied, normalized]
      bindings.push(layer.weights)
      if (keepInput) {
        bindings.push(input)
      } else {
        input.buffer.first = Infinity
        input.buffer.last = -1
      }
      return this.operation(inverse + (keepInput ? "NormKeep" : "Norm"), bindings,
        [...parameters, layer.activation], dispatch)
    }
    return this.operation(
      "normVector",
      [input, layer.weights, this.tensor(input.points, input.channels)],
      [input.count, input.channels, layer.activation],
      [Math.ceil(input.count / 256)],
    )
  }

  add(input, bias) {
    return this.operation(
      "add",
      [input, bias, this.tensor(input.points, input.channels)],
      [input.count, bias.count],
      [Math.ceil(input.count / 256)],
    )
  }

  pool(input, valueHead) {
    return this.operation(
      "pool",
      [input, this.tensor(1, input.channels * 3)],
      [input.points, input.channels, Number(valueHead)],
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
      [input, norm.weights, this.tensor(input.points, input.channels)],
      [input.points, input.channels, epsilon],
      [Math.ceil(input.points / RMS_POINTS_PER_GROUP)],
    )
  }

  attention(input, block, qkvTable = null) {
    const rotateQKV = this.useOnlineAttention && this.engine.half && !this.engine.model.graphAttention
    const table = this.engine.model.graphAttention
      ? graphAttentionBias(this.graphFeatures, block)
      : rotaryPhases(this.size, this.playablePoints, block.heads * 16, block.ropeFreqs, rotateQKV)
    const rope = { buffer: bufferWithData(this.device, table, GPUBufferUsage.STORAGE) }
    this.buffers.push(rope.buffer)
    let joined = qkvTable
    if (!joined) {
      const normalized = this.rms(input, block.norm)
      joined = rotateQKV
        ? this.pointwise(
            normalized, block.qkv, this.tensor(input.points, block.qkv.output), "pointwiseRoPE",
            { extraInput: rope, ropeHeads: block.heads },
          )
        : this.conv(normalized, block.qkv)
    }
    const attended = this.tensor(input.points, input.channels)
    if (this.useOnlineAttention) {
      if (!this.engine.half && !this.engine.model.graphAttention) {
        this.operation(
          "yRotateQK", [rope, joined],
          [input.points, block.heads], [Math.ceil(input.points * block.heads / 4)],
        )
      }
      const bindings = this.engine.model.graphAttention ? [joined, rope, attended] : [joined, attended]
      const params = [input.points, block.heads]
      if (qkvTable) bindings.push(this.spatial)
      if (this.engine.model.graphAttention) {
        bindings.push(this.biasIndices)
        params.push(this.graphFeatures.groups, this.graphFeatures.pairStride)
      }
      this.operation(qkvTable ? "bentAttentionIndexed" : "yAttentionOnline", bindings,
        params, [Math.ceil(input.points / ONLINE_Y_QUERY_ROWS), block.heads])
    } else {
      const bindings = [joined, rope, attended]
      const parameters = [input.points, block.heads]
      if (this.engine.half && this.engine.game === "y") {
        const columns = Math.ceil(input.points / 4)
        const keys = this.operation(
          "yAttentionKeys", [joined, rope, this.tensor(columns * 4, input.channels)],
          [input.points, block.heads, columns], [Math.ceil(columns * input.channels / 64)],
        )
        bindings.push(keys)
        parameters.push(columns)
      }
      this.operation("fusedAttention", bindings, parameters,
        [Math.ceil(input.points / FUSED_ATTENTION_QUERY_ROWS), block.heads])
    }
    return this.conv(attended, block.output, input)
  }

  ffn(input, block) {
    const normalized = this.rms(input, block.norm)
    const activated = this.pointwise(normalized, block.up, this.tensor(input.points, block.hidden), "pointwiseSwiGLU")
    return this.conv(activated, block.output, input)
  }

  block(input, block, bentPrefix = null) {
    if (block.kind === "attention") return this.attention(input, block)
    if (block.kind === "ffn") return this.ffn(input, block)
    let mid
    if (bentPrefix) {
      mid = bentPrefix.inner
    } else if (block.pool) {
      const pre = this.norm(input, block.pre.norm)
      const poolConv = this.conv(pre, block.pool.conv)
      const poolAct = this.norm(poolConv, block.pool.norm)
      const pooled = this.pool(poolAct, false)
      const bias = this.conv(pooled, block.pool.matrix)
      // The local tensor can reuse scratch from the pooled path.
      const local = this.conv(pre, block.pre.conv)
      mid = this.add(local, bias)
    } else { mid = this.normConv(input, block.pre) }
    if (block.blocks) {
      let qkvTable = bentPrefix?.qkv
      for (const child of block.blocks) {
        mid = qkvTable ? this.attention(mid, child, qkvTable) : this.block(mid, child)
        qkvTable = null
      }
    }
    const post = this.norm(mid, block.post.norm, mid === input)
    return this.conv(post, block.post.conv, input)
  }

  async evaluate(states, valueOnly, cancelled = null) {
    const model = this.engine.model
    const spatial = model.graphAttention
      ? packBentYRows(states, this.size, this.inputMaps, this.playablePoints)
      : new Uint32Array(this.spatial.count * states.length)
    const queue = this.device.queue
    if (!model.graphAttention) {
      const count = model.globalChannels * states.length
      // Queue writes require a multiple of four bytes.
      const global = this.engine.half ? new Uint16Array(count + count % 2) : new Float32Array(count)
      const one = this.engine.half ? 0x3c00 : 1
      states.forEach(({ board, toPlay, symmetry = 0 }, batch) => {
        const map = this.inputMaps[toPlay - 1][symmetry]
        const start = batch * this.spatial.count
        for (let point = 0; point < board.length; point++) {
          const stone = board[point]
          if (stone) spatial[start + map[point]] = stone === 3 ? 3 : stone === toPlay ? 1 : 2
        }
        global[batch * model.globalChannels] = Number(toPlay === 2) * one
      })
      queue.writeBuffer(this.global.buffer, 0, global)
    }
    queue.writeBuffer(this.spatial.buffer, 0, spatial)
    const ops = valueOnly ? this.valueOps : this.ops
    // Keep the full batch width but let a replacement stop the remaining layers.
    const limit = cancelled ? Math.ceil(ops.length / 3) : ops.length
    let encoder
    for (let first = 0; first < ops.length; first += limit) {
      encoder = this.device.createCommandEncoder()
      const pass = encoder.beginComputePass()
      const end = Math.min(first + limit, ops.length)
      for (let i = first; i < end; i++) {
        const op = ops[i]
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
      if (end === ops.length) break
      queue.submit([encoder.finish()])
      const poll = setInterval(() => queue.submit([]), this.pollMs)
      try {
        await queue.onSubmittedWorkDone()
      } finally {
        clearInterval(poll)
      }
      if (cancelled()) return null
    }
    const stride = valueOnly ? 3 : this.outputStride
    const byteLength = states.length * stride * 4
    encoder.copyBufferToBuffer(this.output.buffer, 0, this.readback, 0, byteLength)
    this.device.queue.submit([encoder.finish()])
    const mapping = this.readback.mapAsync(GPUMapMode.READ, 0, byteLength)
    // Submissions advance completion on implementations that poll pending mappings.
    const timer = setInterval(() => this.device.queue.submit([]), this.pollMs)
    try {
      await mapping
    } finally {
      clearInterval(timer)
    }
    let values
    try {
      values = new Float32Array(this.readback.getMappedRange(0, byteLength).slice(0))
    } finally {
      this.readback.unmap()
    }
    if (values.some((value) => !Number.isFinite(value))) {
      throw new Error("The GPU returned a non-finite evaluation. Analysis stopped.")
    }
    if (valueOnly) return states.map((_, i) => values.subarray(3 * i, 3 * (i + 1)))
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

async function captureGpuAllocation(device, allocate) {
  // Failed buffers can also invalidate their upload or bind groups.
  device.pushErrorScope("validation")
  device.pushErrorScope("out-of-memory")
  let thrown
  try {
    await allocate()
  } catch (error) {
    thrown = error
  }
  const [allocationError, validationError] = await Promise.all([device.popErrorScope(), device.popErrorScope()])
  if (allocationError || thrown || validationError) throw allocationError ?? thrown ?? validationError
}

export class Fp16CnnMemoryError extends Error {}

export class Network {
  static async create(adapter, archive, onError, boardGame, signal, forceFloat32 = false) {
    const model = archive.model
    const game = model.transformer ? boardGame : "hex"
    const half = !forceFloat32 && adapter.features.has("shader-f16")
    const device = await adapter.requestDevice({ requiredFeatures: half ? ["shader-f16"] : [] })
    const engine = new Network(device, model, half, game)
    try {
      signal.throwIfAborted()
      device.lost.then(({ reason, message }) => {
        if (reason !== "destroyed") onError(new Error(`GPU connection lost: ${message}. Press Start to retry.`))
      })
      device.addEventListener("uncapturederror", (event) => onError(event.error))
      const layouts = new Map()
      const layoutFor = (buffers) => {
        const key = buffers.join(" ")
        const existing = layouts.get(key)
        if (existing) return existing
        const bind = device.createBindGroupLayout({
          entries: [...buffers, "uniform"].map((type, binding) => ({
            binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
          })),
        })
        const layout = { bind, pipeline: device.createPipelineLayout({ bindGroupLayouts: [bind] }) }
        layouts.set(key, layout)
        return layout
      }
      const halfY = half && game === "y"
      const kernels = {
        ...createKernels(half, model.transformer, model.graphAttention),
        ...createPointwiseKernels(half, engine.pointwiseTiles, {
          transformer: model.transformer, rotateQKV: halfY, fuseAffine: engine.largeCNN,
        }),
        ...(!model.transformer ? createWinogradKernels(engine.largeCNN, half) : {}),
        ...(!half && !model.transformer ? {
          ...createCompactWinogradKernels(engine.compactTileRows, engine.pairCompactPlanes),
          restoreWinograd32: createWinogradFilter(false),
        } : {}),
        ...(model.transformer ? createTransformerKernels(half) : {}),
        ...(model.graphAttention ? {
          ...bentPrefixKernels(half),
          bentAttentionIndexed: createOnlineYAttention(half, { graphBias: true, indexed: true }).yAttentionOnline,
        } : {}),
        ...(game === "y" || model.graphAttention ? createOnlineYAttention(half, { graphBias: model.graphAttention }) : {}),
        ...(halfY ? { yAttentionKeys: createYAttentionKeys() } : {}),
        ...(game === "y" || game === "yavalath" ? {
          fusedAttention: createFusedAttention(half, { yavalath: game === "yavalath" }),
        } : {}),
      }
      const pipelinesReady = Promise.all(
        Object.entries(kernels).map(async ([name, { code, buffers }]) => {
          const target = layoutFor(buffers)
          const pipeline = await device.createComputePipelineAsync({
            layout: target.pipeline,
            compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
          })
          return [name, { pipeline, layout: target.bind, output: buffers.indexOf(WRITE) }]
        }),
      ).catch((error) => {
        archive.cancel()
        throw error
      })
      const backgroundUpload = half && model.transformer ? (async () => {
        const halfUpload = await prepareHalfUpload(device)
        signal.throwIfAborted()
        await uploadModel(engine, archive, signal, halfUpload)
        await archive.finish()
      })() : null
      const [compiled, loaded] = await Promise.allSettled([pipelinesReady, backgroundUpload])
      if (compiled.status === "rejected") throw compiled.reason
      if (loaded.status === "rejected") throw loaded.reason
      signal.throwIfAborted()
      engine.pipelines = Object.fromEntries(compiled.value)
      if (!backgroundUpload) {
        if (model.transformer) await uploadModel(engine, archive, signal)
        else await captureGpuAllocation(device, () => uploadModel(engine, archive, signal))
        await archive.finish()
      }
      signal.throwIfAborted()
      if (model.graphAttention) engine.bentPrefix = createBentPrefix(engine)
      return engine
    } catch (error) {
      archive.cancel()
      engine.destroy()
      throw engine.memoryError(error)
    }
  }

  constructor(device, model, half, game) {
    this.half = half
    this.winogradFilterCount = model.transformer ? 0 : countWinogradFilters(model)
    const preparedBytes = this.winogradFilterCount * WINOGRAD_COEFFICIENTS * Uint16Array.BYTES_PER_ELEMENT
    this.largeCNN = preparedBytes >= LARGE_CNN_WEIGHT_BYTES
    this.game = game
    this.pointwiseTiles = pointwiseTiles(half, model.transformer)
    this.runtime = half ? "f16" : "f32"
    this.runtimeHelp = model.transformer
      ? `${half ? 16 : 32}-bit floating-point tensors and projections; 32-bit attention and normalization reductions.`
      : half
        ? "16-bit floating-point tensors and Winograd convolutions."
        : "32-bit floating-point tensors and Winograd convolutions with packed weights."
    this.pairCompactPlanes = !half && this.largeCNN
    this.compactTileRows = this.largeCNN ? 64 : 32
    this.bytesPerValue = half ? 2 : 4
    this.device = device
    this.model = model
    this.weights = []
    this.graph = null
  }

  memoryError(error) {
    if (this.half && !this.model.transformer && error instanceof GPUOutOfMemoryError) {
      return new Fp16CnnMemoryError(error.message)
    }
    return error
  }

  async prepare(size) {
    if (this.graph?.size === size && !this.graph.largeBatch) return this.graph
    this.releaseGraph()
    let width = batchSizeFor(size, this)
    // A half-width FP16 graph can retain the loaded weights and useful parallel work.
    const minimumWidth = this.half ? Math.max(1, Math.floor(width / 2)) : 1
    let floatWinograd = useFloatWinograd(size, this)
    for (;;) {
      try {
        this.graph = await this.allocateGraph(size, false, width, floatWinograd)
        return this.graph
      } catch (error) {
        if (this.model.transformer || !(error instanceof GPUOutOfMemoryError)) throw error
        if (width > minimumWidth) {
          width = Math.floor(width / 2)
        } else if (floatWinograd) {
          // At width one the larger tiles can still need a full layer of restored filters.
          floatWinograd = false
        } else throw this.memoryError(error)
      }
    }
  }

  async prepareWideBatch(size) {
    if (this.graph?.size === size && this.graph.batchSize < batchSizeFor(size, this)) return this.graph
    this.releaseGraph()
    try {
      this.graph = await this.allocateGraph(size, true)
      return this.graph
    } catch (error) {
      if (!(error instanceof GPUOutOfMemoryError)) throw error
    }
    return this.prepare(size)
  }

  async allocateGraph(size, wide, width = batchSizeFor(size, this), floatWinograd = useFloatWinograd(size, this)) {
    let graph
    try {
      await captureGpuAllocation(this.device, () => { graph = new Graph(this, size, wide, width, floatWinograd) })
      return graph
    } catch (error) {
      graph?.destroy()
      throw error
    }
  }

  releaseGraph() {
    this.graph?.destroy()
    this.graph = null
  }

  evaluate(states, cancelled) {
    return this.evaluateBatches(states, cancelled, false)
  }

  evaluateValues(states, cancelled) {
    return this.evaluateBatches(states, cancelled, true)
  }

  async evaluateBatches(states, cancelled, valueOnly) {
    const size = states[0].size
    const graph = this.graph?.size === size && !this.graph.largeBatch ? this.graph : await this.prepare(size)
    const results = []
    const submissionSize = graph.batchSize
    for (let offset = 0; offset < states.length; offset += submissionSize) {
      if (cancelled()) return null
      results.push(...(await graph.evaluate(states.slice(offset, offset + submissionSize), valueOnly)))
      // Let worker messages run before submitting another GPU batch.
      if (offset + submissionSize < states.length) await new Promise((resolve) => setTimeout(resolve, 0))
    }
    return results
  }

  destroy() {
    this.releaseGraph()
    for (const buffer of this.weights) buffer.destroy()
    this.device.destroy()
  }
}
