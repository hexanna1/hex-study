// SPDX-License-Identifier: MIT
// KataHex model descriptors, adapted from cpp/neuralnet/desc.cpp.
// See LICENSE and THIRD_PARTY_NOTICES.md in this directory.

class Reader {
  constructor(bytes) {
    this.bytes = bytes
    this.offset = 0
    this.decoder = new TextDecoder()
  }

  token() {
    while (this.offset < this.bytes.length && this.bytes[this.offset] <= 32) this.offset++
    const start = this.offset
    while (this.offset < this.bytes.length && this.bytes[this.offset] > 32) this.offset++
    if (start === this.offset) throw new Error("Could not read the model export.")
    return this.decoder.decode(this.bytes.subarray(start, this.offset))
  }

  number() { return Number(this.token()) }

  floats(count) {
    while (this.offset < this.bytes.length && this.bytes[this.offset] <= 32) this.offset++
    this.offset += 5 // @BIN@
    const start = this.offset
    this.offset += count * 4
    return new Float32Array(this.bytes.slice(start, this.offset).buffer)
  }

  conv() {
    this.token() // Layer name.
    const ky = this.number(), kx = this.number()
    const input = this.number(), output = this.number()
    const dy = this.number(), dx = this.number()
    // The file already stores the kernel as [y, x, input, output].
    return { ky, kx, input, output, dy, dx, weights: this.floats(ky * kx * input * output) }
  }

  matrix() {
    this.token() // Layer name.
    const input = this.number(), output = this.number()
    return { input, output, ky: 1, kx: 1, dy: 1, dx: 1, weights: this.floats(input * output) }
  }

  bias() {
    this.token()
    return this.floats(this.number())
  }

  activation() {
    this.token()
    return { ACTIVATION_IDENTITY: 0, ACTIVATION_RELU: 1, ACTIVATION_MISH: 2, ACTIVATION_SILU: 3 }[this.token()]
  }

  norm() {
    this.token() // Layer name.
    const channels = this.number(), epsilon = this.number()
    const hasScale = this.number(), hasBias = this.number()
    const mean = this.floats(channels), variance = this.floats(channels)
    const scale = hasScale ? this.floats(channels) : new Float32Array(channels).fill(1)
    const bias = hasBias ? this.floats(channels) : new Float32Array(channels)
    const weights = new Float32Array(channels * 2)
    for (let c = 0; c < channels; c++) {
      weights[c] = scale[c] / Math.sqrt(variance[c] + epsilon)
      weights[channels + c] = bias[c] - mean[c] * weights[c]
    }
    return { channels, weights, activation: this.activation() }
  }

  normConv() { return { norm: this.norm(), conv: this.conv() } }

  rms() {
    this.token()
    const channels = this.number(), epsilon = this.number()
    return { kind: "rms", channels, epsilon, weights: this.floats(channels) }
  }

  spatialNorm() {
    this.token()
    const channels = this.number(), epsilon = this.number()
    this.number()
    this.number()
    const weights = new Float32Array(channels * 2)
    weights.set(this.floats(channels))
    weights.set(this.floats(channels), channels)
    return { kind: "spatialRms", channels, epsilon, weights, activation: this.activation() }
  }

  joined(matrices) {
    const input = matrices[0].input
    const output = matrices.reduce((sum, layer) => sum + layer.output, 0)
    const weights = new Float32Array(input * output)
    for (let row = 0; row < input; row++) {
      let offset = row * output
      for (const layer of matrices) {
        weights.set(layer.weights.subarray(row * layer.output, (row + 1) * layer.output), offset)
        offset += layer.output
      }
    }
    return { input, output, ky: 1, kx: 1, dy: 1, dx: 1, weights }
  }

  block() {
    const kind = this.token()
    this.token() // Block name.
    if (kind === "transformer_attention_block" || kind === "transformer_graph_attention_block") {
      const heads = this.number()
      this.number() // Key/value heads.
      this.number()
      this.number() // Query/key and value head dimensions.
      this.number() // Rotary embeddings enabled.
      this.number() // Learned rotary frequencies.
      const norm = this.rms()
      const qkv = this.joined([this.matrix(), this.matrix(), this.matrix()])
      const output = this.matrix()
      if (kind === "transformer_graph_attention_block") {
        return { kind: "attention", heads, norm, qkv, output, biasInput: this.matrix(), biasOutput: this.matrix() }
      }
      this.token()
      const ropeHeads = this.number(), pairs = this.number()
      this.number() // Two spatial coordinates.
      return { kind: "attention", heads, norm, qkv, output, ropeFreqs: this.floats(ropeHeads * pairs * 2) }
    }
    if (kind === "transformer_ffn_block") {
      this.number() // Input channels.
      const hidden = this.number()
      this.number() // SwiGLU enabled.
      const norm = this.rms(), up = this.joined([this.matrix(), this.matrix()])
      return { kind: "ffn", hidden, norm, up, output: this.matrix() }
    }
    if (kind === "nested_bottleneck_block") {
      const count = this.number()
      const pre = this.normConv()
      const blocks = Array.from({ length: count }, () => this.block())
      return { pre, blocks, post: this.normConv() }
    }
    const pre = this.normConv()
    let pool = null
    if (kind === "gpool_block") { pool = { conv: this.conv(), norm: this.norm(), matrix: this.matrix() } }
    return { pre, pool, post: this.normConv() }
  }
}

export async function readModel(file, transformer = false) {
  const bytes = new Uint8Array(
    await new Response(file.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer(),
  )
  const r = new Reader(bytes)
  r.token() // Export name.
  r.token() // Native model-format field.
  const spatialChannels = r.number(), globalChannels = r.number()
  if (transformer) {
    for (let i = 0; i < 7; i++) r.number() // Output scaling.
    for (let i = 0; i < 8; i++) r.number() // Descriptor options.
  }
  r.token() // Trunk name.
  const count = r.number()
  for (let i = 0; i < 5; i++) r.number() // Trunk channel counts; each layer carries its own shape.
  if (transformer) for (let i = 0; i < 6; i++) r.number() // Trunk normalization and options.
  const initial = r.conv(), global = r.matrix()
  const blocks = Array.from({ length: count }, () => r.block())
  const tip = transformer ? r.spatialNorm() : r.norm()
  r.token() // Policy head name.
  if (transformer) for (let i = 0; i < 4; i++) r.number() // Policy channels and options.
  const policy = {
    local: r.conv(),
    pool: r.conv(),
    poolNorm: r.norm(),
    poolBias: r.matrix(),
    norm: r.norm(),
    output: r.conv(),
    pass: r.matrix(),
  }
  if (transformer) {
    const projection = policy.output
    // Board analysis uses the regular policy channel.
    projection.weights = Float32Array.from(
      { length: projection.weights.length / projection.output },
      (_, i) => projection.weights[i * projection.output],
    )
    projection.output = 1
    r.bias()
    r.activation()
    r.matrix() // Pass head.
  }
  r.token() // Value head name.
  if (transformer) for (let i = 0; i < 3; i++) r.number() // Value-head options.
  const value = { conv: r.conv(), norm: r.norm(), hidden: r.matrix(), bias: r.bias(), activation: r.activation() }
  value.output = r.matrix()
  value.outputBias = r.bias()
  const affine = (bias, activation) => {
    const weights = new Float32Array(bias.length * 2)
    weights.fill(1, 0, bias.length)
    weights.set(bias, bias.length)
    return { channels: bias.length, activation, weights }
  }
  value.hiddenNorm = affine(value.bias, value.activation)
  value.outputNorm = affine(value.outputBias, 0)
  delete value.bias
  delete value.outputBias
  delete value.activation
  delete policy.pass
  // Score and ownership heads are not needed for board analysis.
  const graphAttention = blocks.some((block) => block.blocks?.some((child) => child.biasInput))
  return { transformer, graphAttention, spatialChannels, globalChannels, initial, global, blocks, tip, policy, value }
}
