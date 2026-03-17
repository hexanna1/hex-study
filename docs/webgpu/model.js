import { tensorScratch } from "./tensor_scratch.js"

// A five-bit affine label gives the low bits of 16 weights in one channel.
const affineMasks = Uint16Array.from({ length: 32 }, (_, label) => {
  let mask = label & 1 ? 0xffff : 0
  for (const pattern of [0xaaaa, 0xcccc, 0xf0f0, 0xff00]) {
    label >>= 1
    if (label & 1) mask ^= pattern
  }
  return mask
})

// Interleave zero bits so two masks can advance one row at a time together.
const interleavedMasks = Uint32Array.from(affineMasks, (mask) => {
  mask = (mask | (mask << 8)) & 0x00ff00ff
  mask = (mask | (mask << 4)) & 0x0f0f0f0f
  mask = (mask | (mask << 2)) & 0x33333333
  return (mask | (mask << 1)) & 0x55555555
})
const byteLowBits = Uint32Array.from({ length: 16 }, (_, bits) =>
  (bits & 1) | ((bits & 2) << 7) | ((bits & 4) << 14) | ((bits & 8) << 21))

function expandAffine(codes, channels, flags) {
  const words = new DataView(codes.buffer, codes.byteOffset, codes.byteLength)
  // The channel bitmap precedes the packed five-bit labels.
  let bit = Math.ceil(channels / 8) * 8
  for (let base = 0; base < codes.length; base += 16 * channels) {
    for (let channel = 0; channel < channels;) {
      const selected = flags[channel >> 3] >>> (channel & 7)
      const byte = bit >> 3
      const shift = bit & 7
      if (!(channel & 3) && channel + 4 <= channels && (selected & 15) === 15) {
        const labels = (flags[byte] | (flags[byte + 1] << 8) |
          (flags[byte + 2] << 16) | (flags[byte + 3] << 24)) >>> shift
        let first = interleavedMasks[labels & 31] | (interleavedMasks[(labels >>> 5) & 31] << 1)
        let second = interleavedMasks[(labels >>> 10) & 31] | (interleavedMasks[(labels >>> 15) & 31] << 1)
        for (let row = 0, at = base + channel; row < 16; row++, at += channels) {
          // Clear each byte's high bit before doubling so carries stay in that byte.
          const doubled = (words.getUint32(at, true) & 0x7f7f7f7f) << 1
          words.setUint32(at, doubled | byteLowBits[(first & 3) | ((second & 3) << 2)], true)
          first >>>= 2
          second >>>= 2
        }
        bit += 20
        channel += 4
        continue
      }
      if (selected & 1) {
        const label = ((flags[byte] | (flags[byte + 1] << 8)) >>> shift) & 31
        let mask = affineMasks[label]
        for (let row = 0, at = base + channel; row < 16; row++, at += channels) {
          codes[at] = 2 * codes[at] + (mask & 1)
          mask >>>= 1
        }
        bit += 5
      }
      channel++
    }
  }
}

function payloadLengths(model) {
  let floatBytes = 0, codeBytes = 0
  function visit(value) {
    if (!value || typeof value !== "object") return
    if (value.type === "f32") {
      floatBytes = Math.max(floatBytes, value.offset + value.length * 4)
    } else if (value.type === "q7") {
      floatBytes = Math.max(floatBytes, value.scaleOffset + value.channels * 4)
      codeBytes += value.length
    } else {
      for (const child of Object.values(value)) visit(child)
    }
  }
  visit(model)
  return { floatBytes, codeBytes }
}

export async function readModel(stream, signal) {
  const reader = stream.pipeThrough(new DecompressionStream("gzip")).getReader()
  let open = true
  const detach = () => {
    if (!open) return false
    open = false
    signal.removeEventListener("abort", cancel)
    return true
  }
  const cancel = () => {
    if (detach()) void reader.cancel().catch(() => {}).finally(() => reader.releaseLock())
  }
  let chunk
  let offset = 0
  signal.addEventListener("abort", cancel, { once: true })
  async function copy(target, start = 0, end = target.length) {
    while (start < end) {
      if (!chunk || offset === chunk.length) {
        const next = await reader.read()
        if (next.done) {
          signal.throwIfAborted()
          throw new Error("The analysis model is incomplete.")
        }
        chunk = next.value
        offset = 0
      }
      const count = Math.min(end - start, chunk.length - offset)
      if (target) target.set(chunk.subarray(offset, offset + count), start)
      offset += count
      start += count
    }
  }

  let model, prefix, pendingCodes, affineGrids
  try {
    signal.throwIfAborted()
    const headerSize = new Uint8Array(4)
    await copy(headerSize)
    const bytes = new Uint8Array(new DataView(headerSize.buffer).getUint32(0, true))
    await copy(bytes)
    const [network, grids = "{}"] = new TextDecoder().decode(bytes).split("\0")
    model = JSON.parse(network)
    affineGrids = JSON.parse(grids)
    const { floatBytes, codeBytes } = payloadLengths(model)
    prefix = new Uint8Array(floatBytes)
    pendingCodes = codeBytes + Object.values(affineGrids).reduce((total, bytes) => total + bytes, 0)
  } catch (error) {
    cancel()
    throw error
  }

  // Float tensors and scales precede the codes, which follow descriptor order.
  let loaded = 0
  async function loadPrefix(end) {
    if (loaded < end) {
      await copy(prefix, loaded, end)
      loaded = end
    }
  }

  async function finish() {
    try {
      signal.throwIfAborted()
      await copy(null, 0, prefix.length - loaded + pendingCodes)
      while (!(await reader.read()).done) {} // Check the gzip trailer.
      signal.throwIfAborted()
    } catch (error) {
      cancel()
      throw error
    } finally {
      if (detach()) reader.releaseLock()
    }
  }

  const scratch = tensorScratch()
  // Use the code view before reading the next tensor.
  async function quantizedTensor(value) {
    signal.throwIfAborted()
    await loadPrefix(prefix.length)
    const scales = new Float32Array(prefix.buffer, value.scaleOffset, value.channels)
    const gridBytes = affineGrids[value.offset] ?? 0
    const packed = scratch("codes", Uint8Array, gridBytes + value.length)
    await copy(packed)
    const codes = new Int8Array(packed.buffer, packed.byteOffset + gridBytes, value.length)
    if (gridBytes) expandAffine(codes, value.channels, packed)
    pendingCodes -= packed.byteLength
    return { codes, scales }
  }

  function dequantize({ codes, scales }, result) {
    for (let row = 0; row < result.length; row += scales.length) {
      for (let channel = 0; channel < scales.length; channel++) {
        // FP16 upload must narrow the decoded FP32 value.
        result[row + channel] = Math.fround(codes[row + channel] * scales[channel])
      }
    }
    return result
  }

  async function decodeTensor(value, quantized) {
    signal.throwIfAborted()
    if (value.type === "f32") {
      await loadPrefix(value.offset + value.length * 4)
      return new Float32Array(prefix.buffer, value.offset, value.length)
    }
    return dequantize(quantized ?? await quantizedTensor(value), scratch("float", Float32Array, value.length))
  }

  return { model, finish, cancel, decodeTensor, quantizedTensor, dequantize }
}
