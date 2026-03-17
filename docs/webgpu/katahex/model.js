import { tensorScratch } from "./tensor_scratch.js"

function payloadBounds(model) {
  let firstCode = Infinity, end = 0
  function visit(value) {
    if (!value || typeof value !== "object") return
    if (value.type === "f32") {
      end = Math.max(end, value.offset + value.length * 4)
    } else if (value.type === "q7") {
      firstCode = Math.min(firstCode, value.offset)
      end = Math.max(end, value.offset + value.length, value.scaleOffset + value.channels * 4)
    } else {
      for (const child of Object.values(value)) visit(child)
    }
  }
  visit(model)
  return { prefixBytes: Number.isFinite(firstCode) ? firstCode : 0, end }
}

export async function readModel(stream, signal) {
  const reader = stream.pipeThrough(new DecompressionStream("gzip")).getReader()
  const cancel = () => { void reader.cancel().catch(() => {}).finally(() => reader.releaseLock()) }
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

  let model, prefix, payloadEnd
  try {
    signal.throwIfAborted()
    const headerSize = new Uint8Array(4)
    await copy(headerSize)
    const header = new Uint8Array(new DataView(headerSize.buffer).getUint32(0, true))
    await copy(header)
    model = JSON.parse(new TextDecoder().decode(header))
    const { prefixBytes, end } = payloadBounds(model)
    prefix = new Uint8Array(prefixBytes)
    payloadEnd = end
  } catch (error) {
    signal.removeEventListener("abort", cancel)
    cancel()
    throw error
  }

  // The exporter can group floats and scales before the codes. Keep that
  // prefix for random access, then consume tensors in upload order.
  let cursor = 0
  async function advance(end) {
    if (end <= cursor) return
    const prefixEnd = Math.min(end, prefix.length)
    if (cursor < prefixEnd) {
      await copy(prefix, cursor, prefixEnd)
      cursor = prefixEnd
    }
    if (cursor < end) {
      await copy(null, cursor, end)
      cursor = end
    }
  }

  async function readTensor(start, target) {
    signal.throwIfAborted()
    const bytes = new Uint8Array(target.buffer, target.byteOffset, target.byteLength)
    if (start < prefix.length) {
      const end = start + bytes.length
      if (end > cursor) await advance(end)
      bytes.set(prefix.subarray(start, end))
      return
    }
    if (start > cursor) await advance(start)
    await copy(bytes)
    cursor += bytes.length
  }

  async function finish() {
    try {
      await advance(payloadEnd)
      while (!(await reader.read()).done) {} // Check the gzip trailer.
      signal.throwIfAborted()
    } finally {
      signal.removeEventListener("abort", cancel)
      reader.releaseLock()
    }
  }

  const scratch = tensorScratch()
  async function quantizedTensor(value) {
    const scales = scratch("scales", Float32Array, value.channels)
    const codes = scratch("codes", Int8Array, value.length)
    await readTensor(value.scaleOffset, scales)
    await readTensor(value.offset, codes)
    return { codes, scales }
  }

  async function decodeTensor(value, quantized) {
    const result = scratch("float", Float32Array, value.length)
    if (value.type === "f32") {
      await readTensor(value.offset, result)
    } else {
      const { codes, scales } = quantized ?? await quantizedTensor(value)
      for (let row = 0; row < result.length; row += scales.length) {
        for (let channel = 0; channel < scales.length; channel++) {
          result[row + channel] = codes[row + channel] * scales[channel]
        }
      }
    }
    return result
  }

  return { model, finish, decodeTensor, quantizedTensor }
}
