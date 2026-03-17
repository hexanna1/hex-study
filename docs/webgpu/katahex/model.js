// SPDX-License-Identifier: MIT
import { tensorScratch } from "./tensor_scratch.js"

export async function readModel(file, packedInt8 = false) {
  const bytes = new Uint8Array(await new Response(file.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer())
  const headerLength = new DataView(bytes.buffer).getUint32(0, true)
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLength)))
  const payload = 4 + headerLength
  const scratch = tensorScratch()

  function decodeTensor(value, parent) {
    if (value.type === "f32") {
      const result = scratch("float", Float32Array, value.length)
      new Uint8Array(result.buffer, result.byteOffset, result.byteLength).set(
        bytes.subarray(payload + value.offset, payload + value.offset + value.length * 4))
      return result
    }
    if (value.type === "q8") {
      const scales = scratch("scales", Float32Array, value.channels)
      new Uint8Array(scales.buffer, scales.byteOffset, scales.byteLength).set(
        bytes.subarray(payload + value.scaleOffset, payload + value.scaleOffset + value.channels * 4))
      const weights = new Int8Array(bytes.buffer, payload + value.offset, value.length)
      if (packedInt8 && parent?.input % 4 === 0) {
        const packed = scratch("packed", Uint32Array, value.length / 4)
        for (let group = 0; group < packed.length / value.channels; group++) {
          for (let channel = 0; channel < value.channels; channel++) {
            let word = 0
            for (let lane = 0; lane < 4; lane++) {
              word |= (weights[(group * 4 + lane) * value.channels + channel] & 255) << (lane * 8)
            }
            packed[group * value.channels + channel] = word
          }
        }
        return { packed, scales }
      }
      const result = scratch("float", Float32Array, value.length)
      for (let i = 0; i < result.length; i++) result[i] = weights[i] * scales[i % value.channels]
      return result
    }
  }

  return { model: header, decodeTensor }
}
