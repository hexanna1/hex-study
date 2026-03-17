// Convert a native KataHex export into the browser's compact weight archive.
// Usage: node docs/webgpu/quantize_model.mjs input.bin.gz output.gz

import { readFileSync, writeFileSync } from "node:fs"
import { gzipSync } from "node:zlib"
import { readModel } from "./katahex/native_model.js"

const model = await readModel(new Blob([readFileSync(process.argv[2])]))
const parts = []
let length = 0

function append(array) {
  const offset = length
  const bytes = Buffer.from(array.buffer, array.byteOffset, array.byteLength)
  parts.push(bytes)
  length += bytes.length
  return offset
}

function pack(value, parent = null, key = null) {
  if (value instanceof Float32Array) {
    if (key === "weights" && Number.isInteger(parent?.output)) {
      const channels = parent.output
      const scales = new Float32Array(channels)
      for (let i = 0; i < value.length; i++) {
        const channel = i % channels
        scales[channel] = Math.max(scales[channel], Math.abs(value[i]))
      }
      for (let i = 0; i < scales.length; i++) scales[i] = scales[i] / 127 || 1
      const weights = new Int8Array(value.length)
      for (let i = 0; i < value.length; i++) {
        const scaled = Math.round(value[i] / scales[i % channels])
        weights[i] = Math.max(-127, Math.min(127, scaled))
      }
      const scaleOffset = append(scales)
      const offset = append(weights)
      return { type: "q8", scaleOffset, offset, length: value.length, channels }
    }
    return { type: "f32", offset: append(value), length: value.length }
  }
  if (Array.isArray(value)) return value.map((child) => pack(child))
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([childKey, child]) => [childKey, pack(child, value, childKey)]))
  }
  return value
}

const header = Buffer.from(JSON.stringify(pack(model)))
const headerSize = Buffer.alloc(4)
headerSize.writeUInt32LE(header.length)
const archive = gzipSync(Buffer.concat([headerSize, header, ...parts]), { level: 9 })
writeFileSync(process.argv[3], archive)
console.log(`${(archive.length / 1048576).toFixed(2)} MiB`)
