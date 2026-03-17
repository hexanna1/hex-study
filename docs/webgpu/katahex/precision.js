// SPDX-License-Identifier: MIT
// Round float32 values to IEEE 754 binary16, ties to even.
export function toHalf(values) {
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length)
  const result = new Uint16Array(values.length)
  for (let i = 0; i < bits.length; i++) {
    const word = bits[i]
    const sign = (word >>> 16) & 0x8000
    const exponent = (word >>> 23) & 255
    const fraction = word & 0x7fffff
    if (exponent === 255) {
      result[i] = sign | 0x7c00 | (fraction ? 0x200 : 0)
      continue
    }
    if (exponent < 102) { result[i] = sign; continue }
    if (exponent > 142) { result[i] = sign | 0x7c00; continue }
    const shift = exponent < 113 ? 126 - exponent : 13
    const mantissa = exponent < 113 ? fraction | 0x800000 : fraction
    const rounded = (mantissa + (2 ** (shift - 1) - 1) + ((mantissa >>> shift) & 1)) >>> shift
    result[i] = sign | ((exponent < 113 ? 0 : (exponent - 112) << 10) + rounded)
  }
  return result
}
