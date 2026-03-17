export function toHalf(values, result) {
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length)
  for (let i = 0; i < bits.length; i++) {
    const word = bits[i]
    const sign = (word >>> 16) & 0x8000
    const magnitude = word & 0x7fffffff
    if (magnitude >= 0x38800000 && magnitude < 0x47800000) {
      const rounded = magnitude + 0xfff + ((magnitude >>> 13) & 1)
      result[i] = sign | ((rounded >>> 13) - 0x1c000)
    } else if (magnitude < 0x33000000) {
      result[i] = sign
    } else if (magnitude < 0x38800000) {
      const shift = 126 - (magnitude >>> 23)
      const mantissa = (magnitude & 0x7fffff) | 0x800000
      result[i] = sign | ((mantissa + (2 ** (shift - 1) - 1) + ((mantissa >>> shift) & 1)) >>> shift)
    } else {
      result[i] = sign | (magnitude > 0x7f800000 ? 0x7e00 : 0x7c00)
    }
  }
  return result
}
