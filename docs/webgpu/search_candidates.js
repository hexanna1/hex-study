import { toHalf } from "./precision.js"

const Float16 = globalThis.Float16Array
export const COLD_HEADER_BYTES = Uint32Array.BYTES_PER_ELEMENT
// The header records storage and the candidate count.
const HALF = 1
const PACKED_HALF = 2
const HEADER_SHIFT = 2
const MIN_PACKED_LOGITS = 384

function moveArrayType(points) {
  // Move IDs index the full square canvas in every game.
  return points <= 256 ? Uint8Array : Uint16Array
}

export function minimumColdLogitBytes(half, points) {
  if (!half) return Float32Array.BYTES_PER_ELEMENT
  return points >= MIN_PACKED_LOGITS ? 1.5 : Uint16Array.BYTES_PER_ELEMENT
}

export function expandedCandidateBytes(count, points) {
  return count * (Float32Array.BYTES_PER_ELEMENT + moveArrayType(points).BYTES_PER_ELEMENT)
}

export function candidateBytes(children) {
  return children instanceof ArrayBuffer ? children.byteLength : children.moves.buffer.byteLength + (children.cold?.byteLength ?? 0)
}

export function coldCandidateCount(buffer) {
  return new Uint32Array(buffer, 0, 1)[0] >>> HEADER_SHIFT
}

// Candidate data is immutable between root evaluations and shared by snapshots.
export function packCandidates(moves, priors, points, { mapping, order } = {}) {
  const MoveArray = moveArrayType(points)
  const count = order?.length ?? moves.length
  const moveOffset = count * Float32Array.BYTES_PER_ELEMENT
  const buffer = new ArrayBuffer(moveOffset + count * MoveArray.BYTES_PER_ELEMENT)
  const packedPriors = new Float32Array(buffer, 0, count)
  const packedMoves = new MoveArray(buffer, moveOffset, count)
  if (order) {
    for (let i = 0; i < order.length; i++) {
      const index = order[i]
      packedPriors[i] = priors[index]
      packedMoves[i] = moves[index]
    }
  } else {
    packedPriors.set(priors)
    if (mapping) {
      for (let i = 0; i < moves.length; i++) packedMoves[i] = mapping[moves[i]]
    } else packedMoves.set(moves)
  }
  return { moves: packedMoves, priors: packedPriors }
}

// Used only synchronously while packing or materializing one policy.
let scratch = new Float32Array(0)

function policyScratch(points) {
  if (scratch.length < points) scratch = new Float32Array(points)
  return scratch
}

function bitmapCount(bitmap) {
  let count = 0
  for (let word of bitmap) {
    word -= (word >>> 1) & 0x55555555
    word = (word & 0x33333333) + ((word >>> 2) & 0x33333333)
    count += Math.imul((word + (word >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24
  }
  return count
}

// A binary16 word has ten mantissa bits and six sign and exponent bits.
const HALF_PREFIXES = 64
const halfCounts = new Uint16Array(HALF_PREFIXES)
const halfRanks = new Uint8Array(HALF_PREFIXES)

function packHalf(source, header) {
  const count = source.length
  const offset = COLD_HEADER_BYTES
  halfCounts.fill(0)
  for (const bits of source) halfCounts[bits >> 10]++
  let first = -1, second = -1, third = -1
  for (let i = 0; i < HALF_PREFIXES; i++) {
    if (first < 0 || halfCounts[i] > halfCounts[first]) {
      third = second
      second = first
      first = i
    } else if (second < 0 || halfCounts[i] > halfCounts[second]) {
      third = second
      second = i
    } else if (third < 0 || halfCounts[i] > halfCounts[third]) {
      third = i
    }
  }
  // A 12-bit token stores ten mantissa bits and a prefix index. The fourth
  // index draws a prefix from the trailing array, in move order.
  const prefixBytes = 3
  const tokenBytes = Math.ceil(count * 1.5)
  const exceptions = count - halfCounts[first] - halfCounts[second] - halfCounts[third]
  const length = prefixBytes + tokenBytes + exceptions
  const rawSize = count * 2
  if (length + 8 > rawSize) return null
  const result = new ArrayBuffer(COLD_HEADER_BYTES + length)
  const bytes = new Uint8Array(result)
  new Uint32Array(result, 0, 1)[0] = header | PACKED_HALF
  bytes[offset] = first
  bytes[offset + 1] = second
  bytes[offset + 2] = third
  halfRanks.fill(3)
  halfRanks[first] = 0
  halfRanks[second] = 1
  halfRanks[third] = 2
  let tokenOffset = offset + prefixBytes
  let exceptionOffset = tokenOffset + tokenBytes
  const encode = bits => {
    const prefix = bits >> 10
    const tag = halfRanks[prefix]
    if (tag === 3) bytes[exceptionOffset++] = prefix
    return (tag << 10) | (bits & 1023)
  }
  for (let i = 0; i < count; i += 2) {
    const a = encode(source[i])
    const b = i + 1 < count ? encode(source[i + 1]) : 0
    bytes[tokenOffset++] = a
    bytes[tokenOffset++] = (a >> 8) | (b << 4)
    if (i + 1 < count) bytes[tokenOffset++] = b >> 4
  }
  return result
}

function readPackedHalf(buffer, count) {
  const offset = COLD_HEADER_BYTES
  const bytes = new Uint8Array(buffer)
  let tokenOffset = offset + 3
  let exceptionOffset = tokenOffset + Math.ceil(count * 1.5)
  const result = halfScratch(count)
  const decode = token => {
    const tag = token >> 10
    const prefix = tag === 3 ? bytes[exceptionOffset++] : bytes[offset + tag]
    return (prefix << 10) | (token & 1023)
  }
  for (let i = 0; i < count; i += 2) {
    const lo = bytes[tokenOffset++]
    const hi = bytes[tokenOffset++]
    result[i] = decode(lo | ((hi & 15) << 8))
    if (i + 1 < count) result[i + 1] = decode((hi >> 4) | (bytes[tokenOffset++] << 4))
  }
  const LogitArray = Float16 ?? Uint16Array
  return new LogitArray(result.buffer, 0, count)
}

let halfPolicy = new Uint16Array(0)
function halfScratch(count) {
  if (halfPolicy.length < count) halfPolicy = new Uint16Array(count)
  return halfPolicy
}

export function packColdCandidates({ moves, moveBitmap }, policy, half) {
  const count = moveBitmap ? bitmapCount(moveBitmap) : moves.length
  const header = (count << HEADER_SHIFT) | (half ? HALF : 0)
  const LogitArray = half ? Float16 ?? Uint16Array : Float32Array
  const compact = half && count >= MIN_PACKED_LOGITS
  const buffer = compact ? halfScratch(count).buffer : new ArrayBuffer(COLD_HEADER_BYTES + count * LogitArray.BYTES_PER_ELEMENT)
  const logits = new LogitArray(buffer, compact ? 0 : COLD_HEADER_BYTES, count)
  const values = half && !Float16 ? policyScratch(count).subarray(0, count) : logits
  if (moveBitmap) {
    let index = 0
    for (let i = 0; i < moveBitmap.length; i++) {
      for (let word = moveBitmap[i]; word; word &= word - 1) {
        values[index++] = policy[i * 32 + 31 - Math.clz32(word & -word)]
      }
    }
  } else {
    for (let i = 0; i < count; i++) values[i] = policy[moves[i]]
  }
  if (values !== logits) toHalf(values, logits)
  if (compact) {
    const packed = packHalf(new Uint16Array(buffer, 0, count), header)
    if (packed) return packed
    const output = new ArrayBuffer(COLD_HEADER_BYTES + logits.byteLength)
    new Uint32Array(output, 0, 1)[0] = header
    new Uint8Array(output, COLD_HEADER_BYTES).set(new Uint8Array(buffer, 0, logits.byteLength))
    return output
  }
  new Uint32Array(buffer, 0, 1)[0] = header
  return buffer
}

const halfScales = Float32Array.from({ length: 32 }, (_, exponent) => 2 ** (Math.max(1, exponent) - 25))

function halfValue(bits) {
  const exponent = (bits >>> 10) & 31
  const mantissa = bits & 1023
  if (exponent === 31) return mantissa ? NaN : bits & 0x8000 ? -Infinity : Infinity
  const value = (exponent ? mantissa + 1024 : mantissa) * halfScales[exponent]
  return bits & 0x8000 ? -value : value
}

export function candidateMoves(board, allowed, moves = [], mapping = null) {
  // If every empty cell is pruned, search them all.
  const mask = allowed?.some(Boolean) ? allowed : null
  let index = 0
  if (mapping) {
    // Restore the original policy order when an inherited subtree rotates.
    for (const point of mapping) {
      if (mask ? mask[point >>> 5] & (1 << (point & 31)) : !board[point]) moves[index++] = point
    }
  } else if (mask) {
    for (let i = 0; i < mask.length; i++) {
      for (let word = mask[i]; word; word &= word - 1) {
        moves[index++] = i * 32 + 31 - Math.clz32(word & -word)
      }
    }
  } else {
    for (let point = 0; point < board.length; point++) {
      if (!board[point]) moves[index++] = point
    }
  }
  return moves
}

export function unpackColdCandidates(buffer, board, allowed, mapping) {
  const points = board.length
  const header = new Uint32Array(buffer, 0, 1)[0]
  const count = header >>> HEADER_SHIFT
  const LogitArray = header & HALF ? Float16 ?? Uint16Array : Float32Array
  const logits = header & PACKED_HALF ? readPackedHalf(buffer, count) : new LogitArray(buffer, COLD_HEADER_BYTES, count)
  const MoveArray = moveArrayType(points)
  const moves = candidateMoves(board, allowed, new MoveArray(count), mapping)
  const policy = policyScratch(points)
  if (logits instanceof Uint16Array) {
    for (let i = 0; i < count; i++) policy[moves[i]] = halfValue(logits[i])
  } else {
    for (let i = 0; i < count; i++) policy[moves[i]] = logits[i]
  }
  return { moves, policy }
}
