import { toHalf } from "./precision.js"

const Float16 = globalThis.Float16Array
export const COLD_HEADER_BYTES = Uint32Array.BYTES_PER_ELEMENT
// The low bits describe storage, the remaining bits give the candidate count.
const BITMAP = 1
const HALF = 2
const HEADER_SHIFT = 2

function moveArrayType(points) {
  // Move IDs index the full square canvas in every game.
  return points <= 256 ? Uint8Array : Uint16Array
}

export function coldLogitSize(half) {
  return half ? Uint16Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT
}

export function expandedCandidateBytes(count, points) {
  return count * (Float32Array.BYTES_PER_ELEMENT + moveArrayType(points).BYTES_PER_ELEMENT)
}

export function candidateBytes(children) {
  return children instanceof ArrayBuffer ? children.byteLength : children.moves.buffer.byteLength
}

export function coldCandidateCount(buffer) {
  return new Uint32Array(buffer, 0, 1)[0] >>> HEADER_SHIFT
}

// Candidate data is immutable between root evaluations and shared by snapshots.
export function packCandidates(moves, priors, points, { mapping, order } = {}) {
  const MoveArray = moveArrayType(points)
  const moveOffset = moves.length * Float32Array.BYTES_PER_ELEMENT
  const buffer = new ArrayBuffer(moveOffset + moves.length * MoveArray.BYTES_PER_ELEMENT)
  const packedPriors = new Float32Array(buffer, 0, moves.length)
  const packedMoves = new MoveArray(buffer, moveOffset, moves.length)
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

// Count packed bits a word at a time before allocating the exact policy buffer.
function bitmapCount(bitmap) {
  let count = 0
  for (let word of bitmap) {
    word -= (word >>> 1) & 0x55555555
    word = (word & 0x33333333) + ((word >>> 2) & 0x33333333)
    count += Math.imul((word + (word >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24
  }
  return count
}

// Both forms enumerate moves in ascending canvas order.
export function packColdCandidates({ moves, moveBitmap }, policy, points, half) {
  const count = moveBitmap ? bitmapCount(moveBitmap) : moves.length
  const MoveArray = moveArrayType(points)
  const LogitArray = half ? Float16 ?? Uint16Array : Float32Array
  const logitBytes = count * LogitArray.BYTES_PER_ELEMENT
  const bitmapWords = Math.ceil(points / 32)
  const moveBytes = count * MoveArray.BYTES_PER_ELEMENT
  const bitmapBytes = bitmapWords * Uint32Array.BYTES_PER_ELEMENT
  const useBitmap = bitmapBytes < moveBytes
  const buffer = new ArrayBuffer(COLD_HEADER_BYTES + logitBytes + (useBitmap ? bitmapBytes : moveBytes))
  new Uint32Array(buffer, 0, 1)[0] = (count << HEADER_SHIFT) | (half ? HALF : 0) | (useBitmap ? BITMAP : 0)
  const logits = new LogitArray(buffer, COLD_HEADER_BYTES + (useBitmap ? bitmapBytes : 0), count)
  const values = half && !Float16 ? policyScratch(count).subarray(0, count) : logits
  if (useBitmap) {
    const bitmap = new Uint32Array(buffer, COLD_HEADER_BYTES, bitmapWords)
    if (moveBitmap) {
      bitmap.set(moveBitmap)
      let index = 0
      for (let i = 0; i < bitmap.length; i++) {
        for (let word = bitmap[i]; word; word &= word - 1) {
          values[index++] = policy[i * 32 + 31 - Math.clz32(word & -word)]
        }
      }
    } else {
      for (let i = 0; i < count; i++) {
        const move = moves[i]
        bitmap[move >>> 5] |= 1 << (move & 31)
        values[i] = policy[move]
      }
    }
  } else {
    const packed = new MoveArray(buffer, COLD_HEADER_BYTES + logitBytes, count)
    if (moveBitmap) bitmapMoves(moveBitmap, packed)
    else packed.set(moves)
    for (let i = 0; i < count; i++) values[i] = policy[packed[i]]
  }
  if (values !== logits) toHalf(values, logits)
  return buffer
}

// Enumerate in original order so remapped policies keep reduction order and ties.
function bitmapMoves(bitmap, result, mapping) {
  let count = 0
  for (let i = 0; i < bitmap.length; i++) {
    for (let word = bitmap[i]; word; word &= word - 1) {
      const point = i * 32 + 31 - Math.clz32(word & -word)
      result[count++] = mapping ? mapping[point] : point
    }
  }
}

const halfScales = Float32Array.from({ length: 32 }, (_, exponent) => 2 ** (Math.max(1, exponent) - 25))

function halfValue(bits) {
  const exponent = (bits >>> 10) & 31
  const mantissa = bits & 1023
  if (exponent === 31) return mantissa ? NaN : bits & 0x8000 ? -Infinity : Infinity
  const value = (exponent ? mantissa + 1024 : mantissa) * halfScales[exponent]
  return bits & 0x8000 ? -value : value
}

export function unpackColdCandidates(buffer, points, mapping) {
  const header = new Uint32Array(buffer, 0, 1)[0]
  const count = header >>> HEADER_SHIFT
  const bitmap = (header & BITMAP) !== 0
  const half = (header & HALF) !== 0
  const MoveArray = moveArrayType(points)
  const LogitArray = half ? Float16 ?? Uint16Array : Float32Array
  const bitmapWords = bitmap ? Math.ceil(points / 32) : 0
  const logits = new LogitArray(buffer, COLD_HEADER_BYTES + bitmapWords * Uint32Array.BYTES_PER_ELEMENT, count)
  const moveOffset = COLD_HEADER_BYTES + logits.byteLength
  let moves = bitmap ? new MoveArray(count) : new MoveArray(buffer, moveOffset, count)
  if (bitmap) bitmapMoves(new Uint32Array(buffer, COLD_HEADER_BYTES, bitmapWords), moves, mapping)
  else if (mapping) moves = moves.map((point) => mapping[point])
  const policy = policyScratch(points)
  if (logits instanceof Uint16Array) {
    for (let i = 0; i < count; i++) policy[moves[i]] = halfValue(logits[i])
  } else {
    for (let i = 0; i < count; i++) policy[moves[i]] = logits[i]
  }
  return { moves, policy }
}
