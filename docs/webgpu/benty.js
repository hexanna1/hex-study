// SPDX-License-Identifier: MIT
// Bent-Y input features and graph attention bias adapted from KataGomo.
import "../benty_geometry.js"
import { ONLINE_Y_KEYS_PER_TILE, ONLINE_Y_QUERY_ROWS } from "./y_attention.js"

const biasFeaturesBySize = new Map()
export const BENT_SIDE_BIT = 13

// Match the spatial channel bits, followed by the scalar global input.
export function bentYFeatureRows(states, size, points, emit) {
  const { neighbors, sideMasks } = globalThis.BentYGeometry.topology(size)
  const componentSides = new Uint8Array(size * size)
  const queue = new Uint16Array(points.length)
  for (const [batch, { board, toPlay }] of states.entries()) {
    // Side masks range from 0 to 7. Mark each stone when it enters the queue.
    componentSides.fill(255)
    for (const start of points) {
      const color = board[start]
      if (!color || componentSides[start] !== 255) continue
      let count = 1, sides = 0
      componentSides[start] = 0
      queue[0] = start
      for (let index = 0; index < count; index++) {
        const point = queue[index]
        sides |= sideMasks[point]
        for (const next of neighbors[point]) {
          if (board[next] === color && componentSides[next] === 255) {
            componentSides[next] = 0
            queue[count++] = next
          }
        }
      }
      for (let index = 0; index < count; index++) componentSides[queue[index]] = sides
    }

    const side = Number(toPlay === 2) << BENT_SIDE_BIT
    for (const point of points) {
      const stone = board[point]
      let features = side | 1 | (Number(neighbors[point].length === 5) << 4)
      if (stone) {
        features |= stone === toPlay ? 2 : 4
      } else {
        let ownCount = 0, otherCount = 0, ownSides = sideMasks[point], otherSides = sideMasks[point]
        for (const next of neighbors[point]) {
          if (board[next] === toPlay) {
            ownCount++
            ownSides |= componentSides[next]
          } else if (board[next] === 3 - toPlay) {
            otherCount++
            otherSides |= componentSides[next]
          }
        }
        features |= (Number(ownCount >= 1) << 5) | (Number(ownCount >= 2) << 6) |
          (Number(otherCount >= 1) << 7) | (Number(otherCount >= 2) << 8) |
          (Number((ownSides & (ownSides - 1)) !== 0) << 9) | (Number((otherSides & (otherSides - 1)) !== 0) << 10) |
          (Number(ownSides === 7) << 11) | (Number(otherSides === 7) << 12)
      }
      emit(batch, point, features)
    }
  }
}

export function graphBiasFeatures(size) {
  if (biasFeaturesBySize.has(size)) return biasFeaturesBySize.get(size)
  const { playable, neighbors, sideMasks } = globalThis.BentYGeometry.topology(size)
  const indices = new Map(playable.map((point, index) => [point, index]))
  const count = playable.length
  let maxDistance = 0
  const distances = playable.map((start) => {
    const distance = new Int16Array(size * size).fill(-1), queue = [start]
    distance[start] = 0
    for (let i = 0; i < queue.length; i++) {
      const point = queue[i]
      for (const next of neighbors[point]) {
        if (distance[next] < 0) {
          distance[next] = distance[point] + 1
          queue.push(next)
        }
      }
    }
    maxDistance = Math.max(maxDistance, distance[queue.at(-1)])
    return Int16Array.from(playable, (point) => distance[point])
  })
  const corners = [5, 3, 6].map((mask) => indices.get(playable.find((point) => sideMasks[point] === mask)))
  const [cornerA, cornerB, cornerC] = corners.map((corner) => distances[corner])
  const cornerDistance = cornerA[corners[1]]
  // Numeric keys preserve the lexicographic order of the corner distance pairs.
  const distanceBase = maxDistance + 1
  const pairBase = distanceBase * distanceBase
  const features = []
  const groups = new Map()
  // Each word holds group indices for two consecutive keys in one query row.
  // Round both axes up to dispatch tiles. Invalid scores remain masked.
  const pairStride = Math.ceil(count / ONLINE_Y_KEYS_PER_TILE) * (ONLINE_Y_KEYS_PER_TILE / 2)
  const queryRows = Math.ceil(count / ONLINE_Y_QUERY_ROWS) * ONLINE_Y_QUERY_ROWS
  const packedPairs = new Uint32Array(queryRows * pairStride)
  for (let q = 0; q < count; q++) {
    const fromA = cornerA[q] * distanceBase, fromB = cornerB[q] * distanceBase, fromC = cornerC[q] * distanceBase
    for (let k = 0; k < count; k++) {
      const distance = distances[q][k]
      const pairA = fromA + cornerA[k], pairB = fromB + cornerB[k], pairC = fromC + cornerC[k]
      const first = Math.min(pairA, pairB, pairC), last = Math.max(pairA, pairB, pairC)
      const middle = pairA + pairB + pairC - first - last
      const key = ((distance * pairBase + first) * pairBase + middle) * pairBase + last
      let group = groups.get(key)
      if (group === undefined) {
        group = groups.size
        groups.set(key, group)
        features.push(1 / (1 + distance), Number(distance === 0), Number(distance === 1), Number(distance === 2))
        for (const pair of [first, middle, last]) {
          const from = Math.floor(pair / distanceBase), to = pair % distanceBase
          features.push(from / cornerDistance, to / cornerDistance)
        }
      }
      packedPairs[q * pairStride + (k >> 1)] |= group << ((k & 1) * 16)
    }
  }
  const grouped = { features: Float32Array.from(features), packedPairs, pairStride, groups: groups.size }
  biasFeaturesBySize.set(size, grouped)
  return grouped
}

export function graphAttentionBias({ features, groups }, block) {
  const { biasInput, biasOutput, heads } = block
  const groupBias = new Float32Array(heads * groups)
  const hidden = new Float32Array(biasInput.output)
  for (let group = 0; group < groups; group++) {
    for (let h = 0; h < hidden.length; h++) {
      let value = 0
      for (let f = 0; f < 10; f++) value += features[group * 10 + f] * biasInput.weights[f * hidden.length + h]
      hidden[h] = value / (1 + Math.exp(-value))
    }
    for (let head = 0; head < heads; head++) {
      let value = 0
      for (let h = 0; h < hidden.length; h++) value += hidden[h] * biasOutput.weights[h * heads + head]
      groupBias[head * groups + group] = value
    }
  }
  return groupBias
}
