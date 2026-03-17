// Bent-Y's graph-local input planes and position-independent attention biases.
import "../../benty_geometry.js"

export function bentYInput(board, size, toPlay) {
  const { playable, neighbors, sideMasks } = globalThis.BentYGeometry.topology(size)
  const componentSides = new Uint8Array(board.length)
  const seen = new Uint8Array(board.length)
  for (const start of playable) {
    const color = board[start]
    if (!color || seen[start]) continue
    const stack = [start], members = []
    let sides = 0
    seen[start] = 1
    while (stack.length) {
      const point = stack.pop()
      members.push(point)
      sides |= sideMasks[point]
      for (const next of neighbors[point]) {
        if (board[next] === color && !seen[next]) {
          seen[next] = 1
          stack.push(next)
        }
      }
    }
    for (const point of members) componentSides[point] = sides
  }
  const features = new Uint8Array(playable.length * 22)
  playable.forEach((point, index) => {
    const offset = index * 22, stone = board[point]
    features[offset] = 1
    if (stone) features[offset + (stone === toPlay ? 1 : 2)] = 1
    features[offset + 4] = Number(neighbors[point].length === 5)
    if (stone) return
    let ownCount = 0, otherCount = 0, ownSides = sideMasks[point], otherSides = ownSides
    for (const next of neighbors[point]) {
      if (board[next] === toPlay) {
        ownCount++
        ownSides |= componentSides[next]
      } else if (board[next] === 3 - toPlay) {
        otherCount++
        otherSides |= componentSides[next]
      }
    }
    features[offset + 5] = Number(ownCount >= 1)
    features[offset + 6] = Number(ownCount >= 2)
    features[offset + 7] = Number(otherCount >= 1)
    features[offset + 8] = Number(otherCount >= 2)
    features[offset + 9] = Number((ownSides & (ownSides - 1)) !== 0)
    features[offset + 10] = Number((otherSides & (otherSides - 1)) !== 0)
    features[offset + 11] = Number(ownSides === 7)
    features[offset + 12] = Number(otherSides === 7)
  })
  return features
}

export function graphBiasFeatures(size) {
  const { playable, neighbors, sideMasks } = globalThis.BentYGeometry.topology(size)
  const indices = new Map(playable.map((point, index) => [point, index]))
  const count = playable.length
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
    return Int16Array.from(playable, (point) => distance[point])
  })
  const corners = [5, 3, 6].map((mask) => indices.get(playable.find((point) => sideMasks[point] === mask)))
  const diameter = distances[corners[0]][corners[1]]
  const features = new Float32Array(count * count * 10)
  for (let q = 0; q < count; q++)
    for (let k = 0; k < count; k++) {
      const distance = distances[q][k], offset = (q * count + k) * 10
      features.set([1 / (1 + distance), Number(distance === 0), Number(distance === 1), Number(distance === 2)], offset)
      const pairs = corners
        .map((corner) => [distances[corner][q], distances[corner][k]])
        .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      pairs.forEach((pair, i) => {
        features[offset + 4 + i * 2] = pair[0] / diameter
        features[offset + 5 + i * 2] = pair[1] / diameter
      })
    }
  return { features, count }
}

export function graphAttentionBias({ features, count }, block) {
  const { biasInput, biasOutput, heads } = block
  const bias = new Float32Array(heads * count * count), hidden = new Float32Array(biasInput.output)
  for (let pair = 0; pair < count * count; pair++) {
    for (let h = 0; h < hidden.length; h++) {
      let value = 0
      for (let f = 0; f < 10; f++) value += features[pair * 10 + f] * biasInput.weights[f * hidden.length + h]
      hidden[h] = value / (1 + Math.exp(-value))
    }
    for (let head = 0; head < heads; head++) {
      let value = 0
      for (let h = 0; h < hidden.length; h++) value += hidden[h] * biasOutput.weights[h * heads + head]
      bias[head * count * count + pair] = value
    }
  }
  return bias
}
