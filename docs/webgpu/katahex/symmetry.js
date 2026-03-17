// SPDX-License-Identifier: MIT
// Board symmetries and root move pruning follow KataHex's nninputs.cpp.
const HEX_SYMMETRIES = [0, 1]
const Y_SYMMETRIES = [0, 1, 2, 3, 4, 5]
const YAVALATH_SYMMETRIES = Array.from({ length: 12 }, (_, i) => i)
// Permutations of the three barycentric coordinates of the Y triangle.
const Y_PERMUTATIONS = [
  [0, 1, 2], [1, 2, 0], [2, 0, 1],
  [1, 0, 2], [0, 2, 1], [2, 1, 0],
]

export function symmetriesFor(game) {
  return game === "yavalath" ? YAVALATH_SYMMETRIES : game === "y" ? Y_SYMMETRIES : HEX_SYMMETRIES
}

export function symmetryPoint(point, size, game, symmetry) {
  if (game === "hex") return symmetry ? size * size - 1 - point : point
  if (!symmetry) return point
  if (game === "yavalath") {
    const radius = (size - 1) / 2
    const coords = [point % size - radius, Math.floor(point / size) - radius]
    coords.push(-coords[0] - coords[1])
    const permutation = Y_PERMUTATIONS[symmetry % 6]
    let inversions = 0
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) if (permutation[i] > permutation[j]) inversions++
    const sign = (inversions % 2 ? -1 : 1) * (symmetry >= 6 ? -1 : 1)
    const x = point % size, y = Math.floor(point / size)
    if (x + y < radius || x + y > 3 * radius) return point
    return (sign * coords[permutation[1]] + radius) * size + sign * coords[permutation[0]] + radius
  }
  const x = point % size, y = Math.floor(point / size)
  if (x + y >= size) return point
  const coords = [x, y, size - 1 - x - y]
  const permutation = Y_PERMUTATIONS[symmetry]
  return coords[permutation[1]] * size + coords[permutation[0]]
}

export function rootSymmetryPruning({ board, size, toPlay }, game) {
  const symmetries = symmetriesFor(game).filter((symmetry) =>
    symmetry === 0 || board.every((color, point) => color === board[symmetryPoint(point, size, game, symmetry)]))
  const duplicates = new Uint8Array(board.length)
  // Match native representative ordering.
  if (symmetries.length > 1) {
    for (let col = 0; col < size; col++) {
      for (let row = 0; row < size; row++) {
        const x = toPlay === 1 ? size - 1 - col : col
        const y = toPlay === 1 ? row : size - 1 - row
        const move = y * size + x
        if (board[move] || duplicates[move]) continue
        for (const symmetry of symmetries) {
          const other = symmetryPoint(move, size, game, symmetry)
          if (other !== move) duplicates[other] = 1
        }
      }
    }
  }
  return { symmetries, duplicates }
}
