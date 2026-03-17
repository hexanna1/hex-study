// SPDX-License-Identifier: MIT
// Board symmetries and root move pruning follow KataHex's nninputs.cpp.
const HEX_SYMMETRIES = [0, 1]
const Y_SYMMETRIES = [0, 1, 2, 3, 4, 5]
// Permutations of the three barycentric coordinates of the Y triangle.
const Y_PERMUTATIONS = [
  [0, 1, 2], [1, 2, 0], [2, 0, 1],
  [1, 0, 2], [0, 2, 1], [2, 1, 0],
]

export function symmetriesFor(game) {
  return game === "y" ? Y_SYMMETRIES : HEX_SYMMETRIES
}

export function symmetryPoint(point, size, game, symmetry) {
  if (game === "hex") return symmetry ? size * size - 1 - point : point
  if (!symmetry) return point
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
  // Match native representative ordering: right-to-left columns for Black,
  // left-to-right for White, with the opposite row order.
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
