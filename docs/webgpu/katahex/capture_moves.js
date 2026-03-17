// Playing a stone only changes the capture patterns of neighboring empty cells.
export class CaptureMoves {
  constructor(board, neighbors, table) {
    const points = board.length
    this.table = table
    this.rootPatterns = new Uint16Array(points)
    this.rootAllowed = new Uint32Array(Math.ceil(points / 32))
    this.patterns = new Uint16Array(points)
    this.allowed = new Uint32Array(this.rootAllowed.length)

    // The 42² canvas fits 12-bit cell IDs, and six references per cell fit 16-bit offsets.
    // The remaining four bits of each reference hold the neighbor's shift.
    const offsets = new Uint16Array(points + 1)
    for (let point = 0; point < points; point++) {
      if (board[point]) continue
      let pattern = 0
      for (let i = 0; i < 6; i++) {
        const neighbor = neighbors[point * 6 + i]
        const color = neighbor >= 0 ? board[neighbor] : -neighbor
        pattern |= color << (2 * i)
        if (color === 0) offsets[neighbor + 1]++
      }
      this.rootPatterns[point] = pattern
      if (!table[pattern]) this.rootAllowed[point >>> 5] |= 1 << (point & 31)
    }
    for (let point = 0; point < points; point++) offsets[point + 1] += offsets[point]
    this.offsets = offsets
    const next = offsets.slice()
    this.references = new Uint16Array(offsets[points])
    for (let point = 0; point < points; point++) {
      if (board[point]) continue
      for (let i = 0; i < 6; i++) {
        const neighbor = neighbors[point * 6 + i]
        // Each reference pairs the cell with the bit offset of this neighbor.
        if (neighbor >= 0 && board[neighbor] === 0) {
          this.references[next[neighbor]++] = (point << 4) | (2 * i)
        }
      }
    }
  }

  reset() {
    this.patterns.set(this.rootPatterns)
    this.allowed.set(this.rootAllowed)
  }

  play(move, color, board) {
    const { patterns, allowed, offsets, references, table } = this
    allowed[move >>> 5] &= ~(1 << (move & 31))
    for (let i = offsets[move]; i < offsets[move + 1]; i++) {
      const reference = references[i], point = reference >>> 4
      if (board[point]) continue
      // A legal move only replaces a zero-valued neighbor slot.
      const pattern = patterns[point] | (color << (reference & 15))
      patterns[point] = pattern
      const word = point >>> 5, bit = 1 << (point & 31)
      if (table[pattern]) allowed[word] &= ~bit
      else allowed[word] |= bit
    }
  }
}
