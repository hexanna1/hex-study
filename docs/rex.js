(() => {
  const { makeLookup } = window.HexRexTable

  function makeOracle(table) {
    const lookup = makeLookup(table, { prefix: table.size === 6 ? 12 : 0 })
    const oracle = (board) => {
      let parity = 0
      for (const digit of board) parity ^= digit === 0
      const regular = lookup(board, false)
      const dual = lookup(board, true)
      return parity ? dual | (regular << 1) : (regular ^ 1) | ((dual ^ 1) << 1)
    }
    return addPlacementQueries(oracle, table, lookup.dispatch)
  }

  function addPlacementQueries(oracle, { root, children, size }, dispatch) {
    const cells = size * size
    const alternate = new Uint32Array(cells * 2)
    const path = new Uint32Array(cells)
    const { heads, order, dualOrder, prefix } = dispatch
    const powers = Uint32Array.from({ length: prefix }, (_, level) => 3 ** (prefix - 1 - level))

    function direct(value, board, original) {
      while (value >= 2) {
        const p = 3 * (value >>> 1)
        const cell = children[p] >>> 26
        const saved = path[cell]
        if (value >>> 1 === saved >>> 1) { return original ^ ((value ^ saved) & 1) }
        const digit = board[cell]
        value = (children[p + digit] & 0x03ffffff) ^ (value & 1)
      }
      return value
    }

    function dual(value, board, original) {
      while (value >= 2) {
        const p = 3 * (value >>> 1)
        const cell = children[p + 1] >>> 26
        const saved = path[cell]
        if (value >>> 1 === saved >>> 1) { return original ^ ((value ^ saved) & 1) }
        const digit = board[cell]
        value = (children[p + ((24 >>> (2 * digit)) & 3)] & 0x03ffffff) ^ (value & 1)
      }
      return value
    }

    function neighborhood(board, transposeColors, answer, parity) {
      // Reuse the base path for one-cell alternatives, carrying complement phases.
      alternate.fill(0xffffffff)
      path.fill(0)
      let value = root
      if (prefix) {
        const positions = transposeColors ? dualOrder : order
        let code = 0
        for (let level = 0; level < prefix; level++) {
          const digit = board[positions[level]]
          code = 3 * code + (transposeColors ? (24 >>> (2 * digit)) & 3 : digit)
        }
        value = heads[code]
        for (let level = 0; level < prefix; level++) {
          const cell = positions[level]
          if (!board[cell]) {
            alternate[2 * cell] = heads[code + (transposeColors ? 2 : 1) * powers[level]]
            alternate[2 * cell + 1] = heads[code + (transposeColors ? 1 : 2) * powers[level]]
          }
        }
      }
      while (value >= 2) {
        const p = 3 * (value >>> 1)
        const cell = children[p + Number(transposeColors)] >>> 26
        const digit = board[cell]
        const phase = value & 1
        path[cell] = value
        if (!digit) {
          alternate[2 * cell] = (children[p + (transposeColors ? 2 : 1)] & 0x03ffffff) ^ phase
          alternate[2 * cell + 1] = (children[p + (transposeColors ? 1 : 2)] & 0x03ffffff) ^ phase
        }
        const branch = transposeColors ? (24 >>> (2 * digit)) & 3 : digit
        value = (children[p + branch] & 0x03ffffff) ^ phase
      }
      for (let cell = 0; cell < cells; cell++) {
        if (board[cell]) continue
        for (let color = 0; color < 2; color++) {
          const index = 2 * cell + color
          const reference = alternate[index]
          const result =
            reference === 0xffffffff
              ? value
              : transposeColors
                ? dual(reference, board, value)
                : direct(reference, board, value)
          if (!transposeColors) { answer[index] = result } else {
            const regular = answer[index]
            answer[index] = parity ? result | (regular << 1) : (regular ^ 1) | ((result ^ 1) << 1)
          }
        }
      }
    }

    // Two outcome masks per cell, for colors 1 and 2; the UI checks connecting losses.
    oracle.placements = (board) => {
      let parity = 1
      for (let cell = 0; cell < cells; cell++) parity ^= board[cell] === 0
      const answer = new Uint8Array(cells * 2)
      answer.fill(255)
      neighborhood(board, false, answer, parity)
      neighborhood(board, true, answer, parity)
      return answer
    }
    return oracle
  }

  window.HexRexUI({
    gameName: "Rex",
    gamePath: "rex",
    sizes: [2, 3, 4, 5, 6],
    defaultSize: 6,
    makeOracle,
    batchTurns: false,
  })
})()
