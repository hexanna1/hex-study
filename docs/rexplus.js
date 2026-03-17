(() => {
  function makeOneCellQuery({ size, root, children }) {
    const cells = size * size
    const path = new Uint32Array(cells)
    const pathCells = new Uint8Array(cells)
    return (board, transposeColors, allowed) => {
      path.fill(0)
      let edge = root
      let length = 0
      let covered = 0
      while (edge >= 2) {
        const p = 3 * (edge >>> 1)
        const cell = children[p + Number(transposeColors)] >>> 26
        path[cell] = edge
        pathCells[length++] = cell
        covered |= 1 << cell
        const digit = board[cell]
        const branch = transposeColors ? (24 >>> (2 * digit)) & 3 : digit
        edge = (children[p + branch] & 0x03ffffff) ^ (edge & 1)
      }
      const original = edge
      if (!original && allowed & ~covered) return true
      for (let k = 0; k < length; k++) {
        const cell = pathCells[k]
        if (!(allowed & (1 << cell))) continue
        const prefix = path[cell]
        // Transposition swaps colors, so the current player's stone is always digit 1.
        let candidate = (children[3 * (prefix >>> 1) + 1] & 0x03ffffff) ^ (prefix & 1)
        while (candidate >= 2) {
          const p = 3 * (candidate >>> 1)
          const next = children[p + Number(transposeColors)] >>> 26
          if (candidate >>> 1 === path[next] >>> 1) {
            candidate = original ^ ((candidate ^ path[next]) & 1)
            break
          }
          const digit = board[next]
          const branch = transposeColors ? (24 >>> (2 * digit)) & 3 : digit
          candidate = (children[p + branch] & 0x03ffffff) ^ (candidate & 1)
        }
        if (!candidate) return true
      }
      return false
    }
  }

  function makeOracle(table) {
    const { size } = table
    const cells = size * size
    const full = 2 ** cells - 1
    const query = makeOneCellQuery(table)
    let west = 0
    for (let row = 0; row < size; row++) west |= 1 << (row * size)
    const east = west << (size - 1)
    const north = (1 << size) - 1
    const south = north << (cells - size)
    const spread = (x) =>
      ((x << 1) & ~west) |
      ((x >>> 1) & ~east) |
      (x << size) |
      (x >>> size) |
      ((x << (size - 1)) & ~east) |
      ((x >>> (size - 1)) & ~west)

    function boundary(stones, goal) {
      let seen = stones & goal
      for (;;) {
        const next = seen | (spread(seen) & stones)
        if (seen === next) return (spread(seen) | goal) & full
        seen = next
      }
    }

    return (board) => {
      let black = 0
      let white = 0
      for (let cell = 0; cell < cells; cell++) {
        if (board[cell] === 1) black |= 1 << cell
        if (board[cell] === 2) white |= 1 << cell
      }
      // A placement loses immediately if it touches components from both goals.
      const blackUnsafe = boundary(black, north) & boundary(black, south)
      const whiteUnsafe = boundary(white, west) & boundary(white, east)
      const empty = full & ~(black | white)
      return Number(query(board, false, empty & ~blackUnsafe)) | (Number(query(board, true, empty & ~whiteUnsafe)) << 1)
    }
  }

  window.HexRexUI({
    gameName: "Rex+",
    gamePath: "rexplus",
    sizes: [2, 3, 4, 5],
    defaultSize: 5,
    makeOracle,
    batchTurns: true,
  })
})()
