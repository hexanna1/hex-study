// Yavalath terminal value from the next player's perspective, on legal histories.
;(() => {
  function terminalValue(state) {
    const previous = 3 - state.toPlay
    let three = false
    for (let point = 0; point < state.board.length; point++) {
      if (state.board[point] !== previous) continue
      const x = point % state.size, y = Math.floor(point / state.size)
      for (const [dx, dy] of [[1, 0], [0, 1], [1, -1]]) {
        let length = 0, nx = x, ny = y
        while (
          globalThis.BoardGeometry.isPlayable("yavalath", nx, ny, state.size) &&
          state.board[ny * state.size + nx] === previous
        ) {
          length++
          nx += dx
          ny += dy
        }
        if (length >= 4) return -1
        if (length === 3) three = true
      }
    }
    if (three) return 1
    return state.board.includes(0) ? null : 0
  }
  globalThis.YavalathRules = { terminalValue }
})()
