// Yavalath terminal value from the next player's perspective.
;(() => {
  const masksBySize = new Map()

  function masksFor(size) {
    let masks = masksBySize.get(size)
    if (masks) return masks
    const { isPlayable, playablePoints } = globalThis.BoardGeometry
    const points = playablePoints("yavalath", size)
    masks = [[1, 0], [0, 1], [-1, 1]].map(([dx, dy]) => {
      const triple = new Uint32Array(3)
      const four = new Uint32Array(triple.length)
      // Limit the starts to runs that fit on the board, including across word boundaries.
      for (const point of points) {
        const x = point % size, y = Math.floor(point / size)
        if (!isPlayable("yavalath", x + 2 * dx, y + 2 * dy, size)) continue
        triple[point >>> 5] |= 1 << (point & 31)
        if (isPlayable("yavalath", x + 3 * dx, y + 3 * dy, size)) four[point >>> 5] |= 1 << (point & 31)
      }
      return { shift: dx + dy * size, triple, four }
    })
    masksBySize.set(size, masks)
    return masks
  }

  function terminalValue({ board, size, toPlay }) {
    const previous = 3 - toPlay
    // The largest supported canvas is 9×9, which fits in three 32-bit words.
    let low = 0, middle = 0, high = 0
    for (let point = 0; point < Math.min(32, board.length); point++) low |= (board[point] === previous) << point
    for (let point = 32; point < Math.min(64, board.length); point++) middle |= (board[point] === previous) << (point - 32)
    for (let point = 64; point < board.length; point++) high |= (board[point] === previous) << (point - 64)
    let three = false
    for (const { shift, triple, four } of masksFor(size)) {
      const pairs0 = low & (low >>> shift | middle << (32 - shift))
      const pairs1 = middle & (middle >>> shift | high << (32 - shift))
      const pairs2 = high & (high >>> shift)
      const step = 2 * shift
      // A four starts where two pairs begin two cells apart.
      if ((pairs0 & (pairs0 >>> step | pairs1 << (32 - step)) & four[0]) ||
          (pairs1 & (pairs1 >>> step | pairs2 << (32 - step)) & four[1]) ||
          (pairs2 & (pairs2 >>> step) & four[2])) return -1
      three ||= Boolean((pairs0 & (low >>> step | middle << (32 - step)) & triple[0]) ||
        (pairs1 & (middle >>> step | high << (32 - step)) & triple[1]) ||
        (pairs2 & (high >>> step) & triple[2]))
    }
    if (three) return 1
    return board.includes(0) ? null : 0
  }
  globalThis.YavalathRules = { terminalValue }
})()
