(() => {
  const { createSvgTools, rgbText, THEME } = window.HexStudyUI
  const { pointKey } = window.HexPosition

  function draw(svg, { game, boardSize, board, orientation = "flat" }) {
    if (game === "hex") {
      const boardSvg = window.HexMoveTree.createBoardSvg(svg)
      boardSvg.setBoardOrientation(orientation)
      window.HexMoveTree.renderMoveTreeBoard({
        boardSvg,
        boardSize,
        boardState: board,
        enableCellClicks: false,
        showMoveNumbers: false,
      })
      return
    }
    if (game === "bentY") {
      svg.replaceChildren()
      const geometry = globalThis.BentYGeometry
      const graph = geometry.topology(boardSize)
      const scale = geometry.widthForTensorLen(boardSize) * 63
      const tools = createSvgTools({
        board: svg,
        hexSize: 24,
        defaultFill: rgbText(THEME.OFF_WHITE_RGB),
        defaultStroke: THEME.GRID_EDGE,
        defaultStrokeWidth: 1,
        projectPoint: (_size, col, row, direction) =>
          geometry.projectPoint(boardSize, col, row, direction).map((value) => value * scale),
      })
      tools.setBoardOrientation(orientation)
      const pixel = (point) => tools.pointToPixel((point % boardSize) + 1, Math.floor(point / boardSize) + 1)
      for (const [a, b] of graph.edges) {
        const boundary = graph.sideMasks[a] & graph.sideMasks[b]
        tools.appendLine(...pixel(a), ...pixel(b), boundary ? "#5c5348" : "#9b9080", boundary ? 3.2 : 1.8)
      }
      const cells = graph.playable.map((point) => ({
        col: (point % boardSize) + 1,
        row: Math.floor(point / boardSize) + 1,
      }))
      for (const { col, row } of cells) {
        const stone = board.occupied.get(pointKey(col, row))
        const [cx, cy] = tools.pointToPixel(col, row)
        tools.appendCircle(cx, cy, 26, {
          stroke: "#75664f",
          strokeWidth: 1.7,
          fill: stone ? rgbText(stone.color === "red" ? THEME.RED_RGB : THEME.BLUE_RGB) : rgbText(THEME.OFF_WHITE_RGB),
        })
        if (stone?.isLast) tools.appendCircle(cx, cy, 3, { fill: rgbText(THEME.OFF_WHITE_RGB) })
      }
      tools.setViewBoxFromPoints({ boardPoints: cells, coordPoints: [], viewPadding: 32, coordViewPadding: 0 })
      return
    }
    svg.replaceChildren()
    const tools = createSvgTools({
      board: svg,
      hexSize: 24,
      defaultFill: rgbText(THEME.OFF_WHITE_RGB),
      defaultStroke: THEME.GRID_EDGE,
      defaultStrokeWidth: 1,
    })
    tools.setBoardOrientation(orientation)
    const cells = []
    for (let row = 1; row <= boardSize; row++) {
      for (let col = 1; col <= boardSize; col++) {
        if (!globalThis.BoardGeometry.isPlayable(game, col - 1, row - 1, boardSize)) continue
        cells.push({ col, row })
        const stone = board.occupied.get(pointKey(col, row))
        const hex = tools.appendHex(col, row, {
          fill: stone ? rgbText(stone.color === "red" ? THEME.RED_RGB : THEME.BLUE_RGB) : rgbText(THEME.OFF_WHITE_RGB),
        })
        if (stone?.isLast) tools.appendCircle(hex.cx, hex.cy, 3, { fill: rgbText(THEME.OFF_WHITE_RGB) })
      }
    }
    tools.setViewBoxFromPoints({ boardPoints: cells, coordPoints: [], viewPadding: 26, coordViewPadding: 0 })
  }

  function place(card, x, y) {
    card.hidden = false
    const rect = card.getBoundingClientRect()
    const left = Math.max(8, Math.min(x + 14, window.innerWidth - rect.width - 8))
    const top =
      y - rect.height - 12 >= 8
        ? y - rect.height - 12
        : Math.max(8, Math.min(y + 16, window.innerHeight - rect.height - 8))
    card.style.left = `${left}px`
    card.style.top = `${top}px`
  }

  window.BoardGraphPreview = { draw, place }
})()
