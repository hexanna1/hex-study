(() => {
  const { createSvgTools, rgbText, THEME } = window.HexStudyUI
  const { pointKey } = window.HexPosition

  function draw(svg, { game, boardSize, board, orientation = "flat" }) {
    if (game === "hex") {
      const boardSvg = window.HexMoveTree.createBoardSvg(svg)
      boardSvg.setBoardOrientation(orientation)
      window.HexMoveTree.renderMoveTreeBoard({
        boardSvg, boardSize, boardState: board, enableCellClicks: false, showMoveNumbers: false,
      })
      return
    }
    svg.replaceChildren()
    const tools = createSvgTools({ board: svg, hexSize: 24,
      defaultFill: rgbText(THEME.OFF_WHITE_RGB), defaultStroke: THEME.GRID_EDGE, defaultStrokeWidth: 1 })
    tools.setBoardOrientation(orientation)
    const cells = []
    for (let row = 1; row <= boardSize; row++) {
      for (let col = 1; col <= boardSize + 1 - row; col++) {
        cells.push({ col, row })
        const stone = board.occupied.get(pointKey(col, row))
        const hex = tools.appendHex(col, row, { fill: stone
          ? rgbText(stone.color === "red" ? THEME.RED_RGB : THEME.BLUE_RGB) : rgbText(THEME.OFF_WHITE_RGB) })
        if (stone?.isLast) tools.appendCircle(hex.cx, hex.cy, 3, { fill: rgbText(THEME.OFF_WHITE_RGB) })
      }
    }
    tools.setViewBoxFromPoints({ boardPoints: cells, coordPoints: [], viewPadding: 26, coordViewPadding: 0 })
  }

  function place(card, x, y) {
    card.hidden = false
    const rect = card.getBoundingClientRect()
    const left = Math.max(8, Math.min(x + 14, window.innerWidth - rect.width - 8))
    const top = y - rect.height - 12 >= 8
      ? y - rect.height - 12 : Math.max(8, Math.min(y + 16, window.innerHeight - rect.height - 8))
    card.style.left = `${left}px`
    card.style.top = `${top}px`
  }

  window.BoardGraphPreview = { draw, place }
})()
