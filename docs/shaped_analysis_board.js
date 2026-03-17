(() => {
  function create({ game, defaultBoardSize, minBoardSize, maxBoardSize, boardSizes = null, model }) {
    const HEX_SIZE = 24
    const VIEW_PADDING = 34
    const COORD_VIEW_PADDING = 8

    const BOARD_SIDE = "rgb(82, 82, 82)"

    const {
      analysisOverlayFill,
      createSvgTools,
      rgbText,
      stoneTextColor,
      THEME: { BLUE_RGB, GRID_EDGE, OFF_WHITE_RGB, RED_RGB },
      turnRgbaText,
    } = window.HexStudyUI
    const {
      alphaLabel,
      formatCell,
      materializeBoardState: materializeFullBoardState,
      pointKey,
      tryParseCell,
    } = globalThis.HexPosition
    const { applyBranchOutline, branchChildrenByKey, collectBoardEditorElements, createBranchingBoardEditor } =
      window.HexBoardEditor

    const elements = collectBoardEditorElements(`${game}-status`)

    const boardSvg = createSvgTools({
      board: elements.board,
      hexSize: HEX_SIZE,
      defaultFill: rgbText(OFF_WHITE_RGB),
      defaultStroke: GRID_EDGE,
      defaultStrokeWidth: "0.85",
    })
    let analysis = null
    let viewBoxBoardSize = null

    function isLegalCell(col, row, size) {
      return (
        Number.isInteger(col) &&
        Number.isInteger(row) &&
        globalThis.BoardGeometry.isPlayable(game, col - 1, row - 1, size)
      )
    }

    function materializeBoardState(moves, boardSize) {
      if (game === "yavalath" && moves.includes("pass")) { throw new Error("Yavalath does not allow pass") }
      const board = materializeFullBoardState(moves, boardSize, {
        isLegalCell,
        swapStone: (stone) => ({ ...stone, color: "blue", ply: "S" }),
      })
      if (game === "yavalath") {
        const cells = new Uint8Array(boardSize * boardSize).fill(3)
        for (const point of globalThis.BoardGeometry.playablePoints(game, boardSize)) cells[point] = 0
        for (let ply = 0; ply < moves.length; ply++) {
          if (
            globalThis.YavalathRules.terminalValue({ board: cells, size: boardSize, toPlay: 1 + (ply % 2) }) !== null
          ) { throw new Error("Cannot play after the game is over") }
          const move = moves[ply]
          const { col, row } = tryParseCell(move === "swap" ? moves[0] : move)
          cells[(row - 1) * boardSize + col - 1] = move === "swap" ? 2 : 1 + (ply % 2)
        }
      }
      return board
    }

    function cellsForBoardSize(size) {
      return globalThis.BoardGeometry.playablePoints(game, size).map((point) => ({
        col: (point % size) + 1,
        row: Math.floor(point / size) + 1,
      }))
    }

    function coordinatePoints(size) {
      const points = []
      for (let col = 1; col <= size; col++) {
        let row = 1
        while (!isLegalCell(col, row, size)) row++
        points.push({
          col,
          row: game === "yavalath" && col <= (size - 1) / 2 ? size + 1 : row - 1,
          label: alphaLabel(col),
        })
      }
      for (let row = 1; row <= size; row++) {
        let col = 1
        while (!isLegalCell(col, row, size)) col++
        points.push({ col: col - 1, row, label: String(row) })
      }
      return points
    }

    function setupViewBox(boardSize, boardPoints) {
      if (viewBoxBoardSize === boardSize) return
      viewBoxBoardSize = boardSize
      const coordPoints = coordinatePoints(boardSize)
      boardSvg.setViewBoxFromPoints({
        boardPoints,
        coordPoints,
        viewPadding: VIEW_PADDING,
        coordViewPadding: COORD_VIEW_PADDING,
      })
    }

    function drawBorderSegment(col, row, c1, c2) {
      const [cx, cy] = boardSvg.pointToPixel(col, row)
      const a = boardSvg.hexCorner(cx, cy, HEX_SIZE - 1.5, c1)
      const b = boardSvg.hexCorner(cx, cy, HEX_SIZE - 1.5, c2)
      boardSvg.appendLine(a[0], a[1], b[0], b[1], BOARD_SIDE, 4)
    }

    function renderFrame(boardSize) {
      const neighbors = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]]
      for (const { col, row } of cellsForBoardSize(boardSize)) {
        neighbors.forEach(([dx, dy], edge) => {
          if (!isLegalCell(col + dx, row + dy, boardSize)) drawBorderSegment(col, row, edge, (edge + 1) % 6)
        })
      }
      for (const { col, row, label } of coordinatePoints(boardSize)) {
        const [cx, cy] = boardSvg.pointToPixel(col, row)
        boardSvg.appendText(cx, cy, label, "coord-text")
      }
    }

    function renderBoard({ analysisByKey, board, cells, branchChildren, display, position }) {
      boardSvg.beginFrame()
      if (boardSvg.setBoardOrientation(display.boardOrientation)) viewBoxBoardSize = null
      const overlayMode = analysis?.overlayMode() || "winrate"
      boardSvg.clear()
      setupViewBox(position.boardSize, cells)
      const dragActive = Boolean(display.drag)
      elements.board.classList.toggle("dragging", dragActive)
      const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
      const hoverFill = turnRgbaText(board.toPlay, 0.12)
      const dragSourceKey = dragActive ? pointKey(display.drag.startPoint.col, display.drag.startPoint.row) : null
      const branchByKey = branchChildrenByKey(branchChildren, board.occupied)

      for (const cell of cells) {
        const key = pointKey(cell.col, cell.row)
        const stone = board.occupied.get(key) || null
        const branchChild = branchByKey.get(key) || null
        const candidate = analysisByKey.get(key) || null
        const overlay = overlayMode === "prior" && !candidate?.visits ? null : candidate
        let fill = rgbText(OFF_WHITE_RGB)
        let stroke = GRID_EDGE
        const classes = ["board-hex", "board-hex-face"]
        let tappable = true

        if (stone) {
          fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
          stroke = "none"
          tappable = Boolean(stone.isLast)
        } else if (overlay) { fill = analysisOverlayFill(overlay, { mode: overlayMode }) || fill }
        if (dragSourceKey === key) { classes.push("drag-source") }

        const hitClasses = ["board-hover-hit"]
        if (tappable) { hitClasses.push("clickable") }
        if (stone) { hitClasses.push("board-drag-hit") }
        if (!stone) { hitClasses.push("hoverable") }
        const hoverHex = boardSvg.appendHex(cell.col, cell.row, {
          fill: "transparent",
          stroke: "none",
          className: hitClasses.join(" "),
          size: HEX_SIZE,
          title: formatCell(cell.col, cell.row),
          boardPoint: true,
        })
        hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
        const hex = boardSvg.appendHex(cell.col, cell.row, { fill, stroke, className: classes.join(" ") })
        if (branchChild && !stone) { applyBranchOutline(hex.polygon, branchChild, board.toPlay) } else { hex.polygon.style.setProperty("--hover-outline", hoverColor) }
        if (display.showCoords) {
          boardSvg.appendText(
            hex.cx,
            hex.cy,
            formatCell(cell.col, cell.row),
            "cell-text",
            stone ? stoneTextColor(stone) : null,
          )
        } else if (overlay && !stone) {
          const { primary, secondary } = window.BoardAnalysis.overlayText(overlay, overlayMode)
          if (primary && secondary) boardSvg.appendStackedText(hex.cx, hex.cy, primary, secondary)
          else if (primary || secondary) boardSvg.appendText(hex.cx, hex.cy, primary || secondary)
        } else if (stone && display.showMoveNumbers) {
          boardSvg.appendText(hex.cx, hex.cy, stone.ply, "cell-text", stoneTextColor(stone))
        } else if (stone?.isLast) {
          boardSvg.appendCircle(hex.cx, hex.cy, Math.max(2, HEX_SIZE * 0.14), {
            className: "last-move-dot",
            fill: rgbText(OFF_WHITE_RGB),
          })
        }
      }
      if (dragActive && display.drag.targetPoint) {
        const ghostFill = display.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        boardSvg.appendHex(display.drag.targetPoint.col, display.drag.targetPoint.row, {
          fill: ghostFill,
          stroke: "none",
          className: "board-ghost",
        })
      }
      renderFrame(position.boardSize)
      if (!dragActive) analysis?.renderPv(boardSvg, board.toPlay, display)
      boardSvg.endFrame()
    }

    const editor = createBranchingBoardEditor({
      game,
      elements,
      defaultBoardSize,
      minBoardSize,
      maxBoardSize,
      boardSizes,
      ...(game === "yavalath"
        ? { boardSizeToDisplay: (size) => (size + 1) / 2, boardSizeFromDisplay: (size) => 2 * size - 1 }
        : {}),
      allowPass: game !== "yavalath",
      isLegalCell,
      materializeBoardState,
      getBoardCells: cellsForBoardSize,
      renderBoard,
      getBoardDecorations: () => analysis?.decorations() || {},
      shortcutHelpGroups: window.BoardAnalysis.shortcutHelpGroups(game),
    })
    analysis = window.BoardAnalysis.create(editor, model)
    return editor
  }
  window.ShapedAnalysisBoard = { create }
})()
