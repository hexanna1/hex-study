(() => {
  const game = "bentY"
  const RADIUS = 26
  const { topology, tensorLenForWidth, widthForTensorLen, winner } = globalThis.BentYGeometry
  const {
    formatCell,
    pointKey,
    tryParseCell,
    materializeBoardState: materializeFullBoardState,
  } = globalThis.HexPosition
  const {
    createSvgTools,
    analysisOverlayFill,
    rgbText,
    turnRgbaText,
    stoneTextColor,
    THEME: { OFF_WHITE_RGB, RED_RGB, BLUE_RGB },
  } = window.HexStudyUI
  const { collectBoardEditorElements, createBranchingBoardEditor, branchChildrenByKey, applyBranchOutline } =
    window.HexBoardEditor
  const elements = collectBoardEditorElements("bentY-status")
  let analysis = null
  let boardSvg = null
  let drawingSize = null

  function isLegalCell(col, row, size) {
    return (
      Number.isInteger(col) &&
      Number.isInteger(row) &&
      globalThis.BoardGeometry.isPlayable(game, col - 1, row - 1, size)
    )
  }

  function cellsForBoardSize(size) {
    return topology(size).playable.map((point) => ({ col: (point % size) + 1, row: Math.floor(point / size) + 1 }))
  }

  function materializeBoardState(moves, size) {
    const board = materializeFullBoardState(moves, size, {
      isLegalCell,
      swapStone: (stone) => ({ ...stone, color: "blue", ply: "S" }),
    })
    const cells = new Uint8Array(size * size).fill(3)
    for (const point of topology(size).playable) cells[point] = 0
    for (let ply = 0; ply < moves.length; ply++) {
      if (winner(cells, size)) throw new Error("Cannot play after the game is over")
      const move = moves[ply]
      if (move === "pass") continue
      const { col, row } = tryParseCell(move === "swap" ? moves[0] : move)
      cells[(row - 1) * size + col - 1] = move === "swap" ? 2 : 1 + (ply % 2)
    }
    return board
  }

  function createBoardSvg(size) {
    const scale = widthForTensorLen(size) * 63
    return createSvgTools({
      board: elements.board,
      hexSize: 24,
      defaultFill: rgbText(OFF_WHITE_RGB),
      defaultStroke: "#75664f",
      defaultStrokeWidth: "0.9",
      projectPoint: (_hexSize, col, row, orientation) =>
        globalThis.BentYGeometry.projectPoint(size, col, row, orientation).map((value) => value * scale),
    })
  }

  function renderBoard({ analysisByKey, board, cells, branchChildren, display, position }) {
    if (drawingSize !== position.boardSize) {
      drawingSize = position.boardSize
      boardSvg = createBoardSvg(drawingSize)
    }
    boardSvg.setBoardOrientation(display.boardOrientation)
    boardSvg.beginFrame()
    boardSvg.setViewBoxFromPoints({ boardPoints: cells, coordPoints: [], viewPadding: 38, coordViewPadding: 0 })
    const geometry = topology(drawingSize)
    const pixel = (point) => boardSvg.pointToPixel((point % drawingSize) + 1, Math.floor(point / drawingSize) + 1)
    for (const [a, b] of geometry.edges) {
      const boundary = geometry.sideMasks[a] & geometry.sideMasks[b]
      boardSvg.appendLine(...pixel(a), ...pixel(b), boundary ? "#5c5348" : "#9b9080", boundary ? 3.2 : 1.8)
    }
    const overlayMode = analysis?.overlayMode() || "winrate"
    const dragActive = Boolean(display.drag)
    elements.board.classList.toggle("dragging", dragActive)
    const hoverColor = rgbText(board.toPlay === "red" ? RED_RGB : BLUE_RGB)
    const hoverFill = turnRgbaText(board.toPlay, 0.12)
    const dragSource = dragActive ? pointKey(display.drag.startPoint.col, display.drag.startPoint.row) : null
    const branchByKey = branchChildrenByKey(branchChildren, board.occupied)
    for (const { col, row } of cells) {
      const key = pointKey(col, row)
      const stone = board.occupied.get(key) || null
      const branchChild = branchByKey.get(key) || null
      const candidate = analysisByKey.get(key) || null
      const overlay = overlayMode === "prior" && !candidate?.visits ? null : candidate
      const [cx, cy] = boardSvg.pointToPixel(col, row)
      const fill = stone
        ? rgbText(stone.color === "red" ? RED_RGB : BLUE_RGB)
        : overlay
          ? analysisOverlayFill(overlay, { mode: overlayMode }) || rgbText(OFF_WHITE_RGB)
          : rgbText(OFF_WHITE_RGB)
      const classes = ["board-hex", "board-hex-face"]
      if (dragSource === key) classes.push("drag-source")
      const hitClasses = ["board-hover-hit"]
      if (!stone || stone.isLast) hitClasses.push("clickable")
      hitClasses.push(stone ? "board-drag-hit" : "hoverable")
      const hit = boardSvg.appendCircle(cx, cy, RADIUS + 3, {
        className: hitClasses.join(" "),
        fill: "transparent",
        boardPoint: true,
        col,
        row,
        title: formatCell(col, row),
      })
      hit.style.setProperty("--hover-fill", hoverFill)
      const face = boardSvg.appendCircle(cx, cy, RADIUS, {
        fill,
        stroke: stone ? "none" : "#75664f",
        strokeWidth: 1.7,
        className: classes.join(" "),
      })
      if (branchChild && !stone) applyBranchOutline(face, branchChild, board.toPlay)
      else face.style.setProperty("--hover-outline", hoverColor)
      if (display.showCoords) {
        boardSvg.appendText(cx, cy, formatCell(col, row), "cell-text", stone ? stoneTextColor(stone) : null)
      } else if (overlay && !stone) {
        const { primary, secondary } = window.BoardAnalysis.overlayText(overlay, overlayMode)
        if (primary && secondary) boardSvg.appendStackedText(cx, cy, primary, secondary)
        else if (primary || secondary) boardSvg.appendText(cx, cy, primary || secondary)
      } else if (stone && display.showMoveNumbers) {
        boardSvg.appendText(cx, cy, stone.ply, "cell-text", stoneTextColor(stone))
      } else if (stone?.isLast) {
        boardSvg.appendCircle(cx, cy, 3.4, { className: "last-move-dot", fill: rgbText(OFF_WHITE_RGB) })
      }
    }
    if (dragActive && display.drag.targetPoint) {
      const { col, row } = display.drag.targetPoint
      const [cx, cy] = boardSvg.pointToPixel(col, row)
      boardSvg.appendCircle(cx, cy, RADIUS, {
        fill: rgbText(display.drag.sourceColor === "red" ? RED_RGB : BLUE_RGB),
        stroke: "none",
        className: "board-ghost",
      })
    }
    if (!dragActive)
      analysis?.renderPv(
        {
          ...boardSvg,
          appendHex: (col, row, options) => {
            const [cx, cy] = boardSvg.pointToPixel(col, row)
            const polygon = boardSvg.appendCircle(cx, cy, RADIUS, options)
            return { cx, cy, polygon }
          },
        },
        board.toPlay,
        display,
      )
    boardSvg.endFrame()
  }

  const editor = createBranchingBoardEditor({
    game,
    elements,
    defaultBoardSize: 7,
    minBoardSize: 7,
    maxBoardSize: 13,
    boardSizes: [7, 13],
    boardSizeToDisplay: widthForTensorLen,
    boardSizeFromDisplay: tensorLenForWidth,
    isLegalCell,
    materializeBoardState,
    getBoardCells: cellsForBoardSize,
    renderBoard,
    getBoardDecorations: () => analysis?.decorations() || {},
    shortcutHelpGroups: window.BoardAnalysis.shortcutHelpGroups(game),
  })
  analysis = window.BoardAnalysis.create(editor, "t4nbt-benty-9")
})()
