(() => {
  const HEX_SIZE = 24
  const VIEW_PADDING = 34
  const COORD_VIEW_PADDING = 8

  const Y_SIDE = "rgb(82, 82, 82)"
  const MIN_BOARD_SIZE = 1
  const MAX_BOARD_SIZE = 42
  const DEFAULT_BOARD_SIZE = 15

  const {
    analysisOverlayFill,
    createSvgTools,
    rgbText,
    stoneTextColor,
    THEME: {
      BLUE_RGB,
      GRID_EDGE,
      OFF_WHITE_RGB,
      RED_RGB,
    },
    turnRgbaText,
  } = window.HexStudyUI
  const {
    alphaLabel,
    formatCell,
    materializeBoardState: materializeFullBoardState,
    pointKey,
  } = globalThis.HexPosition
  const {
    applyBranchOutline,
    branchChildrenByKey,
    collectBoardEditorElements,
    createBranchingBoardEditor,
  } = window.HexBoardEditor

  const elements = collectBoardEditorElements("y-status")

  const boardSvg = createSvgTools({
    board: elements.board,
    hexSize: HEX_SIZE,
    defaultFill: rgbText(OFF_WHITE_RGB),
    defaultStroke: GRID_EDGE,
    defaultStrokeWidth: "0.85",
  })

  function isLegalCell(col, row, boardSize) {
    const size = Number(boardSize)
    return (
      Number.isInteger(col)
      && Number.isInteger(row)
      && col >= 1
      && row >= 1
      && col <= size
      && row <= size
      && col + row <= size + 1
    )
  }

  function materializeBoardState(moves, boardSize) {
    return materializeFullBoardState(moves, boardSize, {
      isLegalCell,
      swapStone: (firstStone) => ({
        ...firstStone,
        color: "blue",
        ply: "S",
      }),
    })
  }

  function cellsForBoardSize(boardSize) {
    const cells = []
    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize + 1 - row; col += 1) {
        cells.push({ col, row })
      }
    }
    return cells
  }

  function setupViewBox(boardSize, boardPoints) {
    const coordPoints = []
    for (let col = 1; col <= boardSize; col += 1) {
      coordPoints.push({ col, row: 0 })
    }
    for (let row = 1; row <= boardSize; row += 1) {
      coordPoints.push({ col: 0, row })
    }
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
    boardSvg.appendLine(a[0], a[1], b[0], b[1], Y_SIDE, 4)
  }

  function renderFrame(boardSize) {
    for (let col = 1; col <= boardSize; col += 1) {
      drawBorderSegment(col, 1, 4, 5)
      drawBorderSegment(col, 1, 5, 0)
    }
    for (let row = 1; row <= boardSize; row += 1) {
      drawBorderSegment(1, row, 2, 3)
      drawBorderSegment(1, row, 3, 4)
    }
    for (let row = 1; row <= boardSize; row += 1) {
      const col = boardSize + 1 - row
      drawBorderSegment(col, row, 0, 1)
      drawBorderSegment(col, row, 1, 2)
    }
    for (let col = 1; col <= boardSize; col += 1) {
      const [cx, cy] = boardSvg.pointToPixel(col, 0)
      boardSvg.appendText(cx, cy, alphaLabel(col), "coord-text")
    }
    for (let row = 1; row <= boardSize; row += 1) {
      const [cx, cy] = boardSvg.pointToPixel(0, row)
      boardSvg.appendText(cx, cy, String(row), "coord-text")
    }
  }

  function renderBoard({
    analysisByKey,
    board,
    cells,
    branchChildren,
    display,
    position,
  }) {
    boardSvg.setBoardOrientation(display.boardOrientation)
    boardSvg.clear()
    setupViewBox(position.boardSize, cells)
    const dragActive = Boolean(display.drag)
    elements.board.classList.toggle("dragging", dragActive)
    const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = turnRgbaText(board.toPlay, 0.12)
    const dragSourceKey = dragActive
      ? pointKey(display.drag.startPoint.col, display.drag.startPoint.row)
      : null
    const branchByKey = branchChildrenByKey(branchChildren, board.occupied)

    for (const cell of cells) {
      const key = pointKey(cell.col, cell.row)
      const stone = board.occupied.get(key) || null
      const branchChild = branchByKey.get(key) || null
      const analysis = analysisByKey.get(key) || null
      let fill = rgbText(OFF_WHITE_RGB)
      let stroke = GRID_EDGE
      const classes = ["board-hex", "board-hex-face"]
      let tappable = true

      if (stone) {
        fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        stroke = "none"
        tappable = Boolean(stone.isLast)
      } else if (analysis) {
        fill = analysisOverlayFill(analysis) || fill
      }
      if (dragSourceKey === key) {
        classes.push("drag-source")
      }

      const hitClasses = ["board-hover-hit"]
      if (tappable) {
        hitClasses.push("clickable")
      }
      if (stone) {
        hitClasses.push("board-drag-hit")
      }
      if (!stone) {
        hitClasses.push("hoverable")
      }
      const hoverHex = boardSvg.appendHex(cell.col, cell.row, {
        fill: "transparent",
        stroke: "none",
        className: hitClasses.join(" "),
        size: HEX_SIZE,
        title: formatCell(cell.col, cell.row),
        boardPoint: true,
      })
      hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
      const hex = boardSvg.appendHex(cell.col, cell.row, {
        fill,
        stroke,
        className: classes.join(" "),
      })
      if (branchChild && !stone) {
        applyBranchOutline(hex.polygon, branchChild, board.toPlay)
      } else {
        hex.polygon.style.setProperty("--hover-outline", hoverColor)
      }
      if (display.showCoords) {
        boardSvg.appendText(hex.cx, hex.cy, formatCell(cell.col, cell.row), "cell-text", stone ? stoneTextColor(stone) : null)
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
        className: "board-ghost board-ghost-target",
      })
    }
    renderFrame(position.boardSize)
  }

  createBranchingBoardEditor({
    game: "y",
    elements,
    defaultBoardSize: DEFAULT_BOARD_SIZE,
    minBoardSize: MIN_BOARD_SIZE,
    maxBoardSize: MAX_BOARD_SIZE,
    isLegalCell,
    materializeBoardState,
    getBoardCells: cellsForBoardSize,
    renderBoard,
  })
})()
