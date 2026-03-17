(() => {
  const HEX_SIZE = 24
  const VIEW_PADDING = 34
  const COORD_VIEW_PADDING = 8

  const Y_SIDE = "rgb(82, 82, 82)"
  const MIN_BOARD_SIZE = 1
  const MAX_BOARD_SIZE = 42
  const DEFAULT_BOARD_SIZE = 15

  const {
    createSvgTools,
    rgbText,
    turnRgbaText,
  } = window.HexStudyUI
  const {
    GRID_EDGE,
    THEME: {
      BLUE_RGB,
      OFF_WHITE_RGB,
      RED_RGB,
    },
    alphaLabel,
    formatCell,
    materializeBoardState: materializeFullBoardState,
    pointKey,
    swapControlPoint,
  } = window.HexMoveTree
  const {
    createBranchingBoardEditor,
    renderSwapSideAction,
  } = window.HexBoardEditor

  const elements = {
    board: document.getElementById("board"),
    currentLine: document.getElementById("current-line"),
    lineStatus: document.getElementById("line-status"),
    moveNav: document.getElementById("move-nav"),
    moveFirstBtn: document.getElementById("move-first-btn"),
    moveLastBtn: document.getElementById("move-last-btn"),
    moveList: document.getElementById("move-list"),
    moveNextBtn: document.getElementById("move-next-btn"),
    movePrevBtn: document.getElementById("move-prev-btn"),
    resetBtn: document.getElementById("reset-btn"),
    sizeInput: document.getElementById("size-input"),
    sizeNextBtn: document.getElementById("size-next-btn"),
    sizePrevBtn: document.getElementById("size-prev-btn"),
    sizeStepper: document.getElementById("size-stepper"),
    status: document.getElementById("y-status"),
  }

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

  function boardCells(boardSize) {
    const cells = []
    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize + 1 - row; col += 1) {
        cells.push({ col, row })
      }
    }
    return cells
  }

  function setupViewBox(state) {
    const boardPixels = []
    for (const cell of boardCells(state.boardSize)) {
      boardPixels.push(boardSvg.pointToPixel(cell.col, cell.row))
    }
    const coordPixels = []
    for (let col = 1; col <= state.boardSize; col += 1) {
      coordPixels.push(boardSvg.pointToPixel(col, 0))
    }
    for (let row = 1; row <= state.boardSize; row += 1) {
      coordPixels.push(boardSvg.pointToPixel(0, row))
    }
    const swapPoint = swapControlPoint(state.boardSize)
    const extraPixels = [boardSvg.pointToPixel(swapPoint.col, swapPoint.row)]

    const boardXs = boardPixels.map((point) => point[0])
    const boardYs = boardPixels.map((point) => point[1])
    const coordXs = coordPixels.map((point) => point[0])
    const coordYs = coordPixels.map((point) => point[1])
    const extraXs = extraPixels.map((point) => point[0])
    const extraYs = extraPixels.map((point) => point[1])

    const primaryMinX = Math.min(Math.min(...boardXs) - VIEW_PADDING, Math.min(...coordXs) - COORD_VIEW_PADDING)
    const primaryMaxX = Math.max(Math.max(...boardXs) + VIEW_PADDING, Math.max(...coordXs) + COORD_VIEW_PADDING)
    const primaryMinY = Math.min(Math.min(...boardYs) - VIEW_PADDING, Math.min(...coordYs) - COORD_VIEW_PADDING)
    const primaryMaxY = Math.max(Math.max(...boardYs) + VIEW_PADDING, Math.max(...coordYs) + COORD_VIEW_PADDING)
    const extraMinX = Math.min(...extraXs) - VIEW_PADDING
    const extraMaxX = Math.max(...extraXs) + VIEW_PADDING
    const extraMinY = Math.min(...extraYs) - VIEW_PADDING
    const extraMaxY = Math.max(...extraYs) + VIEW_PADDING

    const centerX = (primaryMinX + primaryMaxX) / 2
    const halfWidth = Math.max(
      (primaryMaxX - primaryMinX) / 2,
      centerX - extraMinX,
      extraMaxX - centerX,
    )
    const minX = centerX - halfWidth
    const maxX = centerX + halfWidth
    const minY = Math.min(primaryMinY, extraMinY)
    const maxY = Math.max(primaryMaxY, extraMaxY)
    elements.board.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`)
  }

  function drawBorderSegment(col, row, c1, c2) {
    const [cx, cy] = boardSvg.pointToPixel(col, row)
    const a = boardSvg.hexCorner(cx, cy, HEX_SIZE - 1.5, c1)
    const b = boardSvg.hexCorner(cx, cy, HEX_SIZE - 1.5, c2)
    boardSvg.appendLine(a[0], a[1], b[0], b[1], Y_SIDE, 4)
  }

  function renderFrame(state) {
    for (let col = 1; col <= state.boardSize; col += 1) {
      drawBorderSegment(col, 1, 4, 5)
      drawBorderSegment(col, 1, 5, 0)
    }
    for (let row = 1; row <= state.boardSize; row += 1) {
      drawBorderSegment(1, row, 2, 3)
      drawBorderSegment(1, row, 3, 4)
    }
    for (let row = 1; row <= state.boardSize; row += 1) {
      const col = state.boardSize + 1 - row
      drawBorderSegment(col, row, 0, 1)
      drawBorderSegment(col, row, 1, 2)
    }
    for (let col = 1; col <= state.boardSize; col += 1) {
      const [cx, cy] = boardSvg.pointToPixel(col, 0)
      boardSvg.appendText(cx, cy, alphaLabel(col), "coord-text")
    }
    for (let row = 1; row <= state.boardSize; row += 1) {
      const [cx, cy] = boardSvg.pointToPixel(0, row)
      boardSvg.appendText(cx, cy, String(row), "coord-text")
    }
  }

  function renderBoard({
    state,
    currentMoves,
    materializeBoardState: materializeCurrentBoardState,
    boardCells: currentBoardCells,
    playMove,
    goPrevious,
    canSwap,
    swapMove,
  }) {
    boardSvg.clear()
    setupViewBox(state)
    const board = materializeCurrentBoardState(currentMoves())
    const dragActive = Boolean(state.drag)
    elements.board.classList.toggle("dragging", dragActive)
    const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = turnRgbaText(board.toPlay, 0.12)
    const dragSourceKey = dragActive
      ? pointKey(state.drag.startPoint.col, state.drag.startPoint.row)
      : null

    for (const cell of currentBoardCells()) {
      const key = pointKey(cell.col, cell.row)
      const stone = board.occupied.get(key) || null
      let fill = rgbText(OFF_WHITE_RGB)
      let stroke = GRID_EDGE
      const classes = ["board-hex", "board-hex-face"]
      let onClick = () => {
        playMove(formatCell(cell.col, cell.row))
      }

      if (stone) {
        fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        stroke = "none"
        onClick = stone.isLast ? goPrevious : null
      }
      if (dragSourceKey === key) {
        classes.push("drag-source")
      }

      const hitClasses = ["board-hover-hit"]
      if (onClick) {
        hitClasses.push("clickable")
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
        onClick,
      })
      hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
      const hex = boardSvg.appendHex(cell.col, cell.row, {
        fill,
        stroke,
        className: classes.join(" "),
      })
      hex.polygon.style.setProperty("--hover-outline", hoverColor)
      if (state.showCoords) {
        boardSvg.appendText(hex.cx, hex.cy, formatCell(cell.col, cell.row), "cell-text", stone?.textColor || null)
      } else if (stone) {
        boardSvg.appendText(hex.cx, hex.cy, stone.ply, "cell-text", stone.textColor)
      }
    }
    if (dragActive && state.drag.targetPoint) {
      const ghostFill = state.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
      boardSvg.appendHex(state.drag.targetPoint.col, state.drag.targetPoint.row, {
        fill: ghostFill,
        stroke: "none",
        className: "board-ghost board-ghost-target",
      })
    }
    renderFrame(state)
    renderSwapSideAction({ boardSvg, state, board, canSwap, swapMove })
  }

  createBranchingBoardEditor({
    elements,
    defaultBoardSize: DEFAULT_BOARD_SIZE,
    minBoardSize: MIN_BOARD_SIZE,
    maxBoardSize: MAX_BOARD_SIZE,
    isLegalCell,
    materializeBoardState,
    boardCells,
    renderBoard,
  })
})()
