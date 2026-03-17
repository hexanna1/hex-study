(() => {
  const HEX_SIZE = 24
  const VIEW_PADDING = 34
  const COORD_VIEW_PADDING = 8

  const Y_SIDE = "rgb(82, 82, 82)"
  const MIN_BOARD_SIZE = 1
  const MAX_BOARD_SIZE = 31

  const {
    copyTextToClipboard,
    createSvgTools,
    handleStandardKeydown,
    installHoldButton,
    navButtonDisabled,
    replaceHash,
    rgbText,
    setNavButtonDisabled,
    setTurnStatus,
    shouldIgnoreGlobalKeydown,
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
    compactCursorHash,
    compactCursorText,
    formatCell,
    formatLine,
    materializeBoardState: materializeFullBoardState,
    parseCompactCursorHash,
    parseMoves,
    pointKey,
    renderLineMoveList,
    renderSideActionHex,
    swapControlPoint,
    tryParseCell,
  } = window.HexMoveTree

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

  const state = {
    boardSize: 15,
    moves: [],
    cursor: 0,
  }

  function isLegalCell(col, row, boardSize = state.boardSize) {
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

  function isSupportedBoardSize(boardSize) {
    const size = Number(boardSize)
    return Number.isInteger(size) && size >= MIN_BOARD_SIZE && size <= MAX_BOARD_SIZE
  }

  function parseBoardSize(value) {
    if (String(value || "").trim() === "") {
      return null
    }
    const size = Number(value)
    if (!Number.isFinite(size)) {
      return null
    }
    return Math.max(MIN_BOARD_SIZE, Math.min(MAX_BOARD_SIZE, Math.trunc(size)))
  }

  function currentLineText() {
    return compactCursorText({
      boardSize: state.boardSize,
      line: formatLine(currentMoves()),
      defaultBoardSize: state.boardSize,
      keepEmptyLineComma: true,
    })
  }

  function normalizeYLine(line, boardSize) {
    const raw = String(line || "").trim()
    const moves = parseMoves(raw)
    if (raw && moves.length === 0) {
      return ""
    }
    try {
      materializeBoardState(moves, boardSize)
    } catch (_error) {
      return ""
    }
    return formatLine(moves)
  }

  function parseLine(text) {
    const parsed = parseCompactCursorHash(text, {
      defaultBoardSize: state.boardSize,
      isBoardSizeSupported: isSupportedBoardSize,
      normalizeLineFn: normalizeYLine,
    })
    if (!parsed.valid) {
      return null
    }
    return {
      boardSize: parsed.boardSize,
      moves: parseMoves(parsed.fullLine),
      cursor: parsed.cursor,
    }
  }

  function currentMoves() {
    return state.moves.slice(0, state.cursor)
  }

  function futureTailLines() {
    return state.moves.slice(state.cursor).map((_, index) => (
      formatLine(state.moves.slice(0, state.cursor + index + 1))
    ))
  }

  function materializeBoardState(moves, boardSize = state.boardSize) {
    return materializeFullBoardState(moves, boardSize, {
      isLegalCell,
      swapStone: (firstStone) => ({
        ...firstStone,
        color: "blue",
        ply: "S",
      }),
    })
  }

  function boardCells(boardSize = state.boardSize) {
    const cells = []
    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize + 1 - row; col += 1) {
        cells.push({ col, row })
      }
    }
    return cells
  }

  function setupViewBox() {
    const boardPixels = []
    for (const cell of boardCells()) {
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

  function renderFrame() {
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

  function setHash() {
    const nextHash = compactCursorHash({
      boardSize: state.boardSize,
      line: formatLine(currentMoves()),
      futureLines: futureTailLines(),
      defaultBoardSize: state.boardSize,
      keepEmptyLineComma: true,
    })
    replaceHash(nextHash)
  }

  function syncSizeInput() {
    elements.sizeInput.value = String(state.boardSize)
    setNavButtonDisabled(elements.sizePrevBtn, state.boardSize <= MIN_BOARD_SIZE)
    setNavButtonDisabled(elements.sizeNextBtn, state.boardSize >= MAX_BOARD_SIZE)
  }

  function syncNavigationButtons() {
    setNavButtonDisabled(elements.moveFirstBtn, state.cursor <= 0)
    setNavButtonDisabled(elements.movePrevBtn, state.cursor <= 0)
    setNavButtonDisabled(elements.moveNextBtn, state.cursor >= state.moves.length)
    setNavButtonDisabled(elements.moveLastBtn, state.cursor >= state.moves.length)
  }

  function renderMoveLine() {
    renderLineMoveList({
      container: elements.moveList,
      currentLine: formatLine(currentMoves()),
      futureTailLines,
      setCursorLine: (line) => {
        const lineMoves = parseMoves(line)
        state.cursor = Math.max(0, Math.min(state.moves.length, lineMoves.length))
        sync()
      },
    })
  }

  function renderBoard() {
    boardSvg.clear()
    setupViewBox()
    const board = materializeBoardState(currentMoves())
    const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = turnRgbaText(board.toPlay, 0.12)

    for (const cell of boardCells()) {
      const key = pointKey(cell.col, cell.row)
      const stone = board.occupied.get(key) || null
      let fill = rgbText(OFF_WHITE_RGB)
      let stroke = GRID_EDGE
      let onClick = () => {
        playMove(formatCell(cell.col, cell.row))
      }

      if (stone) {
        fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        stroke = "none"
        onClick = stone.isLast ? goPrevious : null
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
        onClick,
      })
      hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
      const hex = boardSvg.appendHex(cell.col, cell.row, {
        fill,
        stroke,
        className: "board-hex board-hex-face",
      })
      hex.polygon.style.setProperty("--hover-outline", hoverColor)
      if (stone) {
        boardSvg.appendText(hex.cx, hex.cy, stone.ply, "cell-text", stone.textColor)
      }
    }
    renderFrame()
    renderSwapHex(board)
  }

  function renderSwapHex(board) {
    if (!canSwap()) {
      return
    }
    renderSideActionHex({
      boardSvg,
      point: swapControlPoint(state.boardSize),
      toPlay: board.toPlay,
      labelText: "Swap",
      title: "Swap",
      onClick: swapMove,
      primaryText: "S",
    })
  }

  function sync({ rewriteHash = true, message = "" } = {}) {
    const board = materializeBoardState(currentMoves())
    setTurnStatus(elements.status, board.toPlay)
    elements.currentLine.textContent = currentLineText()
    elements.lineStatus.textContent = message
    syncSizeInput()
    syncNavigationButtons()
    renderMoveLine()
    renderBoard()
    if (rewriteHash) {
      setHash()
    }
  }

  function truncateFuture() {
    if (state.cursor < state.moves.length) {
      state.moves = state.moves.slice(0, state.cursor)
    }
  }

  function playMove(move) {
    truncateFuture()
    const nextMoves = [...state.moves, formatLine([move])]
    try {
      materializeBoardState(nextMoves)
    } catch (_error) {
      return
    }
    state.moves = nextMoves
    state.cursor = state.moves.length
    sync()
  }

  function canSwap() {
    if (state.cursor !== 1) {
      return false
    }
    const first = formatLine([state.moves[0]])
    return first !== "pass" && first !== "swap" && tryParseCell(first) !== null
  }

  function passMove() {
    playMove("pass")
  }

  function swapMove() {
    if (canSwap()) {
      playMove("swap")
    }
  }

  function handleSwapShortcut(event) {
    if (shouldIgnoreGlobalKeydown(event)) {
      return false
    }
    if (!(event.key === "s" || event.key === "S")) {
      return false
    }
    const moves = currentMoves()
    if (canSwap()) {
      event.preventDefault()
      playMove("swap")
      return true
    }
    if (moves.length === 2 && moves[1] === "swap") {
      event.preventDefault()
      goPrevious()
      return true
    }
    return false
  }

  function handlePassShortcut(event) {
    if (shouldIgnoreGlobalKeydown(event)) {
      return false
    }
    if (!(event.shiftKey && event.key === "P")) {
      return false
    }
    event.preventDefault()
    passMove()
    return true
  }

  function resetBoard(boardSize = state.boardSize) {
    const size = parseBoardSize(boardSize)
    if (size === null) {
      syncSizeInput()
      return
    }
    state.boardSize = size
    state.moves = []
    state.cursor = 0
    sync()
  }

  function applySizeInput() {
    const size = parseBoardSize(elements.sizeInput.value)
    if (size === null) {
      syncSizeInput()
      return
    }
    if (size === state.boardSize) {
      syncSizeInput()
      return
    }
    resetBoard(size)
  }

  function stepSize(delta) {
    const nextSize = Math.max(
      MIN_BOARD_SIZE,
      Math.min(MAX_BOARD_SIZE, state.boardSize + Number(delta)),
    )
    if (nextSize === state.boardSize) {
      syncSizeInput()
      return false
    }
    resetBoard(nextSize)
    return true
  }

  function loadHash() {
    const hashText = window.location.hash ? decodeURIComponent(window.location.hash.slice(1)) : ""
    if (hashText) {
      const parsed = parseLine(hashText)
      if (parsed) {
        state.boardSize = parsed.boardSize
        state.moves = parsed.moves
        state.cursor = Math.max(0, Math.min(state.moves.length, parsed.cursor))
      } else {
        replaceHash("")
        state.boardSize = 15
        state.moves = []
        state.cursor = 0
      }
    }
    sync({ rewriteHash: !hashText })
  }

  function goPrevious() {
    if (state.cursor > 0) {
      state.cursor -= 1
      sync()
    }
  }

  function goNext() {
    if (state.cursor < state.moves.length) {
      state.cursor += 1
      sync()
    }
  }

  function goFirst() {
    if (state.cursor !== 0) {
      state.cursor = 0
      sync()
    }
  }

  function goLast() {
    if (state.cursor !== state.moves.length) {
      state.cursor = state.moves.length
      sync()
    }
  }

  function stepCursor(delta) {
    const nextCursor = Math.max(0, Math.min(state.moves.length, state.cursor + Number(delta)))
    if (nextCursor === state.cursor) {
      syncNavigationButtons()
      return false
    }
    state.cursor = nextCursor
    sync()
    return true
  }

  function deleteFromCursor() {
    if (state.cursor < state.moves.length) {
      state.moves = state.moves.slice(0, state.cursor)
    } else if (state.moves.length > 0) {
      state.moves.pop()
      state.cursor = state.moves.length
    }
    sync()
  }

  elements.currentLine.addEventListener("click", () => {
    void copyTextToClipboard(currentLineText())
  })
  elements.sizeStepper.addEventListener("contextmenu", (event) => {
    event.preventDefault()
  }, { capture: true })
  elements.sizeStepper.addEventListener("selectstart", (event) => {
    event.preventDefault()
  }, { capture: true })
  elements.moveNav.addEventListener("contextmenu", (event) => {
    event.preventDefault()
  }, { capture: true })
  elements.moveNav.addEventListener("selectstart", (event) => {
    event.preventDefault()
  }, { capture: true })
  elements.resetBtn.addEventListener("click", () => resetBoard())
  installHoldButton(elements.sizePrevBtn, () => stepSize(-1))
  installHoldButton(elements.sizeNextBtn, () => stepSize(1))
  installHoldButton(elements.movePrevBtn, () => stepCursor(-1))
  installHoldButton(elements.moveNextBtn, () => stepCursor(1))
  elements.moveFirstBtn.addEventListener("click", () => {
    if (!navButtonDisabled(elements.moveFirstBtn)) {
      goFirst()
    }
  })
  elements.moveLastBtn.addEventListener("click", () => {
    if (!navButtonDisabled(elements.moveLastBtn)) {
      goLast()
    }
  })
  elements.sizeInput.addEventListener("blur", syncSizeInput)
  elements.sizeInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      applySizeInput()
      elements.sizeInput.blur()
    } else if (event.key === "Escape") {
      syncSizeInput()
      elements.sizeInput.blur()
    }
  })
  window.addEventListener("hashchange", loadHash)
  window.addEventListener("keydown", (event) => {
    if (handleSwapShortcut(event) || handlePassShortcut(event)) {
      return
    }
    handleStandardKeydown(event, {
      goPrevious,
      goNext,
      goFirst,
      goLast,
      deleteFromCursor,
    })
  })

  loadHash()
})()
