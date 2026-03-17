(() => {
  const MIN_BOARD_SIZE = 1
  const MAX_BOARD_SIZE = 42
  const DEFAULT_BOARD_SIZE = 14

  const {
    rgbText,
  } = window.HexStudyUI
  const {
    appendMoveToLine,
    createBoardSvg,
    formatCell,
    formatLine,
    materializeBoardState: materializeFullBoardState,
    pointKey,
    renderMoveTreeBoard,
    swapControlPoint,
    THEME: {
      BLUE_RGB,
      RED_RGB,
    },
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
    status: document.getElementById("hex-status"),
  }

  const boardSvg = createBoardSvg(elements.board)

  function isLegalCell(col, row, boardSize) {
    const size = Number(boardSize)
    return (
      Number.isInteger(col)
      && Number.isInteger(row)
      && col >= 1
      && row >= 1
      && col <= size
      && row <= size
    )
  }

  function materializeBoardState(moves, boardSize) {
    return materializeFullBoardState(moves, boardSize)
  }

  function boardCells(boardSize) {
    const cells = []
    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize; col += 1) {
        cells.push({ col, row })
      }
    }
    return cells
  }

  function renderBoard({
    state,
    currentMoves,
    materializeBoardState: materializeCurrentBoardState,
    boardCells: currentBoardCells,
    goPrevious,
    goToLine,
    canSwap,
    swapMove,
  }) {
    function currentNode() {
      const board = materializeCurrentBoardState(currentMoves())
      const candidates = []
      for (const cell of currentBoardCells()) {
        if (!board.occupied.has(pointKey(cell.col, cell.row))) {
          candidates.push({
            move: formatCell(cell.col, cell.row),
            retained: true,
          })
        }
      }
      return {
        line: formatLine(currentMoves()),
        candidates,
      }
    }

    const board = renderMoveTreeBoard({
      boardSvg,
      boardSize: state.boardSize,
      currentLine: formatLine(currentMoves()),
      currentNode: currentNode(),
      extraViewboxPoints: [swapControlPoint(state.boardSize)],
      childLineForCandidate: (line, candidate) => appendMoveToLine(line, candidate?.move),
      mirrorRootCandidates: false,
      buildOverlay: ({ childLine }) => ({
        childLine,
        className: "board-hex board-hex-face",
      }),
      candidateFill: () => "",
      overlayPrimaryText: () => "",
      onGoToLine: goToLine,
      onGoPrevious: goPrevious,
      showCellCoords: state.showCoords,
    })
    const dragActive = Boolean(state.drag)
    elements.board.classList.toggle("dragging", dragActive)
    if (dragActive) {
      const dragSourceKey = pointKey(state.drag.startPoint.col, state.drag.startPoint.row)
      const dragSource = board.cellsByKey.get(dragSourceKey) || null
      if (dragSource?.hex?.polygon) {
        dragSource.hex.polygon.classList.add("drag-source")
      }
      if (state.drag.targetPoint) {
        const ghostFill = state.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        boardSvg.appendHex(state.drag.targetPoint.col, state.drag.targetPoint.row, {
          fill: ghostFill,
          stroke: "none",
          className: "board-ghost board-ghost-target",
        })
      }
    }
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
    targetMoveForDrag: ({ moveIndex, targetPoint, currentMoves }) => {
      if (Number(moveIndex) === 0 && currentMoves.length >= 2 && currentMoves[1] === "swap") {
        return formatCell(targetPoint.row, targetPoint.col)
      }
      return formatCell(targetPoint.col, targetPoint.row)
    },
  })
})()
