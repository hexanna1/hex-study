(() => {
  const MIN_BOARD_SIZE = 1
  const MAX_BOARD_SIZE = 42
  const DEFAULT_BOARD_SIZE = 14

  const {
    analysisOverlayFill,
    rgbText,
    THEME: {
      BLUE_RGB,
      RED_RGB,
    },
  } = window.HexStudyUI
  const {
    createBoardSvg,
    renderMoveTreeBoard,
  } = window.HexMoveTree
  const {
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

  const elements = collectBoardEditorElements("hex-status")

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

  function cellsForBoardSize(boardSize) {
    const cells = []
    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize; col += 1) {
        cells.push({ col, row })
      }
    }
    return cells
  }

  function renderBoard({
    analysisByKey,
    board,
    branchChildren,
    display,
    position,
  }) {
    boardSvg.setBoardOrientation(display.boardOrientation)

    const renderedBoard = renderMoveTreeBoard({
      boardSvg,
      boardSize: position.boardSize,
      currentLine: position.line,
      boardState: board,
      cellOverlays: analysisByKey,
      isLegalCell: (col, row, currentBoard) => !currentBoard.occupied.has(pointKey(col, row)),
      mirrorRootCandidates: false,
      candidateFill: analysisOverlayFill,
      overlayPrimaryText: () => "",
      enableCellClicks: false,
      showCellCoords: display.showCoords,
      showMoveNumbers: display.showMoveNumbers,
    })
    const branchByKey = branchChildrenByKey(branchChildren, board.occupied)
    for (const renderedCell of renderedBoard.cellsByKey.values()) {
      if (renderedCell.stone && renderedCell.hoverHex?.polygon) {
        renderedCell.hoverHex.polygon.classList.add("board-drag-hit")
      }
    }
    for (const [key, child] of branchByKey.entries()) {
      const renderedCell = renderedBoard.cellsByKey.get(key) || null
      if (renderedCell?.hex?.polygon) {
        applyBranchOutline(renderedCell.hex.polygon, child, board.toPlay)
      }
    }
    const dragActive = Boolean(display.drag)
    elements.board.classList.toggle("dragging", dragActive)
    if (dragActive) {
      const dragSourceKey = pointKey(display.drag.startPoint.col, display.drag.startPoint.row)
      const dragSource = renderedBoard.cellsByKey.get(dragSourceKey) || null
      if (dragSource?.hex?.polygon) {
        dragSource.hex.polygon.classList.add("drag-source")
      }
      if (display.drag.targetPoint) {
        const ghostFill = display.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        boardSvg.appendHex(display.drag.targetPoint.col, display.drag.targetPoint.row, {
          fill: ghostFill,
          stroke: "none",
          className: "board-ghost board-ghost-target",
        })
      }
    }
  }

  createBranchingBoardEditor({
    game: "hex",
    elements,
    defaultBoardSize: DEFAULT_BOARD_SIZE,
    minBoardSize: MIN_BOARD_SIZE,
    maxBoardSize: MAX_BOARD_SIZE,
    isLegalCell,
    materializeBoardState,
    getBoardCells: cellsForBoardSize,
    renderBoard,
    supportsHexWorldImport: true,
    targetMoveForDrag: ({ moveIndex, targetPoint, currentMoves }) => {
      if (Number(moveIndex) === 0 && currentMoves.length >= 2 && currentMoves[1] === "swap") {
        return formatCell(targetPoint.row, targetPoint.col)
      }
      return formatCell(targetPoint.col, targetPoint.row)
    },
  })
})()
