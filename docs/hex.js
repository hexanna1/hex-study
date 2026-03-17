(() => {
  const MIN_BOARD_SIZE = 1
  const MAX_BOARD_SIZE = 42
  const DEFAULT_BOARD_SIZE = 14

  const { renderBoardFrame, setupBoardViewBox } = window.HexMoveTree
  const { formatCell, materializeBoardState } = globalThis.HexPosition
  const { collectBoardEditorElements, createBranchingBoardEditor } = window.HexBoardEditor
  const elements = collectBoardEditorElements("hex-status")
  let analysis = null

  function isLegalCell(col, row, boardSize) {
    const size = Number(boardSize)
    return Number.isInteger(col) && Number.isInteger(row) && col >= 1 && row >= 1 && col <= size && row <= size
  }

  function cellsForBoardSize(boardSize) {
    const cells = []
    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize; col += 1) { cells.push({ col, row }) }
    }
    return cells
  }

  const renderBoard = window.AnalysisBoardRenderer.create({
    board: elements.board,
    getAnalysis: () => analysis,
    drawFrame(frame, size) {
      setupBoardViewBox(frame, size)
      renderBoardFrame(frame, size)
    },
    dragAboveFrame: true,
  })

  const editor = createBranchingBoardEditor({
    game: "hex",
    elements,
    defaultBoardSize: DEFAULT_BOARD_SIZE,
    minBoardSize: MIN_BOARD_SIZE,
    maxBoardSize: MAX_BOARD_SIZE,
    isLegalCell,
    materializeBoardState,
    getBoardCells: cellsForBoardSize,
    renderBoard,
    getBoardDecorations: () => analysis?.decorations() || {},
    shortcutHelpGroups: window.BoardAnalysis.shortcutHelpGroups("hex"),
    supportsHexWorldImport: true,
    targetMoveForDrag: ({ moveIndex, targetPoint, currentMoves }) => {
      if (Number(moveIndex) === 0 && currentMoves.length >= 2 && currentMoves[1] === "swap") {
        return formatCell(targetPoint.row, targetPoint.col)
      }
      return formatCell(targetPoint.col, targetPoint.row)
    },
  })
  analysis = window.BoardAnalysis.create(editor)
})()
