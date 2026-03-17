(() => {
  function create({ game, defaultBoardSize, minBoardSize, maxBoardSize, boardSizes = null, model }) {
    const HEX_SIZE = 24
    const VIEW_PADDING = 34
    const COORD_VIEW_PADDING = 8

    const BOARD_SIDE = "rgb(82, 82, 82)"

    const { alphaLabel, materializeBoardState: materializeFullBoardState, tryParseCell } = globalThis.HexPosition
    const { collectBoardEditorElements, createBranchingBoardEditor } = window.HexBoardEditor
    const elements = collectBoardEditorElements(`${game}-status`)
    let analysis = null

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

    function drawBorderSegment(frame, col, row, c1, c2) {
      const [cx, cy] = frame.pointToPixel(col, row)
      const a = frame.hexCorner(cx, cy, HEX_SIZE - 1.5, c1)
      const b = frame.hexCorner(cx, cy, HEX_SIZE - 1.5, c2)
      frame.appendLine(a[0], a[1], b[0], b[1], BOARD_SIDE, 4)
    }

    function drawFrame(frame, boardSize, cells) {
      const coordinates = coordinatePoints(boardSize)
      frame.setViewBoxFromPoints({
        boardPoints: cells,
        coordPoints: coordinates,
        viewPadding: VIEW_PADDING,
        coordViewPadding: COORD_VIEW_PADDING,
      })
      const neighbors = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]]
      for (const { col, row } of cells) {
        neighbors.forEach(([dx, dy], edge) => {
          if (!isLegalCell(col + dx, row + dy, boardSize)) drawBorderSegment(frame, col, row, edge, (edge + 1) % 6)
        })
      }
      for (const { col, row, label } of coordinates) {
        const [cx, cy] = frame.pointToPixel(col, row)
        frame.appendText(cx, cy, label, "coord-text")
      }
    }

    const renderBoard = window.AnalysisBoardRenderer.create({
      board: elements.board, getAnalysis: () => analysis, drawFrame,
    })

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
  }

  const configurations = {
    y: { defaultBoardSize: 15, minBoardSize: 1, maxBoardSize: 42, model: "y-t4-24" },
    yavalath: {
      defaultBoardSize: 9,
      minBoardSize: 3,
      maxBoardSize: 9,
      boardSizes: [3, 5, 7, 9],
      model: "yavalath-t4-9",
    },
  }
  const { game } = document.querySelector("main").dataset
  create({ game, ...configurations[game] })
})()
