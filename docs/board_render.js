(() => {
  const HEX_SIZE = 24

  function create({ board: svg, getAnalysis, drawFrame, dragAboveFrame = false }) {
    const {
      analysisOverlayFill, createSvgTools, rgbText, stoneTextColor,
      THEME: { BLUE_RGB, GRID_EDGE, OFF_WHITE_RGB, RED_RGB },
    } = window.HexStudyUI
    const { formatCell, pointKey } = globalThis.HexPosition
    const { applyBranchOutline, branchChildrenByKey } = window.HexBoardEditor
    const emptyFill = rgbText(OFF_WHITE_RGB)
    function createLayer() {
      const layer = document.createElementNS(svg.namespaceURI, "g")
      svg.appendChild(layer)
      return createSvgTools({
        board: svg,
        layer,
        hexSize: HEX_SIZE,
        defaultFill: emptyFill,
        defaultStroke: GRID_EDGE,
        defaultStrokeWidth: "0.85",
      })
    }
    const boardSvg = createLayer()
    const labels = createLayer()
    const frame = createLayer()
    const variation = createLayer()
    const faces = []
    let lastBoard = null, lastBranches = null, lastDragSource = null, framedSize = null

    function renderBoard({ analysisByKey, board, cells, branchChildren, display, position, previewOnly }) {
      const analysis = getAnalysis()
      if (previewOnly && !display.drag) {
        variation.beginFrame()
        analysis?.renderPv(variation, board.toPlay, display)
        variation.endFrame()
        return
      }
      if (boardSvg.setBoardOrientation(display.boardOrientation)) {
        labels.setBoardOrientation(display.boardOrientation)
        frame.setBoardOrientation(display.boardOrientation)
        variation.setBoardOrientation(display.boardOrientation)
        framedSize = null
        lastBoard = null
      }
      if (framedSize !== position.boardSize) {
        framedSize = position.boardSize
        frame.beginFrame()
        drawFrame(frame, framedSize, cells)
        frame.endFrame()
      }
      const overlayMode = analysis?.overlayMode() || "winrate"
      const dragActive = Boolean(display.drag)
      svg.classList.toggle("dragging", dragActive)
      const dragSourceKey = dragActive ? pointKey(display.drag.startPoint.col, display.drag.startPoint.row) : null
      // Search reports only change a face's fill and its text, leaving the hit targets and borders in place.
      const rebuild = board !== lastBoard || branchChildren !== lastBranches || dragSourceKey !== lastDragSource
      if (rebuild) {
        faces.length = 0
        const branchByKey = branchChildrenByKey(branchChildren, board.occupied)
        boardSvg.beginFrame()
        for (const { col, row } of cells) {
          const key = pointKey(col, row)
          const stone = board.occupied.get(key) || null
          const classes = ["board-hex", "board-hex-face"]
          if (dragSourceKey === key) classes.push("drag-source")
          const hitClasses = ["board-hover-hit"]
          if (!stone || stone.isLast) hitClasses.push("clickable")
          hitClasses.push(stone ? "board-drag-hit" : "hoverable")
          const label = formatCell(col, row)
          boardSvg.appendHex(col, row, {
            fill: "transparent",
            stroke: "none",
            className: hitClasses.join(" "),
            size: HEX_SIZE,
            title: label,
            boardPoint: true,
          })
          const fill = stone ? rgbText(stone.color === "red" ? RED_RGB : BLUE_RGB) : emptyFill
          const hex = boardSvg.appendHex(col, row, {
            fill,
            stroke: stone ? "none" : GRID_EDGE,
            className: classes.join(" "),
          })
          const branchChild = branchByKey.get(key)
          if (branchChild) applyBranchOutline(hex.polygon, branchChild, board.toPlay)
          faces.push({ ...hex, key, stone, label, fill, painted: fill })
        }
        boardSvg.endFrame()
        lastBoard = board
        lastBranches = branchChildren
        lastDragSource = dragSourceKey
      }
      labels.beginFrame()
      variation.beginFrame()
      for (const face of faces) {
        const { cx, cy, polygon, key, stone, label, fill } = face
        const overlay = stone ? null : analysisByKey.get(key)
        const faceFill = overlay ? analysisOverlayFill(overlay, { mode: overlayMode }) || fill : fill
        if (face.painted !== faceFill) {
          polygon.setAttribute("fill", faceFill)
          face.painted = faceFill
        }
        if (display.showCoords) {
          labels.appendText(cx, cy, label, "cell-text", stone ? stoneTextColor(stone) : null)
        } else if (overlay) {
          const { primary, secondary } = window.BoardAnalysis.overlayText(overlay, overlayMode)
          if (primary && secondary) labels.appendStackedText(cx, cy, primary, secondary)
          else if (primary || secondary) labels.appendText(cx, cy, primary || secondary)
        } else if (stone && display.showMoveNumbers) {
          labels.appendText(cx, cy, stone.ply, "cell-text", stoneTextColor(stone))
        } else if (stone?.isLast) {
          labels.appendCircle(cx, cy, Math.max(2, HEX_SIZE * 0.14), {
            className: "last-move-dot",
            fill: emptyFill,
          })
        }
      }
      if (dragActive && display.drag.targetPoint) {
        const ghostFill = display.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        const overlay = dragAboveFrame ? variation : labels
        overlay.appendHex(display.drag.targetPoint.col, display.drag.targetPoint.row, {
          fill: ghostFill,
          stroke: "none",
          className: "board-ghost",
        })
      }
      labels.endFrame()
      if (!dragActive) analysis?.renderPv(variation, board.toPlay, display)
      variation.endFrame()
    }

    return renderBoard
  }
  window.AnalysisBoardRenderer = { create }
})()
