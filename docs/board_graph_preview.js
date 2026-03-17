(() => {
  const { createSvgTools, rgbText, THEME } = window.HexStudyUI
  const { pointKey } = window.HexPosition
  const previews = new WeakMap()
  const emptyFill = rgbText(THEME.OFF_WHITE_RGB)
  const stoneFill = { red: rgbText(THEME.RED_RGB), blue: rgbText(THEME.BLUE_RGB) }
  const hexSize = 24

  function createPreview(svg, game, size, orientation) {
    svg.replaceChildren()
    const hex = game === "hex"
    const bent = game === "bentY"
    const geometry = globalThis.BentYGeometry
    const scale = bent ? geometry.widthForTensorLen(size) * 63 : 0
    const tools = hex ? window.HexMoveTree.createBoardSvg(svg) : createSvgTools({
      board: svg,
      hexSize,
      defaultFill: emptyFill,
      defaultStroke: THEME.GRID_EDGE,
      defaultStrokeWidth: 1,
      projectPoint: bent ? (_size, col, row) =>
        geometry.projectPoint(size, col, row).map((value) => value * scale) : null,
    })
    if (!bent) tools.setBoardOrientation(orientation)
    const points = hex ? Array.from({ length: size * size }, (_, point) => point)
      : globalThis.BoardGeometry.playablePoints(game, size)
    const cells = points.map((point) => ({
      col: (point % size) + 1,
      row: Math.floor(point / size) + 1,
    }))
    if (bent) {
      const graph = geometry.topology(size)
      const pixel = (point) => tools.pointToPixel((point % size) + 1, Math.floor(point / size) + 1)
      for (const [a, b] of graph.edges) {
        const boundary = graph.sideMasks[a] & graph.sideMasks[b]
        tools.appendLine(...pixel(a), ...pixel(b), boundary ? "#5c5348" : "#9b9080", boundary ? 3.2 : 1.8)
      }
    }
    const hexOptions = hex ? { className: "board-hex board-hex-face" } : {}
    const faces = cells.map(({ col, row }) => {
      const key = pointKey(col, row)
      if (bent) {
        const [cx, cy] = tools.pointToPixel(col, row)
        const polygon = tools.appendCircle(cx, cy, 26, { stroke: "#75664f", strokeWidth: 1.7, fill: emptyFill })
        return { key, cx, cy, polygon, color: null }
      }
      return { key, ...tools.appendHex(col, row, hexOptions), color: null }
    })
    const dot = tools.appendCircle(0, 0, hex ? Math.max(2, hexSize * 0.14) : 3,
      { fill: emptyFill, ...(hex && { className: "last-move-dot" }) })
    dot.remove()
    if (hex) {
      tools.setupViewBox(size)
      tools.renderFrame(size)
    } else {
      tools.setViewBoxFromPoints({ boardPoints: cells, coordPoints: [], viewPadding: bent ? 32 : 26, coordViewPadding: 0 })
    }
    function render(board) {
      let last = null
      for (const face of faces) {
        const stone = board.occupied.get(face.key)
        const color = stone?.color ?? null
        if (face.color !== color) {
          face.color = color
          face.polygon.setAttribute("fill", stoneFill[color] ?? emptyFill)
          if (hex) face.polygon.setAttribute("stroke", stone ? "none" : THEME.GRID_EDGE)
        }
        if (stone?.isLast) last = face
      }
      if (last) {
        dot.setAttribute("cx", last.cx)
        dot.setAttribute("cy", last.cy)
        last.polygon.after(dot)
      } else { dot.remove() }
    }
    return { game, size, orientation, render }
  }

  function draw(svg, { game, boardSize, board, orientation = "flat" }) {
    let preview = previews.get(svg)
    if (!preview || preview.game !== game || preview.size !== boardSize || preview.orientation !== orientation) {
      preview = createPreview(svg, game, boardSize, orientation)
      previews.set(svg, preview)
    }
    preview.render(board)
  }

  function place(card, x, y) {
    card.hidden = false
    const rect = card.getBoundingClientRect()
    const left = Math.max(8, Math.min(x + 14, window.innerWidth - rect.width - 8))
    const top =
      y - rect.height - 12 >= 8
        ? y - rect.height - 12
        : Math.max(8, Math.min(y + 16, window.innerHeight - rect.height - 8))
    card.style.left = `${left}px`
    card.style.top = `${top}px`
  }

  window.BoardGraphPreview = { draw, place }
})()
