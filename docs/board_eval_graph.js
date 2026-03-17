(() => {
  const { rgbText, THEME } = window.HexStudyUI
  const MAX_EVALUATIONS = 4096
  const SVG_NS = "http://www.w3.org/2000/svg"

  function create(editor) {
    const wrap = document.getElementById("analysis-graph-wrap")
    const svg = document.getElementById("analysis-graph")
    const card = document.getElementById("analysis-graph-card")
    const label = document.createElement("p")
    const preview = document.createElementNS(SVG_NS, "svg")
    card.append(label, preview)
    const evaluations = new Map()
    let variation = { cursor: 0, points: [] }
    let width = 320
    let hoverPly = null
    let previewKey = null
    let boardOrientation = "flat"
    let pointer = null
    const height = 164, left = 34, right = 12, top = 14, bottom = 24
    const red = rgbText(THEME.RED_RGB)

    function append(tag, attributes, text = "") {
      const node = document.createElementNS(SVG_NS, tag)
      for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value))
      node.textContent = text
      svg.appendChild(node)
    }

    const xFor = (ply) => left + ply / Math.max(10, variation.points.length - 1) * (width - left - right)
    const yFor = (value) => top + (1 - value) * (height - top - bottom)

    function hideCard() {
      hoverPly = null
      pointer = null
      card.hidden = true
    }

    function drawPreview(point) {
      const { game, boardSize, board } = point.preview()
      window.BoardGraphPreview.draw(preview, { game, boardSize, board, orientation: boardOrientation })
    }

    function showCard() {
      const point = variation.points[hoverPly]
      if (!point || !pointer) { card.hidden = true; return }
      const value = evaluations.get(point.key)
      const move = point.ply ? `${point.ply}. ${point.move}` : "Start"
      label.textContent = value
        ? `${move} · Red ${(100 * value.redWinrate).toFixed(1)}%`
        : `${move} · Not analyzed`
      if (previewKey !== point.key) {
        drawPreview(point)
        previewKey = point.key
      }
      window.BoardGraphPreview.place(card, pointer.x, pointer.y)
    }

    function render() {
      wrap.hidden = !variation.points.some((point) => evaluations.has(point.key))
      if (wrap.hidden) { hideCard(); return }
      width = Math.max(100, svg.getBoundingClientRect().width || 320)
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`)
      svg.replaceChildren()
      for (const value of [0, 0.5, 1]) {
        const y = yFor(value)
        append("line", { x1: left, x2: width - right, y1: y, y2: y, class: "analysis-graph-grid" })
        append("text", { x: left - 6, y: y + 4, "text-anchor": "end", class: "eval-graph-axis-label" }, `${value * 100}`)
      }
      const maxPly = Math.max(10, variation.points.length - 1)
      const step = Math.max(1, Math.ceil(maxPly / Math.max(2, Math.floor(width / 65))))
      for (let ply = 0; ply <= maxPly; ply += step) {
        append("text", { x: xFor(ply), y: height - 5, "text-anchor": "middle", class: "eval-graph-axis-label" }, ply)
      }
      const cursorX = xFor(variation.cursor)
      append("line", { x1: cursorX, x2: cursorX, y1: top, y2: height - bottom, class: "analysis-graph-cursor" })
      let segment = []
      function flush() {
        if (segment.length > 1) append("polyline", { points: segment.join(" "), fill: "none", stroke: red, "stroke-width": 2 })
        segment = []
      }
      for (const point of variation.points) {
        const value = evaluations.get(point.key)
        if (!value) { flush(); continue }
        const x = xFor(point.ply), y = yFor(value.redWinrate)
        segment.push(`${x},${y}`)
        append("circle", { cx: x, cy: y, r: point.ply === variation.cursor ? 4 : 2, fill: red })
      }
      flush()
      showCard()
    }

    function plyAt(event) {
      const rect = svg.getBoundingClientRect()
      const x = (event.clientX - rect.left) * width / rect.width
      const ply = Math.round((x - left) / (width - left - right) * Math.max(10, variation.points.length - 1))
      return Math.max(0, Math.min(variation.points.length - 1, ply))
    }
    svg.addEventListener("pointermove", (event) => {
      if (event.pointerType === "touch") { hideCard(); return }
      hoverPly = plyAt(event)
      pointer = { x: event.clientX, y: event.clientY }
      showCard()
    })
    svg.addEventListener("pointerleave", hideCard)
    svg.addEventListener("pointercancel", hideCard)
    svg.addEventListener("click", (event) => {
      const point = variation.points[plyAt(event)]
      hideCard()
      point?.select()
    })
    svg.addEventListener("keydown", (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
      const ply = { ArrowLeft: variation.cursor - 1, ArrowRight: variation.cursor + 1,
        Home: 0, End: variation.points.length - 1 }[event.key]
      if (ply === undefined) return
      event.preventDefault()
      event.stopPropagation()
      hideCard()
      variation.points[Math.max(0, Math.min(variation.points.length - 1, ply))]?.select()
    })
    window.addEventListener("blur", hideCard)
    window.addEventListener("scroll", hideCard, true)
    new ResizeObserver(render).observe(svg)
    editor.subscribeVariation((next) => {
      variation = next
      hideCard()
      render()
    })
    editor.subscribeBoardOrientation((next) => {
      boardOrientation = next
      previewKey = null
      if (!card.hidden) showCard()
    })
    function remember(key, value) {
      const previous = evaluations.get(key)
      if (previous && previous.visits > value.visits) return
      evaluations.delete(key)
      evaluations.set(key, value)
      if (evaluations.size > MAX_EVALUATIONS) evaluations.delete(evaluations.keys().next().value)
    }
    return {
      recordBatch(line, results) {
        for (const row of results) {
          const key = line.points[row.ply].key
          if (!evaluations.has(key)) remember(key, row)
        }
        render()
      },
      record(position, result) {
        const winrate = result.candidates.find((row) => row.visits > 0)?.winrate ?? result.winrate
        if (!Number.isFinite(winrate)) return
        remember(position.key, { visits: result.visits,
          redWinrate: position.toPlay === "red" ? winrate : 1 - winrate })
        render()
      },
      clear() {
        evaluations.clear()
        previewKey = null
        render()
      },
    }
  }

  window.BoardEvalGraph = { create }
})()
