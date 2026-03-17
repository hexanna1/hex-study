(() => {
  const SVG_NS = "http://www.w3.org/2000/svg"
  const STACK_TEXT_GAP_PX = -2

  function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value)))
  }

  function lerpRgb(a, b, t) {
    const tt = clamp01(t)
    return a.map((value, index) => Math.round(value + (b[index] - value) * tt))
  }

  function rgbText(rgb) {
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
  }

  function fractionPercent(value) {
    return `${(100 * Number(value)).toFixed(1)}%`
  }

  function formatVisits(value) {
    if (value === null || value === undefined) {
      return ""
    }
    const v = Number(value)
    if (!Number.isFinite(v)) {
      return ""
    }
    if (v < 1000) {
      return String(Math.trunc(v))
    }
    return `${(v / 1000).toFixed(1)}k`
  }

  function makeResultFill(lowRgb, highRgb, exponent = 0.9) {
    return (value) => rgbText(lerpRgb(lowRgb, highRgb, clamp01(value) ** exponent))
  }

  function buildDescendantCounts(nodesByLine, getChildLines) {
    const memo = new Map()
    const visiting = new Set()

    function count(line) {
      if (memo.has(line)) {
        return memo.get(line)
      }
      if (visiting.has(line)) {
        return 0
      }
      const node = nodesByLine.get(line)
      if (!node) {
        memo.set(line, 0)
        return 0
      }
      visiting.add(line)
      let total = 0
      for (const childLineValue of getChildLines(node, line) || []) {
        const childLine = String(childLineValue || "")
        total += 1
        if (nodesByLine.has(childLine)) {
          total += count(childLine)
        }
      }
      visiting.delete(line)
      memo.set(line, total)
      return total
    }

    for (const line of nodesByLine.keys()) {
      count(line)
    }
    return memo
  }

  function buildRetainedLeafLines(nodesByLine, getChildLines) {
    const leaves = new Set()
    for (const [line, node] of nodesByLine.entries()) {
      const childLines = []
      for (const childLineValue of getChildLines(node, line) || []) {
        const childLine = String(childLineValue || "")
        if (childLine) {
          childLines.push(childLine)
        }
      }
      if (childLines.length === 0) {
        leaves.add(String(line))
        continue
      }
      for (const childLine of childLines) {
        if (!nodesByLine.has(childLine)) {
          leaves.add(String(childLine))
        }
      }
    }
    return [...leaves]
  }

  function buildCoreLines(nodesByLine, importanceMin) {
    const lines = []
    for (const [line, node] of nodesByLine.entries()) {
      if (!line) {
        continue
      }
      if (Number(node?.importance || 0) >= Number(importanceMin)) {
        lines.push(line)
      }
    }
    return lines
  }

  function renderMoveList({
    container,
    parts,
    currentMoveCount,
    activateLine,
  }) {
    container.replaceChildren()
    for (let i = 0; i < parts.length; i += 2) {
      const row = document.createElement("div")
      row.className = "move-list-row"

      const ply = document.createElement("span")
      ply.className = "move-list-ply"
      ply.textContent = `${i + 1}.`
      row.appendChild(ply)

      for (let j = i; j < i + 2; j += 1) {
        const part = parts[j]
        if (part) {
          const move = document.createElement("span")
          const classes = ["move-list-move", "move-list-link"]
          if (part.isFuture) {
            classes.push("move-list-future")
          } else {
            classes.push(j % 2 === 0 ? "move-list-red" : "move-list-blue")
          }
          move.className = classes.join(" ")
          move.setAttribute("role", "button")
          move.tabIndex = 0
          move.textContent = String(part.text || "")
          if (j + 1 === Number(currentMoveCount)) {
            move.dataset.currentMove = "true"
          }
          const activate = () => {
            activateLine(part.line)
          }
          move.addEventListener("click", activate)
          move.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault()
              activate()
            }
          })
          row.appendChild(move)
        } else {
          const empty = document.createElement("span")
          empty.className = "move-list-move"
          row.appendChild(empty)
        }
      }

      container.appendChild(row)
    }
    const currentMove = container.querySelector("[data-current-move='true']")
    if (currentMove instanceof HTMLElement) {
      currentMove.scrollIntoView({ block: "nearest" })
    }
  }

  function createSvgTools({
    board,
    hexSize,
    defaultFill,
    defaultStroke,
    defaultStrokeWidth,
  }) {
    function createNode(tagName) {
      return document.createElementNS(SVG_NS, tagName)
    }

    function pointToPixel(col, row) {
      return [
        hexSize * Math.sqrt(3) * (col + row / 2),
        hexSize * 1.5 * row,
      ]
    }

    function hexCorner(cx, cy, size, index) {
      const angle = ((60 * index - 30) * Math.PI) / 180
      return [cx + size * Math.cos(angle), cy + size * Math.sin(angle)]
    }

    function hexPolygonPoints(cx, cy, size) {
      const points = []
      for (let i = 0; i < 6; i += 1) {
        const angle = ((60 * i - 30) * Math.PI) / 180
        points.push(`${cx + size * Math.cos(angle)},${cy + size * Math.sin(angle)}`)
      }
      return points.join(" ")
    }

    function clear() {
      while (board.firstChild) {
        board.removeChild(board.firstChild)
      }
    }

    function appendLine(x1, y1, x2, y2, stroke, strokeWidth) {
      const line = createNode("line")
      line.setAttribute("x1", String(x1))
      line.setAttribute("y1", String(y1))
      line.setAttribute("x2", String(x2))
      line.setAttribute("y2", String(y2))
      line.setAttribute("stroke", stroke)
      line.setAttribute("stroke-width", String(strokeWidth))
      line.setAttribute("stroke-linecap", "square")
      board.appendChild(line)
    }

    function appendHex(col, row, options = {}) {
      const [cx, cy] = pointToPixel(col, row)
      const polygon = createNode("polygon")
      polygon.setAttribute("points", hexPolygonPoints(cx, cy, options.size || (hexSize - 1.5)))
      polygon.setAttribute("class", options.className || "board-hex")
      polygon.setAttribute("fill", options.fill ?? defaultFill)
      polygon.setAttribute("stroke", options.stroke ?? defaultStroke)
      polygon.setAttribute("stroke-width", options.strokeWidth ?? defaultStrokeWidth)
      if (typeof options.title === "string" && options.title) {
        const title = createNode("title")
        title.textContent = options.title
        polygon.appendChild(title)
      }
      if (typeof options.onClick === "function") {
        polygon.addEventListener("click", options.onClick)
      }
      board.appendChild(polygon)
      return { cx, cy, polygon }
    }

    function appendText(cx, cy, text, className = "cell-text", fill = null) {
      const node = createNode("text")
      node.setAttribute("class", className)
      node.setAttribute("x", String(cx))
      node.setAttribute("y", String(cy + 0.4))
      if (typeof fill === "string" && fill) {
        node.style.fill = fill
      }
      node.textContent = text
      board.appendChild(node)
      return node
    }

    function appendTextAtY(cx, y, text, className = "cell-text", fill = null) {
      const node = appendText(cx, y - 0.4, text, className, fill)
      node.setAttribute("y", String(y))
      return node
    }

    function appendStackedText(
      cx,
      cy,
      topText,
      bottomText,
      topClassName = "cell-stack-text",
      bottomClassName = "cell-stack-text",
    ) {
      const centerY = cy + 0.4
      const topNode = appendTextAtY(cx, 0, topText, topClassName)
      const bottomNode = appendTextAtY(cx, 0, bottomText, bottomClassName, "rgba(17, 17, 17, 0.82)")

      const topBox = topNode.getBBox()
      const bottomBox = bottomNode.getBBox()
      const topCenterOffset = topBox.y + (topBox.height / 2)
      const bottomCenterOffset = bottomBox.y + (bottomBox.height / 2)
      const totalHeight = topBox.height + STACK_TEXT_GAP_PX + bottomBox.height
      const topTargetCenter = centerY - (totalHeight / 2) + (topBox.height / 2)
      const bottomTargetCenter = centerY + (totalHeight / 2) - (bottomBox.height / 2)

      topNode.setAttribute("y", String(topTargetCenter - topCenterOffset))
      bottomNode.setAttribute("y", String(bottomTargetCenter - bottomCenterOffset))
    }

    return {
      appendHex,
      appendLine,
      appendStackedText,
      appendText,
      clear,
      createNode,
      hexCorner,
      hexPolygonPoints,
      pointToPixel,
    }
  }

  function createLineNavigator({
    state,
    parseLine,
    linePrefixes,
    lineParent,
    sanitizeLine,
    setHashFromLine,
    render,
    entryEquals = (a, b) => a === b,
    canFollowLine = null,
  }) {
    function normalize(line) {
      return sanitizeLine(String(line || ""))
    }

    function resetLineHistory(line) {
      const current = normalize(line)
      if (!current) {
        state.lineHistory = [""]
        state.lineHistoryIndex = 0
        return
      }
      state.lineHistory = ["", ...linePrefixes(current)]
      state.lineHistoryIndex = state.lineHistory.length - 1
    }

    function jumpToLine(line) {
      state.currentLine = normalize(line)
      resetLineHistory(state.currentLine)
      setHashFromLine(state.currentLine)
      render()
    }

    function futureTailLines() {
      const current = normalize(state.currentLine)
      const lines = []
      let previousLine = current
      let previousEntries = parseLine(current)
      for (let i = state.lineHistoryIndex + 1; i < state.lineHistory.length; i += 1) {
        const nextLine = normalize(state.lineHistory[i])
        if (typeof canFollowLine === "function" && !canFollowLine(previousLine, nextLine)) {
          break
        }
        const nextEntries = parseLine(nextLine)
        if (nextEntries.length !== previousEntries.length + 1) {
          break
        }
        let matchesPrefix = true
        for (let j = 0; j < previousEntries.length; j += 1) {
          if (!entryEquals(previousEntries[j], nextEntries[j])) {
            matchesPrefix = false
            break
          }
        }
        if (!matchesPrefix) {
          break
        }
        lines.push(nextLine)
        previousLine = nextLine
        previousEntries = nextEntries
      }
      return lines
    }

    function moveListHistory() {
      return [...linePrefixes(String(state.currentLine || "")), ...futureTailLines()]
    }

    function setCursorLine(line) {
      const nextLine = normalize(line)
      const history = moveListHistory()
      const historyIndex = history.indexOf(nextLine)
      if (historyIndex >= 0) {
        state.lineHistory = history
        state.lineHistoryIndex = historyIndex
        state.currentLine = nextLine
        setHashFromLine(nextLine)
        render()
        return
      }
      jumpToLine(nextLine)
    }

    function goToLine(line) {
      const nextLine = normalize(line)
      if (nextLine === state.currentLine) {
        setHashFromLine(nextLine)
        render()
        return
      }
      const hasFuture = state.lineHistoryIndex + 1 < state.lineHistory.length
      const futureLine = hasFuture ? normalize(state.lineHistory[state.lineHistoryIndex + 1]) : null
      if (hasFuture && futureLine === nextLine) {
        state.lineHistoryIndex += 1
      } else {
        state.lineHistory = state.lineHistory.slice(0, state.lineHistoryIndex + 1)
        state.lineHistory.push(nextLine)
        state.lineHistoryIndex = state.lineHistory.length - 1
      }
      state.currentLine = nextLine
      setHashFromLine(state.currentLine)
      render()
    }

    function goPrevious() {
      if (state.lineHistoryIndex <= 0) {
        const parent = normalize(lineParent(state.currentLine))
        if (parent === state.currentLine) {
          return
        }
        state.lineHistory.unshift(parent)
        state.lineHistoryIndex += 1
      }
      state.lineHistoryIndex -= 1
      state.currentLine = normalize(state.lineHistory[state.lineHistoryIndex])
      setHashFromLine(state.currentLine)
      render()
    }

    function goNext() {
      if (state.lineHistoryIndex + 1 >= state.lineHistory.length) {
        return
      }
      state.lineHistoryIndex += 1
      state.currentLine = normalize(state.lineHistory[state.lineHistoryIndex])
      setHashFromLine(state.currentLine)
      render()
    }

    function goFirst() {
      if (state.lineHistory.length === 0) {
        return
      }
      state.lineHistoryIndex = 0
      state.currentLine = normalize(state.lineHistory[0])
      setHashFromLine(state.currentLine)
      render()
    }

    function goLast() {
      if (state.lineHistory.length === 0) {
        return
      }
      state.lineHistoryIndex = state.lineHistory.length - 1
      state.currentLine = normalize(state.lineHistory[state.lineHistoryIndex])
      setHashFromLine(state.currentLine)
      render()
    }

    function deleteFromCursor() {
      if (state.lineHistoryIndex + 1 < state.lineHistory.length) {
        state.lineHistory = state.lineHistory.slice(0, state.lineHistoryIndex + 1)
        render()
        return
      }
      if (!state.currentLine) {
        return
      }
      const parent = normalize(lineParent(state.currentLine))
      if (state.lineHistoryIndex > 0) {
        state.lineHistory = state.lineHistory.slice(0, state.lineHistoryIndex)
        state.lineHistoryIndex -= 1
      } else {
        state.lineHistory = [parent]
        state.lineHistoryIndex = 0
      }
      state.currentLine = parent
      setHashFromLine(state.currentLine)
      render()
    }

    return {
      deleteFromCursor,
      futureTailLines,
      goFirst,
      goLast,
      goNext,
      goPrevious,
      goToLine,
      jumpToLine,
      resetLineHistory,
      setCursorLine,
    }
  }

  window.HexStudyUI = {
    buildCoreLines,
    buildDescendantCounts,
    buildRetainedLeafLines,
    clamp01,
    createLineNavigator,
    createSvgTools,
    fractionPercent,
    formatVisits,
    lerpRgb,
    makeResultFill,
    renderMoveList,
    rgbText,
  }
})()
