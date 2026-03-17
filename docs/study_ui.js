(() => {
  const SVG_NS = "http://www.w3.org/2000/svg"
  const STACK_TEXT_GAP_PX = -2
  const HOLD_REPEAT_INTERVAL_NUMERATOR_MS = 200
  const THEME = {
    BLUE_RGB: [40, 100, 220],
    CANDIDATE_HIGH: [170, 125, 210],
    CANDIDATE_LOW: [244, 232, 250],
    GRID_EDGE: "rgb(182, 182, 182)",
    OFF_WHITE_RGB: [246, 241, 232],
    RED_RGB: [220, 60, 60],
    TEXT_ON_DARK_RGB: [250, 250, 250],
  }

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

  function rgbaText(rgb, alpha) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`
  }

  function turnRgb(toPlay) {
    return toPlay === "red" ? THEME.RED_RGB : THEME.BLUE_RGB
  }

  function turnRgbaText(toPlay, alpha) {
    return rgbaText(turnRgb(toPlay), alpha)
  }

  function decodeThousandths(raw) {
    if (typeof raw !== "number") {
      return null
    }
    return Number(raw) / 1000
  }

  function decodeOptionalThousandths(raw, nullSentinel = 1023) {
    return raw === nullSentinel ? null : decodeThousandths(raw)
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
    if (v < 100000) {
      return `${(v / 1000).toFixed(1)}k`
    }
    return `${Math.trunc(v / 1000)}k`
  }

  function percentText(value) {
    return typeof value === "number" && Number.isFinite(value) ? (100 * value).toFixed(1) : ""
  }

  function hexWorldUrlWithCursor(base, pastStream = "", futureStream = "") {
    const past = String(pastStream || "")
    const future = String(futureStream || "")
    if (!future) {
      return past ? `${base},${past}` : String(base || "")
    }
    if (!past) {
      return `${base},,${future}`
    }
    return `${base},${past},${future}`
  }

  function decodeLocationHash() {
    if (!window.location.hash) {
      return ""
    }
    try {
      return decodeURIComponent(window.location.hash.slice(1))
    } catch {
      return null
    }
  }

  async function fetchOk(url, options = {}) {
    const { label = "", ...fetchOptions } = options
    const response = await fetch(url, fetchOptions)
    if (!response.ok) {
      throw new Error(`${label ? `${label}: ` : ""}HTTP ${response.status}`)
    }
    return response
  }

  async function fetchJson(url, options = {}) {
    return (await fetchOk(url, options)).json()
  }

  async function fetchArrayBuffer(url, options = {}) {
    return (await fetchOk(url, options)).arrayBuffer()
  }

  function renderExternalLink(container, url, text = "View in HexWorld") {
    if (!(container instanceof HTMLElement) || !String(url || "")) {
      return
    }
    const a = document.createElement("a")
    a.href = String(url)
    a.target = "_blank"
    a.rel = "noopener noreferrer"
    a.textContent = text
    container.appendChild(a)
  }

  function replaceHash(hash = "") {
    const hashText = String(hash || "")
    const normalizedHash = hashText && !hashText.startsWith("#") ? `#${hashText}` : hashText
    // Hash rewrites intentionally discard query parameters.
    const nextUrl = `${window.location.pathname}${normalizedHash}`
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`
    if (nextUrl !== currentUrl) {
      window.history.replaceState(null, "", nextUrl)
    }
  }

  function setTurnStatus(element, toPlay) {
    if (!(element instanceof HTMLElement)) {
      return
    }
    if (!(toPlay === "red" || toPlay === "blue")) {
      element.textContent = "Turn: —"
      element.className = "turn-indicator"
      return
    }
    element.textContent = `Turn: ${toPlay === "red" ? "Red" : "Blue"}`
    element.className = `turn-indicator ${toPlay === "red" ? "turn-red" : "turn-blue"}`
  }

  async function copyTextToClipboard(text) {
    const value = String(text || "").trim()
    if (!value) {
      return
    }
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(value)
      }
    } catch (_error) {}
  }

  function syncPressedButtonGroup(rows, activeValue, isActive = (value, current) => value === current) {
    for (const [value, button] of rows || []) {
      if (!(button instanceof HTMLButtonElement)) {
        continue
      }
      const active = Boolean(isActive(value, activeValue))
      button.classList.toggle("is-active", active)
      button.setAttribute("aria-pressed", active ? "true" : "false")
    }
  }

  function createModeButtonGroup({
    state,
    field,
    values,
    rows,
    defaultValue = null,
    render = null,
  }) {
    const allowedValues = Array.isArray(values) ? values.map((value) => String(value)) : []
    const fallback = defaultValue === null ? allowedValues[0] : String(defaultValue)
    const buttonRows = Array.isArray(rows) ? rows : []

    function normalizeMode(mode) {
      const modeText = String(mode || "")
      return allowedValues.includes(modeText) ? modeText : fallback
    }

    function sync() {
      syncPressedButtonGroup(buttonRows, normalizeMode(state[field]))
    }

    function set(mode) {
      const nextMode = normalizeMode(mode)
      if (state[field] === nextMode) {
        return
      }
      state[field] = nextMode
      if (typeof render === "function") {
        render()
      }
    }

    function toggle() {
      const currentMode = normalizeMode(state[field])
      const currentIndex = allowedValues.indexOf(currentMode)
      const nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % allowedValues.length
      set(allowedValues[nextIndex])
    }

    return { set, sync, toggle }
  }

  function shouldIgnoreGlobalKeydown(event) {
    if (!event || event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
      return true
    }
    const target = event.target
    if (target instanceof HTMLElement) {
      const tag = target.tagName.toLowerCase()
      if (tag === "input" || tag === "textarea" || target.isContentEditable) {
        return true
      }
    }
    return false
  }

  function handleStandardKeydown(event, actions) {
    if (shouldIgnoreGlobalKeydown(event)) {
      return false
    }
    if (event.key === "t" || event.key === "T") {
      event.preventDefault()
      actions.toggleOverlayMode?.()
      return true
    }
    if (event.key === "p" || event.key === "P" || event.key === "ArrowLeft") {
      event.preventDefault()
      actions.goPrevious?.()
      return true
    }
    if (event.key === "n" || event.key === "N" || event.key === "ArrowRight") {
      event.preventDefault()
      actions.goNext?.()
      return true
    }
    if (event.key === "f" || event.key === "F") {
      event.preventDefault()
      actions.goFirst?.()
      return true
    }
    if (event.key === "l" || event.key === "L") {
      event.preventDefault()
      actions.goLast?.()
      return true
    }
    if (event.key === "Backspace" || event.key === "Delete") {
      if (typeof actions.canDelete === "function" && !actions.canDelete()) {
        return false
      }
      event.preventDefault()
      actions.deleteFromCursor?.()
      return true
    }
    return false
  }

  function setNavButtonDisabled(button, disabled) {
    if (!(button instanceof HTMLButtonElement)) {
      return
    }
    button.disabled = false
    button.setAttribute("aria-disabled", disabled ? "true" : "false")
    button.tabIndex = disabled ? -1 : 0
  }

  function navButtonDisabled(button) {
    return button?.getAttribute("aria-disabled") === "true"
  }

  function setSvgViewBoxFromPixels(svg, pixels, padding = 0) {
    if (!(svg instanceof SVGSVGElement) || !Array.isArray(pixels) || pixels.length === 0) {
      return
    }
    const xs = pixels.map((point) => point[0])
    const ys = pixels.map((point) => point[1])
    const pad = Number(padding) || 0
    const minX = Math.min(...xs) - pad
    const maxX = Math.max(...xs) + pad
    const minY = Math.min(...ys) - pad
    const maxY = Math.max(...ys) + pad
    svg.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`)
  }

  function readAsciiMagic(view, offset = 0, length = 4) {
    const chars = []
    for (let idx = 0; idx < length; idx += 1) {
      chars.push(String.fromCharCode(view.getUint8(Number(offset) + idx)))
    }
    return chars.join("")
  }

  function readPackedWordAtBit(view, offset, bitOffset, bits, byteCount = 4) {
    if (bits === 0) {
      return 0
    }
    const byteOffset = offset + Math.trunc(bitOffset / 8)
    const shift = bitOffset % 8
    let chunk = 0
    for (let idx = 0; idx < byteCount && byteOffset + idx < view.byteLength; idx += 1) {
      chunk += view.getUint8(byteOffset + idx) * (2 ** (8 * idx))
    }
    return Math.trunc(chunk / (2 ** shift)) & ((2 ** bits) - 1)
  }

  function createKeyedDataLoader({
    state,
    loadingKeyField,
    current,
    load,
    apply,
    render,
  }) {
    return async (key) => {
      const existing = current(key)
      if (existing) {
        return existing
      }
      if (state.loadingPromise && state[loadingKeyField] === key) {
        return state.loadingPromise
      }
      state.loadAbortController?.abort()
      const abortController = new AbortController()
      const loadGeneration = Number(state.loadGeneration || 0) + 1
      state.loadGeneration = loadGeneration
      state.isLoadingData = true
      state[loadingKeyField] = key
      state.loadAbortController = abortController
      state.dataError = null
      render()
      state.loadingPromise = (async () => {
        const loaded = await load(key, abortController.signal)
        if (loadGeneration !== state.loadGeneration) {
          return
        }
        apply(loaded, key)
        state.dataError = null
        return loaded
      })().catch((error) => {
        if (loadGeneration !== state.loadGeneration) {
          return
        }
        if (error?.name === "AbortError") {
          return
        }
        state.dataError = String(error instanceof Error ? error.message : error)
      }).finally(() => {
        if (loadGeneration !== state.loadGeneration) {
          return
        }
        state.isLoadingData = false
        state.loadingPromise = null
        state[loadingKeyField] = null
        state.loadAbortController = null
      })
      return state.loadingPromise
    }
  }

  function holdRepeatInterval(repeatCount) {
    const count = Math.max(1, Number(repeatCount) || 1)
    return HOLD_REPEAT_INTERVAL_NUMERATOR_MS / (count ** (2 / 3))
  }

  function installHoldButton(button, action, isDisabled = null) {
    if (!(button instanceof HTMLButtonElement) || typeof action !== "function") {
      return
    }

    const disabled = () => (
      typeof isDisabled === "function" ? Boolean(isDisabled()) : navButtonDisabled(button)
    )

    let timerId = null
    let repeatCount = 0
    let pointerActive = false
    let suppressClick = false

    const stopHold = () => {
      if (timerId !== null) {
        window.clearTimeout(timerId)
        timerId = null
      }
      repeatCount = 0
      pointerActive = false
      button.classList.remove("is-holding")
    }

    const scheduleRepeat = (delayMs) => {
      timerId = window.setTimeout(() => {
        timerId = null
        if (!pointerActive) {
          return
        }
        if (!action() || disabled()) {
          stopHold()
          return
        }
        repeatCount += 1
        scheduleRepeat(holdRepeatInterval(repeatCount))
      }, delayMs)
    }

    button.addEventListener("pointerdown", (event) => {
      if (disabled() || event.button > 0) {
        return
      }
      event.preventDefault()
      stopHold()
      suppressClick = true
      pointerActive = true
      button.classList.add("is-holding")
      try {
        button.setPointerCapture(event.pointerId)
      } catch (_error) {
        // Pointer capture is best-effort; the button still works without it.
      }
      if (!action() || disabled()) {
        stopHold()
        return
      }
      repeatCount = 1
      scheduleRepeat(holdRepeatInterval(repeatCount))
    })
    button.addEventListener("pointerup", stopHold)
    button.addEventListener("pointercancel", stopHold)
    button.addEventListener("lostpointercapture", stopHold)
    button.addEventListener("click", (event) => {
      if (suppressClick) {
        suppressClick = false
        event.preventDefault()
        return
      }
      if (disabled()) {
        event.preventDefault()
        return
      }
      action()
    })
    window.addEventListener("pointerup", stopHold)
    window.addEventListener("pointercancel", stopHold)
    window.addEventListener("blur", stopHold)
  }

  function makeResultFill(lowRgb, highRgb, exponent = 0.9) {
    return (value) => rgbText(lerpRgb(lowRgb, highRgb, clamp01(value) ** exponent))
  }

  function hexataCandidateFill(lowRgb, highRgb, count, topCount, biasMetric) {
    const denom = Math.log(Math.max(2, Number(topCount) || 0))
    let t = denom <= 0 ? 0 : (Math.log(Math.max(1, Number(count) || 0)) / denom)
    t = clamp01(t) ** 1.1
    const bias = (Number(biasMetric) - 0.5) * 2.0
    t = clamp01(t + (0.35 * bias))
    return rgbText(lerpRgb(lowRgb, highRgb, t))
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

  function buildCoreLines(nodesByLine) {
    const lines = []
    for (const [line, node] of nodesByLine.entries()) {
      if (!line) {
        continue
      }
      if (node?.is_core) {
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
      const containerRect = container.getBoundingClientRect()
      const currentRect = currentMove.getBoundingClientRect()
      const currentTop = container.scrollTop + (currentRect.top - containerRect.top)
      const currentBottom = container.scrollTop + (currentRect.bottom - containerRect.top)
      const visibleTop = container.scrollTop
      const visibleBottom = visibleTop + container.clientHeight
      if (currentTop < visibleTop) {
        container.scrollTop = currentTop
      } else if (currentBottom > visibleBottom) {
        container.scrollTop = currentBottom - container.clientHeight
      }
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

    function appendHex(colOrPoint, rowOrOptions = null, maybeOptions = {}) {
      let col = colOrPoint
      let row = rowOrOptions
      let options = maybeOptions
      if (Array.isArray(colOrPoint)) {
        ;[col, row] = colOrPoint
        options = rowOrOptions || {}
      } else if (colOrPoint && typeof colOrPoint === "object") {
        col = colOrPoint.col
        row = colOrPoint.row
        options = rowOrOptions || {}
      }
      const [cx, cy] = pointToPixel(col, row)
      const polygon = createNode("polygon")
      polygon.setAttribute("points", hexPolygonPoints(cx, cy, options.size || (hexSize - 1.5)))
      polygon.setAttribute("class", options.className || "board-hex")
      polygon.setAttribute("fill", options.fill ?? defaultFill)
      polygon.setAttribute("stroke", options.stroke ?? defaultStroke)
      polygon.setAttribute("stroke-width", options.strokeWidth ?? defaultStrokeWidth)
      if (options.boardPoint) {
        polygon.setAttribute("data-board-point", "1")
        polygon.setAttribute("data-q", String(col))
        polygon.setAttribute("data-r", String(row))
      }
      if (options.attributes && typeof options.attributes === "object") {
        for (const [name, value] of Object.entries(options.attributes)) {
          if (value !== null && value !== undefined) {
            polygon.setAttribute(name, String(value))
          }
        }
      }
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
        setHashFromLine(state.currentLine)
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
    copyTextToClipboard,
    createKeyedDataLoader,
    createLineNavigator,
    createModeButtonGroup,
    createSvgTools,
    decodeLocationHash,
    decodeOptionalThousandths,
    decodeThousandths,
    fetchArrayBuffer,
    fetchJson,
    fetchOk,
    formatVisits,
    handleStandardKeydown,
    hexWorldUrlWithCursor,
    hexataCandidateFill,
    installHoldButton,
    lerpRgb,
    makeResultFill,
    navButtonDisabled,
    percentText,
    readAsciiMagic,
    readPackedWordAtBit,
    replaceHash,
    renderExternalLink,
    renderMoveList,
    rgbText,
    rgbaText,
    setNavButtonDisabled,
    setSvgViewBoxFromPixels,
    setTurnStatus,
    shouldIgnoreGlobalKeydown,
    syncPressedButtonGroup,
    THEME,
    turnRgb,
    turnRgbaText,
  }
})()
