(() => {
  const HEX_SIZE = 24
  const VIEW_PADDING = 34
  const COORD_VIEW_PADDING = 8

  const RED_RGB = [220, 60, 60]
  const BLUE_RGB = [40, 100, 220]
  const TEXT_ON_DARK_RGB = [250, 250, 250]
  const OFF_WHITE_RGB = [246, 241, 232]
  const GRID_EDGE = "rgb(182, 182, 182)"

  const {
    createSvgTools,
    lerpRgb,
    renderMoveList: renderSharedMoveList,
    rgbText,
    shouldIgnoreGlobalKeydown,
  } = window.HexStudyUI

  function numberText(value) {
    return Number(value).toFixed(1)
  }

  function normalizeMoveToken(token) {
    const raw = String(token || "").trim().toLowerCase()
    return raw === ":s" || raw === "swap" ? "swap" : raw
  }

  function parseMoves(line) {
    const raw = String(line || "").trim().toLowerCase()
    if (!raw) {
      return []
    }
    const moves = []
    const re = /(:s|swap|[a-z]+[1-9][0-9]*)/g
    let idx = 0
    while (idx < raw.length) {
      const match = re.exec(raw)
      if (!match || match.index !== idx) {
        return []
      }
      moves.push(normalizeMoveToken(match[0]))
      idx = re.lastIndex
    }
    return moves
  }

  function formatLine(moves) {
    return moves.map((move) => normalizeMoveToken(move)).join("")
  }

  function alphaLabel(index) {
    let n = Number(index)
    if (!Number.isInteger(n) || n <= 0) {
      return ""
    }
    const letters = []
    while (n > 0) {
      n -= 1
      letters.push(String.fromCharCode(97 + (n % 26)))
      n = Math.floor(n / 26)
    }
    return letters.reverse().join("")
  }

  function formatCell(col, row) {
    return `${alphaLabel(col)}${row}`
  }

  function parseCell(move) {
    const match = /^([a-z]+)([1-9][0-9]*)$/.exec(String(move || "").trim().toLowerCase())
    if (!match) {
      throw new Error(`Bad cell '${move}'`)
    }
    let col = 0
    for (const ch of match[1]) {
      col = (26 * col) + (ch.charCodeAt(0) - 96)
    }
    return {
      col,
      row: Number(match[2]),
    }
  }

  function cellIdToMove(cellId, boardSize) {
    const id = Number(cellId)
    const size = Number(boardSize)
    if (!Number.isInteger(id) || !Number.isInteger(size) || size <= 0 || id < 0 || id >= (size * size)) {
      throw new Error(`Bad cell id '${cellId}' for board size ${boardSize}`)
    }
    const row = Math.floor(id / size) + 1
    const col = (id % size) + 1
    return formatCell(col, row)
  }

  function pointKey(col, row) {
    return `${col},${row}`
  }

  function swapControlPoint(boardSize) {
    const size = Number(boardSize || 11)
    return {
      col: -2,
      row: Math.round(1 + ((2 * (size - 1)) / 3)),
    }
  }

  function lineDisplay(line, boardSize) {
    const size = Number(boardSize || 11)
    return line ? `${size},${compactMoveStreamFromLine(line)}` : String(size)
  }

  function lineParent(line) {
    const moves = parseMoves(line)
    if (moves.length === 0) {
      return ""
    }
    return formatLine(moves.slice(0, -1))
  }

  function linePrefixes(line) {
    const moves = parseMoves(line)
    const prefixes = []
    for (let i = 1; i <= moves.length; i += 1) {
      prefixes.push(formatLine(moves.slice(0, i)))
    }
    return prefixes
  }

  function lineFromCompactMoveStream(text) {
    return formatLine(parseMoves(text))
  }

  function compactMoveStreamFromLine(line) {
    return parseMoves(line).map((move) => (move === "swap" ? ":s" : move)).join("")
  }

  function rotateCell180(move, boardSize) {
    const point = parseCell(move)
    const size = Number(boardSize)
    return formatCell((size + 1) - point.col, (size + 1) - point.row)
  }

  function transformMove(move, boardSize, rotation) {
    if (normalizeMoveToken(move) === "swap") {
      return "swap"
    }
    if (Number(rotation) === 180) {
      return rotateCell180(move, boardSize)
    }
    return String(move || "").trim().toLowerCase()
  }

  function transformLine(line, boardSize, rotation) {
    return formatLine(parseMoves(line).map((move) => transformMove(move, boardSize, rotation)))
  }

  function normalizeLine(line, boardSize = null) {
    const moves = parseMoves(line)
    try {
      materializeBoardState(moves, boardSize)
    } catch (_error) {
      return ""
    }
    return formatLine(moves)
  }

  function setHashFromLine(line, { boardSize, defaultBoardSize = 11 }) {
    let hash = ""
    if (line) {
      hash = `#${Number(boardSize)},${compactMoveStreamFromLine(line)}`
    } else if (Number(boardSize) !== Number(defaultBoardSize)) {
      hash = `#${Number(boardSize)}`
    }
    const nextUrl = `${window.location.pathname}${hash}`
    const currentUrl = `${window.location.pathname}${window.location.hash}`
    if (nextUrl !== currentUrl) {
      window.history.replaceState(null, "", nextUrl)
    }
  }

  function clearHash() {
    const nextUrl = `${window.location.pathname}`
    const currentUrl = `${window.location.pathname}${window.location.hash}`
    if (nextUrl !== currentUrl) {
      window.history.replaceState(null, "", nextUrl)
    }
  }

  function parseHashState({ availableBoardSizes, defaultBoardSize = 11 }) {
    const hashText = window.location.hash ? decodeURIComponent(window.location.hash.slice(1)) : ""
    const raw = String(hashText || "").trim().toLowerCase()
    if (!raw) {
      return { boardSize: Number(defaultBoardSize), line: "", valid: true }
    }
    const match = /^([1-9][0-9]*)(?:,(.*))?$/.exec(raw)
    if (!match) {
      return { boardSize: null, line: "", valid: false }
    }
    const boardSize = Number(match[1])
    if (!availableBoardSizes.includes(boardSize)) {
      return { boardSize: null, line: "", valid: false }
    }
    const stream = String(match[2] || "")
    const line = lineFromCompactMoveStream(stream)
    if (stream && !line) {
      return { boardSize: null, line: "", valid: false }
    }
    const normalized = normalizeLine(line, boardSize)
    if (normalized !== line) {
      return { boardSize: null, line: "", valid: false }
    }
    return { boardSize, line: normalized, valid: true }
  }

  function syncLookupState({ currentLine, boardSize, nodesByLine }) {
    const displayLine = normalizeLine(String(currentLine || ""), boardSize)
    const out = {
      currentLine: displayLine,
      lookupLine: displayLine,
      displayRotation: 0,
    }
    if (!parseMoves(displayLine).length) {
      return out
    }
    const rotatedLine = transformLine(displayLine, boardSize, 180)
    if (nodesByLine instanceof Map) {
      if (nodesByLine.has(displayLine)) {
        return out
      }
      if (nodesByLine.has(rotatedLine)) {
        out.lookupLine = rotatedLine
        out.displayRotation = 180
      }
    }
    return out
  }

  function lookupLineToDisplayLine(line, { boardSize, displayRotation }) {
    return transformLine(line, boardSize, displayRotation)
  }

  function materializeBoardState(moves, boardSize = null) {
    const stones = []
    const occupied = new Map()
    for (let i = 0; i < moves.length; i += 1) {
      const move = moves[i]
      if (move === "swap") {
        if (i !== 1 || stones.length !== 1) {
          throw new Error("Illegal swap placement")
        }
        const firstStone = stones[0]
        occupied.delete(pointKey(firstStone.col, firstStone.row))
        const swappedStone = {
          ...firstStone,
          col: firstStone.row,
          row: firstStone.col,
          color: "blue",
          ply: "S",
        }
        if (
          boardSize !== null
          && (
            swappedStone.col < 1
            || swappedStone.col > Number(boardSize)
            || swappedStone.row < 1
            || swappedStone.row > Number(boardSize)
          )
        ) {
          throw new Error("Swap out of bounds")
        }
        occupied.set(pointKey(swappedStone.col, swappedStone.row), swappedStone)
        stones[0] = swappedStone
        continue
      }
      const point = parseCell(move)
      if (
        boardSize !== null
        && (point.col < 1 || point.col > Number(boardSize) || point.row < 1 || point.row > Number(boardSize))
      ) {
        throw new Error("Move out of bounds")
      }
      if (occupied.has(pointKey(point.col, point.row))) {
        throw new Error("Duplicate occupied cell")
      }
      const color = i % 2 === 0 ? "red" : "blue"
      const stone = {
        move,
        col: point.col,
        row: point.row,
        color,
        ply: i + 1,
        isLast: false,
        textColor: "",
      }
      occupied.set(pointKey(point.col, point.row), stone)
      stones.push(stone)
    }
    const lastStone = moves.length === 0 || stones.length === 0
      ? null
      : (moves[moves.length - 1] === "swap" ? stones[0] : stones[stones.length - 1])
    for (const stone of stones) {
      stone.isLast = stone === lastStone
      const base = stone.color === "red" ? RED_RGB : BLUE_RGB
      stone.textColor = rgbText(stone.isLast ? TEXT_ON_DARK_RGB : lerpRgb(base, TEXT_ON_DARK_RGB, 0.45))
    }
    return {
      moves,
      stones,
      occupied,
      toPlay: moves.length % 2 === 0 ? "red" : "blue",
    }
  }

  function buildBoardState(line) {
    return materializeBoardState(parseMoves(line))
  }

  function renderHexWorldLink(container, url, text = "View in HexWorld") {
    container.replaceChildren()
    const a = document.createElement("a")
    a.href = url
    a.target = "_blank"
    a.rel = "noopener noreferrer"
    a.textContent = text
    container.appendChild(a)
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

  function renderLineMoveList({
    container,
    currentLine,
    futureTailLines,
    setCursorLine,
  }) {
    const currentMoves = parseMoves(currentLine)
    const currentMoveCount = currentMoves.length
    const parts = [
      ...currentMoves.map((move, index) => ({
        text: move,
        isFuture: false,
        line: formatLine(currentMoves.slice(0, index + 1)),
      })),
      ...futureTailLines().map((line) => {
        const moves = parseMoves(line)
        return {
          text: moves[moves.length - 1] || "",
          isFuture: true,
          line,
        }
      }),
    ]
    renderSharedMoveList({
      container,
      parts,
      currentMoveCount,
      activateLine: (line) => {
        setCursorLine(line)
      },
    })
  }

  function setTurnStatus(element, toPlay) {
    element.textContent = `Turn: ${toPlay === "red" ? "Red" : "Blue"}`
    element.className = `turn-indicator ${toPlay === "red" ? "turn-red" : "turn-blue"}`
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

  function createBoardSvg(board) {
    const tools = createSvgTools({
      board,
      hexSize: HEX_SIZE,
      defaultFill: rgbText(OFF_WHITE_RGB),
      defaultStroke: GRID_EDGE,
      defaultStrokeWidth: "0.85",
    })

    function setupViewBox(boardSize, applyBoardDensity = null, extraPoints = null) {
      if (typeof applyBoardDensity === "function") {
        applyBoardDensity(boardSize)
      }
      const boardPixels = []
      for (let row = 1; row <= boardSize; row += 1) {
        for (let col = 1; col <= boardSize; col += 1) {
          boardPixels.push(tools.pointToPixel(col, row))
        }
      }
      const coordPixels = []
      for (let row = 1; row <= boardSize; row += 1) {
        coordPixels.push(tools.pointToPixel(0, row))
      }
      for (let col = 1; col <= boardSize; col += 1) {
        coordPixels.push(tools.pointToPixel(col, 0))
      }
      const extraPixels = Array.isArray(extraPoints)
        ? extraPoints
            .filter((point) => point && Number.isFinite(point.col) && Number.isFinite(point.row))
            .map((point) => tools.pointToPixel(point.col, point.row))
        : []
      const boardXs = boardPixels.map((point) => point[0])
      const boardYs = boardPixels.map((point) => point[1])
      const coordXs = [...coordPixels, ...extraPixels].map((point) => point[0])
      const coordYs = [...coordPixels, ...extraPixels].map((point) => point[1])
      const minX = Math.min(Math.min(...boardXs) - VIEW_PADDING, Math.min(...coordXs) - COORD_VIEW_PADDING)
      const maxX = Math.max(Math.max(...boardXs) + VIEW_PADDING, Math.max(...coordXs) + COORD_VIEW_PADDING)
      const minY = Math.min(Math.min(...boardYs) - VIEW_PADDING, Math.min(...coordYs) - COORD_VIEW_PADDING)
      const maxY = Math.max(Math.max(...boardYs) + VIEW_PADDING, Math.max(...coordYs) + COORD_VIEW_PADDING)
      board.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`)
    }

    function renderFrame(boardSize) {
      const borderRed = rgbText(RED_RGB)
      const borderBlue = rgbText(BLUE_RGB)
      const borderWidth = 4

      for (let col = 1; col <= boardSize; col += 1) {
        let cx
        let cy
        ;[cx, cy] = tools.pointToPixel(col, 1)
        let a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
        let b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
        let c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
        tools.appendLine(a[0], a[1], b[0], b[1], borderRed, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderRed, borderWidth)

        ;[cx, cy] = tools.pointToPixel(col, boardSize)
        a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
        b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
        c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
        tools.appendLine(a[0], a[1], b[0], b[1], borderRed, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderRed, borderWidth)
      }

      for (let row = 1; row <= boardSize; row += 1) {
        let cx
        let cy
        ;[cx, cy] = tools.pointToPixel(1, row)
        let a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
        let b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
        let c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
        tools.appendLine(a[0], a[1], b[0], b[1], borderBlue, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderBlue, borderWidth)

        ;[cx, cy] = tools.pointToPixel(boardSize, row)
        a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
        b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
        c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
        tools.appendLine(a[0], a[1], b[0], b[1], borderBlue, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderBlue, borderWidth)
      }

      for (let col = 1; col <= boardSize; col += 1) {
        const [cx, cy] = tools.pointToPixel(col, 0)
        tools.appendText(cx, cy, alphaLabel(col), "coord-text")
      }
      for (let row = 1; row <= boardSize; row += 1) {
        const [cx, cy] = tools.pointToPixel(0, row)
        tools.appendText(cx, cy, String(row), "coord-text")
      }
    }

    return {
      ...tools,
      setupViewBox,
      renderFrame,
    }
  }

  function renderSideActionHex({
    boardSvg,
    point,
    toPlay,
    labelText = "",
    title = "",
    onClick = null,
    fill = rgbText(OFF_WHITE_RGB),
    stroke = GRID_EDGE,
    strokeWidth = "0.85",
    className = "board-hex tenuki board-hex-face",
    primaryText = "",
    secondaryText = "",
    primaryTextClass = "cell-text",
    secondaryTextClass = "cell-stack-text",
    primaryTextFill = null,
    highlight = false,
  }) {
    if (!boardSvg || !point) {
      return null
    }
    const hoverColor = toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = toPlay === "red"
      ? `rgba(${RED_RGB[0]}, ${RED_RGB[1]}, ${RED_RGB[2]}, 0.12)`
      : `rgba(${BLUE_RGB[0]}, ${BLUE_RGB[1]}, ${BLUE_RGB[2]}, 0.12)`
    const hitClasses = ["board-hover-hit"]
    if (typeof onClick === "function") {
      hitClasses.push("clickable", "hoverable")
    }
    const hoverHex = boardSvg.appendHex(point.col, point.row, {
      fill: "transparent",
      stroke: "none",
      className: hitClasses.join(" "),
      title,
      onClick,
      size: HEX_SIZE,
    })
    hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
    const hex = boardSvg.appendHex(point.col, point.row, {
      fill,
      stroke: highlight ? hoverColor : stroke,
      strokeWidth: highlight ? "1.6" : strokeWidth,
      className,
      title,
      onClick,
    })
    if (highlight) {
      hex.polygon.style.strokeWidth = "1.6"
    }
    hex.polygon.style.setProperty("--hover-outline", hoverColor)
    if (primaryText && secondaryText) {
      boardSvg.appendStackedText(hex.cx, hex.cy, primaryText, secondaryText, primaryTextClass, secondaryTextClass)
    } else if (primaryText || secondaryText) {
      boardSvg.appendText(hex.cx, hex.cy, primaryText || secondaryText, primaryTextClass, primaryTextFill)
    }
    if (labelText) {
      boardSvg.appendText(hex.cx, hex.cy - (HEX_SIZE * 1.28), labelText, "tenuki-label")
    }
    return hex
  }

  function renderMoveTreeBoard({
    boardSvg,
    boardSize,
    currentLine,
    currentNode,
    displayRotation = 0,
    applyBoardDensity = null,
    extraViewboxPoints = null,
    childLineForCandidate,
    buildOverlay,
    candidateFill,
    overlayPrimaryText,
    overlaySecondaryText = null,
    onGoToLine,
    onGoPrevious,
    displayMoveForCandidate = null,
    displayLineForLookupLine = null,
    mirrorRootCandidates = true,
  }) {
    boardSvg.clear()
    boardSvg.setupViewBox(boardSize, applyBoardDensity, extraViewboxPoints)
    const node = currentNode || { line: "", candidates: [] }
    const board = buildBoardState(currentLine)
    const overlayByKey = new Map()
    const displayMoveAtRotation = (move, rotation) => (
      typeof displayMoveForCandidate === "function"
        ? displayMoveForCandidate(move, { boardSize, displayRotation: rotation })
        : transformMove(move, boardSize, rotation)
    )
    const displayLineAtRotation = (line, rotation) => (
      typeof displayLineForLookupLine === "function"
        ? displayLineForLookupLine(line, { boardSize, displayRotation: rotation })
        : lookupLineToDisplayLine(line, { boardSize, displayRotation: rotation })
    )

    for (const candidate of node.candidates || []) {
      if (!candidate?.retained) {
        continue
      }
      try {
        const displayMove = displayMoveAtRotation(candidate.move, displayRotation)
        const point = parseCell(displayMove)
        const lookupChildLine = childLineForCandidate(node.line, candidate)
        const childLine = lookupChildLine
          ? displayLineAtRotation(lookupChildLine, displayRotation)
          : null
        const overlay = buildOverlay({
          node,
          candidate,
          displayMove,
          lookupChildLine,
          childLine,
          col: point.col,
          boardRow: point.row,
        })
        overlayByKey.set(pointKey(point.col, point.row), overlay)
        if (mirrorRootCandidates && !node.line) {
          const mirrorMove = displayMoveAtRotation(candidate.move, 180)
          if (mirrorMove !== displayMove) {
            const mirrorPoint = parseCell(mirrorMove)
            overlayByKey.set(pointKey(mirrorPoint.col, mirrorPoint.row), buildOverlay({
              node,
              candidate,
              displayMove: mirrorMove,
              lookupChildLine,
              childLine: lookupChildLine
                ? displayLineAtRotation(lookupChildLine, 180)
                : null,
              col: mirrorPoint.col,
              boardRow: mirrorPoint.row,
            }))
          }
        }
      } catch (_error) {}
    }

    const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = board.toPlay === "red"
      ? `rgba(${RED_RGB[0]}, ${RED_RGB[1]}, ${RED_RGB[2]}, 0.12)`
      : `rgba(${BLUE_RGB[0]}, ${BLUE_RGB[1]}, ${BLUE_RGB[2]}, 0.12)`

    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize; col += 1) {
        const key = pointKey(col, row)
        const stone = board.occupied.get(key) || null
        const overlay = overlayByKey.get(key) || null
        let fill = rgbText(OFF_WHITE_RGB)
        let stroke = GRID_EDGE
        let strokeWidth = "0.85"
        let onClick = null
        let className = "board-hex board-hex-face"

        if (overlay) {
          const nextFill = candidateFill(overlay)
          fill = typeof nextFill === "string" && nextFill ? nextFill : rgbText(OFF_WHITE_RGB)
          className = overlay.className || "board-hex board-hex-face"
          stroke = overlay.stroke ?? GRID_EDGE
          strokeWidth = overlay.strokeWidth ?? "0.85"
          if (overlay.childLine) {
            onClick = () => {
              onGoToLine(overlay.childLine)
            }
          }
        }

        if (stone) {
          fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
          stroke = "none"
          if (stone.isLast) {
            onClick = () => {
              onGoPrevious()
            }
          }
        }

        const hitClasses = ["board-hover-hit"]
        if (onClick) {
          hitClasses.push("clickable")
        }
        if (overlay && overlay.childLine && !stone) {
          hitClasses.push("hoverable")
        }
        const hoverHex = boardSvg.appendHex(col, row, {
          fill: "transparent",
          stroke: "none",
          className: hitClasses.join(" "),
          size: HEX_SIZE,
          title: formatCell(col, row),
          onClick,
        })
        hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
        const hex = boardSvg.appendHex(col, row, {
          fill,
          stroke,
          strokeWidth,
          className,
        })
        hex.polygon.style.setProperty("--hover-outline", hoverColor)

        if (overlay && !stone) {
          const primaryText = String(overlayPrimaryText(overlay) || "")
          const secondaryText = typeof overlaySecondaryText === "function"
            ? String(overlaySecondaryText(overlay) || "")
            : ""
          if (primaryText && secondaryText) {
            boardSvg.appendStackedText(hex.cx, hex.cy, primaryText, secondaryText)
          } else if (primaryText || secondaryText) {
            boardSvg.appendText(hex.cx, hex.cy, primaryText || secondaryText)
          }
        }

        if (stone) {
          boardSvg.appendText(hex.cx, hex.cy, String(stone.ply), "cell-text", stone.textColor)
        }
      }
    }

    boardSvg.renderFrame(boardSize)
    return board
  }

  window.HexMoveTree = {
    GRID_EDGE,
    THEME: {
      BLUE_RGB,
      OFF_WHITE_RGB,
      RED_RGB,
      TEXT_ON_DARK_RGB,
    },
    clearHash,
    copyTextToClipboard,
    cellIdToMove,
    createBoardSvg,
    formatLine,
    lineDisplay,
    lineParent,
    linePrefixes,
    lookupLineToDisplayLine,
    normalizeLine,
    numberText,
    compactMoveStreamFromLine,
    parseHashState,
    parseMoves,
    renderHexWorldLink,
    renderLineMoveList,
    renderMoveTreeBoard,
    renderSideActionHex,
    setHashFromLine,
    setTurnStatus,
    swapControlPoint,
    syncLookupState,
    handleStandardKeydown,
  }
})()
