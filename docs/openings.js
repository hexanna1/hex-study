const HEX_SIZE = 24
const VIEW_PADDING = 34
const AVAILABLE_BOARD_SIZES = [11, 12, 13, 14, 17]
const MANIFEST_URL = "./data/openings_current.json"

const RED_RGB = [220, 60, 60]
const BLUE_RGB = [40, 100, 220]
const TEXT_ON_DARK_RGB = [250, 250, 250]
const OFF_WHITE_RGB = [246, 241, 232]
const GRID_EDGE = "rgb(182, 182, 182)"
const CANDIDATE_LOW = [244, 232, 250]
const CANDIDATE_HIGH = [170, 125, 210]
const RANDOM_CORE_IMPORTANCE_MIN = 0.95
const COORD_VIEW_PADDING = 8
const {
  buildCoreLines,
  buildDescendantCounts,
  buildRetainedLeafLines,
  createLineNavigator,
  createSvgTools,
  fractionPercent,
  formatVisits,
  lerpRgb,
  makeResultFill,
  renderMoveList: renderSharedMoveList,
  rgbText,
} = window.HexStudyUI

const elements = {
  board: document.getElementById("board"),
  status: document.getElementById("opening-status"),
  size11Btn: document.getElementById("size-11-btn"),
  size12Btn: document.getElementById("size-12-btn"),
  size13Btn: document.getElementById("size-13-btn"),
  size14Btn: document.getElementById("size-14-btn"),
  size17Btn: document.getElementById("size-17-btn"),
  resetBtn: document.getElementById("reset-btn"),
  randomBtn: document.getElementById("random-btn"),
  randomCoreBtn: document.getElementById("random-core-btn"),
  randomLeafBtn: document.getElementById("random-leaf-btn"),
  currentLine: document.getElementById("current-line"),
  moveList: document.getElementById("move-list"),
  lineMeta: document.getElementById("line-meta"),
  hexWorldLink: document.getElementById("hexworld-link"),
  metricLabel: document.getElementById("metric-label"),
}

const resultFill = makeResultFill(CANDIDATE_LOW, CANDIDATE_HIGH)
const {
  appendHex,
  appendLine,
  appendStackedText,
  appendText,
  clear: clearSvg,
  hexCorner,
  pointToPixel,
} = createSvgTools({
  board: elements.board,
  hexSize: HEX_SIZE,
  defaultFill: rgbText(OFF_WHITE_RGB),
  defaultStroke: GRID_EDGE,
  defaultStrokeWidth: "0.85",
})

const state = {
  data: null,
  dataByUrl: new Map(),
  manifestByUrl: new Map(),
  nodesByLine: new Map(),
  descendantCountsByLine: new Map(),
  currentLine: "",
  lookupLine: "",
  displayRotation: 0,
  lineHistory: [""],
  lineHistoryIndex: 0,
  dataError: null,
  isLoadingData: false,
  loadingPromise: null,
  overlayTextMode: "winrate",
  boardSize: 11,
  randomMode: "core",
}

function numberText(value) {
  return Number(value).toFixed(1)
}

function moveColor(index) {
  return index % 2 === 0 ? "red" : "blue"
}

function parseMoves(line) {
  const raw = String(line || "").trim().toLowerCase()
  if (!raw) {
    return []
  }
  const moves = []
  const re = /([a-z]+)([1-9][0-9]*)/g
  let idx = 0
  while (idx < raw.length) {
    const match = re.exec(raw)
    if (!match || match.index !== idx) {
      return []
    }
    moves.push(`${match[1]}${match[2]}`)
    idx = re.lastIndex
  }
  return moves
}

function formatLine(moves) {
  return moves.join("")
}

function formatCell(col, row) {
  return `${alphaLabel(col)}${row}`
}

function lineDisplay(line) {
  const boardSize = Number(state.data?.board_size || state.boardSize || 11)
  return line ? `${boardSize},${line}` : String(boardSize)
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
  return formatLine(parseMoves(line))
}

function rotateCell180(move, boardSize) {
  const point = parseCell(move)
  const size = Number(boardSize)
  return formatCell((size + 1) - point.col, (size + 1) - point.row)
}

function transformMove(move, boardSize, rotation) {
  if (Number(rotation) === 180) {
    return rotateCell180(move, boardSize)
  }
  return String(move || "").trim().toLowerCase()
}

function transformLine(line, boardSize, rotation) {
  return formatLine(parseMoves(line).map((move) => transformMove(move, boardSize, rotation)))
}

function lookupLineToDisplayLine(line, rotation = null) {
  const boardSize = Number(state.data?.board_size || state.boardSize || 11)
  const effectiveRotation = rotation === null ? state.displayRotation : rotation
  return transformLine(line, boardSize, effectiveRotation)
}

function syncLookupState() {
  const displayLine = normalizeLine(String(state.currentLine || ""), state.boardSize)
  state.currentLine = displayLine
  state.lookupLine = displayLine
  state.displayRotation = 0
  const moves = parseMoves(displayLine)
  if (!moves.length) {
    return
  }
  const rotatedLine = transformLine(displayLine, state.boardSize, 180)
  if (state.nodesByLine.size > 0) {
    if (state.nodesByLine.has(displayLine)) {
      return
    }
    if (state.nodesByLine.has(rotatedLine)) {
      state.lookupLine = rotatedLine
      state.displayRotation = 180
    }
  }
}

function setHashFromLine(line) {
  let hash = ""
  if (line) {
    const compactMoves = compactMoveStreamFromLine(line)
    hash = `#${state.boardSize},${compactMoves}`
  } else if (Number(state.boardSize) !== 11) {
    hash = `#${state.boardSize}`
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

function normalizeLine(line, boardSize = null) {
  const moves = parseMoves(line)
  const occupied = new Set()
  try {
    for (const move of moves) {
      const point = parseCell(move)
      if (
        boardSize !== null
        && (point.col < 1 || point.col > Number(boardSize) || point.row < 1 || point.row > Number(boardSize))
      ) {
        return ""
      }
      const key = pointKey(point.col, point.row)
      if (occupied.has(key)) {
        return ""
      }
      occupied.add(key)
    }
  } catch (_error) {
    return ""
  }
  return formatLine(moves)
}

function sanitizeLine(line) {
  const boardSize = state.data?.board_size ?? state.boardSize
  return normalizeLine(line, typeof boardSize === "number" ? Number(boardSize) : null)
}

const {
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
} = createLineNavigator({
  state,
  parseLine: parseMoves,
  linePrefixes,
  lineParent,
  sanitizeLine,
  setHashFromLine,
  render: () => render(),
})

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

function pointKey(col, row) {
  return `${col},${row}`
}

function currentBoardState() {
  const moves = parseMoves(state.currentLine)
  const stones = []
  const occupied = new Map()
  for (let i = 0; i < moves.length; i += 1) {
    const move = moves[i]
    const point = parseCell(move)
    const color = moveColor(i)
    const base = color === "red" ? RED_RGB : BLUE_RGB
    const isLast = i + 1 === moves.length
    const stone = {
      move,
      col: point.col,
      row: point.row,
      color,
      ply: i + 1,
      isLast,
      textColor: rgbText(isLast ? TEXT_ON_DARK_RGB : lerpRgb(base, TEXT_ON_DARK_RGB, 0.45)),
    }
    occupied.set(pointKey(point.col, point.row), stone)
    stones.push(stone)
  }
  return {
    moves,
    stones,
    occupied,
    toPlay: moves.length % 2 === 0 ? "red" : "blue",
  }
}

function candidateMetric(row) {
  if (state.overlayTextMode === "prior") {
    if (typeof row.prior === "number") {
      return row.prior
    }
    return null
  }
  if (typeof row.mover_winrate === "number") {
    return row.mover_winrate
  }
  return null
}

function childLineForCandidate(line, row) {
  if (!row.retained) {
    return null
  }
  return formatLine([...parseMoves(line), row.move])
}

function decodeCompactNode(rawNode) {
  const line = formatLine(parseMoves(String(rawNode?.m || "")))
  const candidates = []
  for (const row of rawNode?.c || []) {
    if (!Array.isArray(row) || row.length < 4) {
      continue
    }
    candidates.push({
      move: String(row[0] || "").trim().toLowerCase(),
      prior: typeof row[1] === "number" ? Number(row[1]) : null,
      mover_winrate: typeof row[2] === "number" ? Number(row[2]) : null,
      elo_loss: null,
      retained: Boolean(Number(row[3] || 0)),
    })
  }
  return {
    line,
    importance: typeof rawNode?.i === "number" ? Number(rawNode.i) : 0,
    candidates,
  }
}

function normalizeLoadedData(raw) {
  if (!Array.isArray(raw?.nodes)) {
    throw new Error("Unsupported opening data format")
  }
  return {
    board_size: Number(raw?.board_size || 11),
    completed: Boolean(raw?.completed),
    completed_ply: Number(raw?.completed_ply || 0),
    root_openings: Array.isArray(raw?.root_openings) ? raw.root_openings : [],
    nodes: raw.nodes.map((node) => decodeCompactNode(node)),
  }
}

function rebuildNodeMaps() {
  state.nodesByLine = new Map()
  for (const node of state.data?.nodes || []) {
    state.nodesByLine.set(String(node.line || ""), node)
  }
  state.descendantCountsByLine = buildDescendantCounts(state.nodesByLine, (node, line) => {
    const childLines = []
    for (const row of node?.candidates || []) {
      if (!row?.retained) {
        continue
      }
      const childLine = childLineForCandidate(line, row)
      if (childLine) {
        childLines.push(childLine)
      }
    }
    return childLines
  })
}

function retainedLeafLines() {
  return buildRetainedLeafLines(state.nodesByLine, (node, line) => {
    const childLines = []
    for (const row of node?.candidates || []) {
      if (!row?.retained) {
        continue
      }
      const childLine = childLineForCandidate(line, row)
      if (childLine) {
        childLines.push(childLine)
      }
    }
    return childLines
  })
}

function coreLines() {
  return buildCoreLines(state.nodesByLine, RANDOM_CORE_IMPORTANCE_MIN)
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

function applyBoardDensity(boardSize) {
  void boardSize
  elements.board.style.setProperty("--opening-cell-font-size", "12px")
  elements.board.style.setProperty("--opening-stack-font-size", "12px")
  elements.board.style.setProperty("--opening-coord-font-size", "13px")
}

function setupViewBox(boardSize) {
  applyBoardDensity(boardSize)
  const boardPixels = []
  for (let row = 1; row <= boardSize; row += 1) {
    for (let col = 1; col <= boardSize; col += 1) {
      boardPixels.push(pointToPixel(col, row))
    }
  }
  const coordPixels = []
  for (let row = 1; row <= boardSize; row += 1) {
    coordPixels.push(pointToPixel(0, row))
  }
  for (let col = 1; col <= boardSize; col += 1) {
    coordPixels.push(pointToPixel(col, 0))
  }
  const boardXs = boardPixels.map((point) => point[0])
  const boardYs = boardPixels.map((point) => point[1])
  const coordXs = coordPixels.map((point) => point[0])
  const coordYs = coordPixels.map((point) => point[1])
  const minX = Math.min(
    Math.min(...boardXs) - VIEW_PADDING,
    Math.min(...coordXs) - COORD_VIEW_PADDING,
  )
  const maxX = Math.max(
    Math.max(...boardXs) + VIEW_PADDING,
    Math.max(...coordXs) + COORD_VIEW_PADDING,
  )
  const minY = Math.min(
    Math.min(...boardYs) - VIEW_PADDING,
    Math.min(...coordYs) - COORD_VIEW_PADDING,
  )
  const maxY = Math.max(
    Math.max(...boardYs) + VIEW_PADDING,
    Math.max(...coordYs) + COORD_VIEW_PADDING,
  )
  elements.board.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`)
}

function renderBoard() {
  clearSvg()
  const boardSize = Number(state.data?.board_size || state.boardSize || 11)
  setupViewBox(boardSize)
  const node = state.nodesByLine.get(state.lookupLine) || { line: state.lookupLine, candidates: [] }
  const board = currentBoardState()
  const overlayByKey = new Map()

  for (const row of node.candidates || []) {
    if (!row?.retained) {
      continue
    }
    try {
      const displayMove = transformMove(row.move, boardSize, state.displayRotation)
      const point = parseCell(displayMove)
      const lookupChildLine = childLineForCandidate(node.line, row)
      const primaryOverlay = {
        ...row,
        move: displayMove,
        lookupChildLine,
        col: point.col,
        row: point.row,
        childLine: lookupChildLine ? lookupLineToDisplayLine(lookupChildLine) : null,
      }
      overlayByKey.set(pointKey(point.col, point.row), primaryOverlay)
      if (!node.line) {
        const mirrorMove = transformMove(displayMove, boardSize, 180)
        if (mirrorMove !== displayMove) {
          const mirrorPoint = parseCell(mirrorMove)
          overlayByKey.set(pointKey(mirrorPoint.col, mirrorPoint.row), {
            ...primaryOverlay,
            move: mirrorMove,
            col: mirrorPoint.col,
            row: mirrorPoint.row,
            childLine: lookupChildLine ? lookupLineToDisplayLine(lookupChildLine, 180) : null,
          })
        }
      }
    } catch (_error) {}
  }

  const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
  const hoverFill = board.toPlay === "red"
    ? `rgba(${RED_RGB[0]}, ${RED_RGB[1]}, ${RED_RGB[2]}, 0.12)`
    : `rgba(${BLUE_RGB[0]}, ${BLUE_RGB[1]}, ${BLUE_RGB[2]}, 0.12)`
  const borderRed = rgbText(RED_RGB)
  const borderBlue = rgbText(BLUE_RGB)

  for (let row = 1; row <= boardSize; row += 1) {
    for (let col = 1; col <= boardSize; col += 1) {
      const key = pointKey(col, row)
      const stone = board.occupied.get(key) || null
      const overlay = overlayByKey.get(key) || null
      let fill = rgbText(OFF_WHITE_RGB)
      let stroke = GRID_EDGE
      let strokeWidth = "0.85"
      let title = formatCell(col, row)
      let onClick = null
      let className = "board-hex board-hex-face"

      if (overlay) {
        const metric = candidateMetric(overlay)
        fill = typeof metric === "number" ? resultFill(metric) : rgbText(OFF_WHITE_RGB)
        const candidateClasses = ["board-hex", "candidate", "board-hex-face", "candidate-retained"]
        if (state.overlayTextMode === "prior") {
          candidateClasses.push("candidate-prior")
        }
        className = candidateClasses.join(" ")
        stroke = state.overlayTextMode === "prior" ? GRID_EDGE : "none"
        strokeWidth = "0.85"
        if (overlay.childLine) {
          onClick = () => {
            goToLine(overlay.childLine)
          }
        }
      }
      if (stone) {
        fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
        stroke = "none"
        if (stone.isLast) {
          onClick = () => {
            goPrevious()
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
      const hoverHex = appendHex(col, row, {
        fill: "transparent",
        stroke: "none",
        className: hitClasses.join(" "),
        size: HEX_SIZE,
        title,
        onClick,
      })
      hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
      const hex = appendHex(col, row, {
        fill,
        stroke,
        strokeWidth,
        className,
      })
      hex.polygon.style.setProperty("--hover-outline", hoverColor)

      if (overlay && !stone) {
        const text = overlayText(overlay)
        if (text) {
          if (state.overlayTextMode === "winrate") {
            appendStackedText(hex.cx, hex.cy, text, formatVisits(childSubtreeCount(node.line, overlay)))
          } else {
            appendText(hex.cx, hex.cy, text)
          }
        }
      }
      if (stone) {
        appendText(hex.cx, hex.cy, String(stone.ply), "cell-text", stone.textColor)
      }
    }
  }

  const borderWidth = 4
  for (let col = 1; col <= boardSize; col += 1) {
    let cx
    let cy
    ;[cx, cy] = pointToPixel(col, 1)
    let a = hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
    let b = hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
    let c = hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
    appendLine(a[0], a[1], b[0], b[1], borderRed, borderWidth)
    appendLine(b[0], b[1], c[0], c[1], borderRed, borderWidth)

    ;[cx, cy] = pointToPixel(col, boardSize)
    a = hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
    b = hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
    c = hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
    appendLine(a[0], a[1], b[0], b[1], borderRed, borderWidth)
    appendLine(b[0], b[1], c[0], c[1], borderRed, borderWidth)
  }
  for (let row = 1; row <= boardSize; row += 1) {
    let cx
    let cy
    ;[cx, cy] = pointToPixel(1, row)
    let a = hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
    let b = hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
    let c = hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
    appendLine(a[0], a[1], b[0], b[1], borderBlue, borderWidth)
    appendLine(b[0], b[1], c[0], c[1], borderBlue, borderWidth)

    ;[cx, cy] = pointToPixel(boardSize, row)
    a = hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
    b = hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
    c = hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
    appendLine(a[0], a[1], b[0], b[1], borderBlue, borderWidth)
    appendLine(b[0], b[1], c[0], c[1], borderBlue, borderWidth)
  }

  for (let col = 1; col <= boardSize; col += 1) {
    const [cx, cy] = pointToPixel(col, 0)
    appendText(cx, cy, alphaLabel(col), "coord-text")
  }
  for (let row = 1; row <= boardSize; row += 1) {
    const [cx, cy] = pointToPixel(0, row)
    appendText(cx, cy, String(row), "coord-text")
  }
}

function overlayText(row) {
  if (state.overlayTextMode === "prior") {
    if (typeof row.prior === "number") {
      return numberText(100 * row.prior)
    }
    return ""
  }
  if (typeof row.mover_winrate === "number") {
    return numberText(100 * row.mover_winrate)
  }
  return ""
}

function childSubtreeCount(line, row) {
  const childLine = row?.lookupChildLine || childLineForCandidate(line, row)
  if (!childLine) {
    return 1
  }
  return 1 + Number(state.descendantCountsByLine.get(childLine) || 0)
}

function lineMetaText(line) {
  const count = Number(state.descendantCountsByLine.get(String(state.lookupLine || line || "")) || 0)
  const base = `${formatVisits(count)} position${count === 1 ? "" : "s"} in subtree`
  if (!String(state.lookupLine || line || "")) {
    return `${base} (excl. symmetry)`
  }
  return base
}

function hexWorldUrlForLine(line) {
  const boardSize = Number(state.data?.board_size || state.boardSize || 11)
  const moves = parseMoves(line)
  return `https://hexworld.org/board/#${boardSize}nc1,${moves.join("")}`
}

function renderHexWorldLink(line) {
  elements.hexWorldLink.replaceChildren()
  const url = hexWorldUrlForLine(line)
  const a = document.createElement("a")
  a.href = url
  a.target = "_blank"
  a.rel = "noopener noreferrer"
  a.textContent = "View in HexWorld"
  elements.hexWorldLink.appendChild(a)
}

function renderMoveList() {
  const currentMoves = parseMoves(state.currentLine)
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
    container: elements.moveList,
    parts,
    currentMoveCount,
    activateLine: (line) => {
      setCursorLine(line)
    },
  })
}

function renderMetricLabel() {
  if (!elements.metricLabel) {
    return
  }
  elements.metricLabel.textContent = state.overlayTextMode === "prior" ? "priors" : "winrates"
}

function render() {
  syncLookupState()
  const board = currentBoardState()
  elements.status.textContent = `Turn: ${board.toPlay === "red" ? "Red" : "Blue"}`
  elements.status.className = `turn-indicator ${board.toPlay === "red" ? "turn-red" : "turn-blue"}`
  elements.currentLine.textContent = lineDisplay(state.currentLine)
  renderMoveList()
  renderMetricLabel()
  elements.lineMeta.textContent =
    state.dataError && !state.data
      ? `Data load failed: ${state.dataError}`
      : state.isLoadingData && !state.data
        ? "Loading opening data..."
        : lineMetaText(state.currentLine)
  renderHexWorldLink(state.currentLine)
  renderBoardSizeButtons()
  renderRandomMode()
  renderBoard()
}

function renderRandomMode() {
  const coreActive = state.randomMode === "core"
  elements.randomCoreBtn.classList.toggle("is-active", coreActive)
  elements.randomCoreBtn.setAttribute("aria-pressed", coreActive ? "true" : "false")
  elements.randomLeafBtn.classList.toggle("is-active", !coreActive)
  elements.randomLeafBtn.setAttribute("aria-pressed", !coreActive ? "true" : "false")
}

async function currentDataUrl() {
  let manifest = state.manifestByUrl.get(MANIFEST_URL)
  if (!manifest) {
    const response = await fetch(MANIFEST_URL, { cache: "no-store" })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    manifest = await response.json()
    state.manifestByUrl.set(MANIFEST_URL, manifest)
  }
  const bundle = manifest?.bundles?.[String(state.boardSize)]
  if (typeof bundle !== "string" || !bundle) {
    throw new Error(`Missing openings bundle for board size ${state.boardSize}`)
  }
  return new URL(`./data/${bundle}`, window.location.href).toString()
}

async function ensureDataLoaded() {
  if (state.loadingPromise) {
    return state.loadingPromise
  }
  state.isLoadingData = true
  state.loadingPromise = (async () => {
    const url = await currentDataUrl()
    let data = state.dataByUrl.get(url)
    if (!data) {
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      data = normalizeLoadedData(await response.json())
      state.dataByUrl.set(url, data)
    }
    state.data = data
    state.boardSize = Number(state.data?.board_size || state.boardSize || 11)
    state.currentLine = sanitizeLine(state.currentLine)
    rebuildNodeMaps()
    syncLookupState()
    resetLineHistory(state.currentLine)
    state.dataError = null
  })().catch((error) => {
    state.dataError = String(error instanceof Error ? error.message : error)
  }).finally(() => {
    state.isLoadingData = false
    state.loadingPromise = null
    render()
  })
  return state.loadingPromise
}

function renderBoardSizeButtons() {
  const buttonRows = [
    [11, elements.size11Btn],
    [12, elements.size12Btn],
    [13, elements.size13Btn],
    [14, elements.size14Btn],
    [17, elements.size17Btn],
  ]
  for (const [size, button] of buttonRows) {
    if (!(button instanceof HTMLButtonElement)) {
      continue
    }
    const active = Number(size) === Number(state.boardSize)
    button.classList.toggle("is-active", active)
    button.setAttribute("aria-pressed", active ? "true" : "false")
  }
}

function parseHashState() {
  const hashText = window.location.hash ? decodeURIComponent(window.location.hash.slice(1)) : ""
  const raw = String(hashText || "").trim().toLowerCase()
  if (!raw) {
    return { boardSize: null, line: "", valid: true }
  }
  const match = /^([1-9][0-9]*)(?:,(.*))?$/.exec(raw)
  if (match) {
    const boardSize = Number(match[1])
    if (!AVAILABLE_BOARD_SIZES.includes(boardSize)) {
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
  return { boardSize: null, line: "", valid: false }
}

async function loadBoardSize(boardSize) {
  const size = Number(boardSize)
  if (!AVAILABLE_BOARD_SIZES.includes(size) || size === Number(state.boardSize)) {
    return
  }
  state.boardSize = size
  state.currentLine = ""
  state.data = null
  state.nodesByLine = new Map()
  state.descendantCountsByLine = new Map()
  resetLineHistory("")
  setHashFromLine("")
  await ensureDataLoaded()
}

function syncFromLocationHash() {
  const parsed = parseHashState()
  if (!parsed.valid) {
    state.boardSize = 11
    state.currentLine = ""
    state.data = null
    state.nodesByLine = new Map()
    state.descendantCountsByLine = new Map()
    resetLineHistory("")
    clearHash()
    render()
    void ensureDataLoaded()
    return
  }
  const nextBoardSize = AVAILABLE_BOARD_SIZES.includes(Number(parsed.boardSize)) ? Number(parsed.boardSize) : state.boardSize
  const boardSizeChanged = nextBoardSize !== Number(state.boardSize)
  state.boardSize = nextBoardSize
  if (boardSizeChanged) {
    state.data = null
    state.nodesByLine = new Map()
    state.descendantCountsByLine = new Map()
  }
  state.currentLine = normalizeLine(String(parsed.line || "").trim().toLowerCase(), nextBoardSize)
  resetLineHistory(state.currentLine)
  render()
  if (boardSizeChanged || !state.data) {
    void ensureDataLoaded()
  }
}

async function copyCurrentLine() {
  const text = String(elements.currentLine.textContent || "").trim()
  if (!text) {
    return
  }
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(text)
    }
  } catch (_error) {}
}

async function loadRandomLine() {
  await ensureDataLoaded()
  const lines = (state.randomMode === "leaf" ? retainedLeafLines() : coreLines()).filter((line) => String(line || "") !== "")
  if (lines.length === 0) {
    return
  }
  const current = String(state.lookupLine || "")
  const choices = lines.length > 1 ? lines.filter((line) => line !== current) : lines
  const nextLine = choices[Math.floor(Math.random() * choices.length)]
  jumpToLine(lookupLineToDisplayLine(nextLine, 0))
}

elements.resetBtn.addEventListener("click", () => {
  jumpToLine("")
})
elements.randomBtn.addEventListener("click", () => {
  void loadRandomLine()
})
elements.randomCoreBtn.addEventListener("click", () => {
  state.randomMode = "core"
  renderRandomMode()
})
elements.randomLeafBtn.addEventListener("click", () => {
  state.randomMode = "leaf"
  renderRandomMode()
})
elements.size11Btn?.addEventListener("click", () => {
  void loadBoardSize(11)
})
elements.size12Btn?.addEventListener("click", () => {
  void loadBoardSize(12)
})
elements.size13Btn?.addEventListener("click", () => {
  void loadBoardSize(13)
})
elements.size14Btn?.addEventListener("click", () => {
  void loadBoardSize(14)
})
elements.size17Btn?.addEventListener("click", () => {
  void loadBoardSize(17)
})
elements.currentLine.addEventListener("click", () => {
  void copyCurrentLine()
})
window.addEventListener("hashchange", () => {
  syncFromLocationHash()
})
window.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
    return
  }
  const target = event.target
  if (target instanceof HTMLElement) {
    const tag = target.tagName.toLowerCase()
    if (tag === "input" || tag === "textarea" || target.isContentEditable) {
      return
    }
  }
  if (event.key === "t" || event.key === "T") {
    event.preventDefault()
    state.overlayTextMode = state.overlayTextMode === "winrate" ? "prior" : "winrate"
    render()
  } else if (event.key === "p" || event.key === "P") {
    event.preventDefault()
    goPrevious()
  } else if (event.key === "n" || event.key === "N") {
    event.preventDefault()
    goNext()
  } else if (event.key === "f" || event.key === "F") {
    event.preventDefault()
    goFirst()
  } else if (event.key === "l" || event.key === "L") {
    event.preventDefault()
    goLast()
  } else if (event.key === "ArrowLeft") {
    event.preventDefault()
    goPrevious()
  } else if (event.key === "ArrowRight") {
    event.preventDefault()
    goNext()
  } else if (event.key === "Backspace" || event.key === "Delete") {
    if (state.currentLine || state.lineHistoryIndex + 1 < state.lineHistory.length) {
      event.preventDefault()
      deleteFromCursor()
    }
  }
})

syncFromLocationHash()
void ensureDataLoaded()
