const AVAILABLE_BOARD_SIZES = [11, 12, 13, 14, 17]
const MANIFEST_URL = "./data/openings_current.json"
const BUNDLE_MAGIC = "HOB1"
const BUNDLE_VERSION = 1
const NULL_U16 = 65535
const HEADER_SIZE = 16
const NODE_ROW_SIZE = 10
const CANDIDATE_ROW_SIZE = 7
const CANDIDATE_LOW = [244, 232, 250]
const CANDIDATE_HIGH = [170, 125, 210]
const RANDOM_CORE_IMPORTANCE_MIN = 0.925
const {
  buildCoreLines,
  buildDescendantCounts,
  buildRetainedLeafLines,
  createLineNavigator,
  decodeThousandths,
  formatVisits,
  makeResultFill,
  rgbText,
} = window.HexStudyUI
const {
  GRID_EDGE,
  THEME,
  clearHash,
  copyTextToClipboard,
  cellIdToMove,
  createBoardSvg,
  formatLine,
  lineDisplay: lineDisplayBase,
  lineParent,
  linePrefixes,
  handleStandardKeydown,
  lookupLineToDisplayLine: lookupLineToDisplayLineBase,
  normalizeLine,
  numberText,
  parseHashState: parseHashStateBase,
  parseMoves,
  renderHexWorldLink,
  renderLineMoveList,
  renderMoveTreeBoard,
  setHashFromLine: setHashFromLineBase,
  setTurnStatus,
  syncLookupState: syncLookupStateBase,
} = window.HexMoveTree
const { BLUE_RGB, OFF_WHITE_RGB, RED_RGB } = THEME

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
const boardSvg = createBoardSvg(elements.board)

const state = {
  data: null,
  dataByUrl: new Map(),
  manifestByUrl: new Map(),
  nodesByLine: new Map(),
  descendantCountsByLine: new Map(),
  retainedLeafLines: [],
  coreLines: [],
  currentLine: "",
  lookupLine: "",
  displayRotation: 0,
  lineHistory: [""],
  lineHistoryIndex: 0,
  dataError: null,
  isLoadingData: false,
  loadingPromise: null,
  loadingBoardSize: null,
  loadAbortController: null,
  loadGeneration: 0,
  viewGeneration: 0,
  overlayTextMode: "winrate",
  boardSize: 11,
  randomMode: "core",
}

function currentBoardSize() {
  return Number(
    state.data?.board_size
    || ((!state.data && state.isLoadingData) ? state.loadingBoardSize : null)
    || state.boardSize
    || 11
  )
}

function lookupLineToDisplayLine(line, rotation = null) {
  const effectiveRotation = rotation === null ? state.displayRotation : rotation
  return lookupLineToDisplayLineBase(line, { boardSize: currentBoardSize(), displayRotation: effectiveRotation })
}

function setHashFromLine(line) {
  setHashFromLineBase(line, { boardSize: state.boardSize, defaultBoardSize: 11 })
}

function sanitizeLine(line) {
  const boardSize = state.data?.board_size ?? state.boardSize
  return normalizeLine(line, typeof boardSize === "number" ? Number(boardSize) : null)
}

function sanitizeLineForBoardSize(line, boardSize) {
  return normalizeLine(line, typeof boardSize === "number" ? Number(boardSize) : null)
}

function syncLookupState() {
  const next = syncLookupStateBase({
    currentLine: state.currentLine,
    boardSize: state.boardSize,
    nodesByLine: state.nodesByLine,
  })
  state.currentLine = next.currentLine
  state.lookupLine = next.lookupLine
  state.displayRotation = next.displayRotation
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

function candidateMetric(row) {
  if (state.overlayTextMode === "prior") {
    if (typeof row.prior === "number") {
      return row.prior
    }
    return null
  }
  if (typeof row.tree_mover_winrate === "number") {
    return row.tree_mover_winrate
  }
  return null
}

function childLineForCandidate(line, row) {
  if (!row.retained) {
    return null
  }
  return String(row.childLine || formatLine([...parseMoves(line), row.move]))
}

function appendMoveToLine(line, move) {
  const lineText = String(line || "")
  const moveText = String(move || "").trim().toLowerCase()
  return moveText ? `${lineText}${moveText}` : lineText
}

function decodeBundleParentIndex(rawParent, decodedNodesLength) {
  if (!Number.isInteger(rawParent)) {
    throw new Error("Unsupported opening node parent index")
  }
  const parentIndex = Number(rawParent)
  if (parentIndex < -1) {
    throw new Error("Unsupported opening node parent index")
  }
  if (parentIndex >= decodedNodesLength) {
    throw new Error("Opening node parent index is out of order")
  }
  return parentIndex
}

function decodeOptionalBundleMetric(rawValue) {
  if (rawValue === NULL_U16) {
    return null
  }
  return decodeThousandths(rawValue)
}

function decodeCompactNode(rawNode, rawCandidates, decodedNodes, candidateStart, boardSize) {
  const parentIndex = decodeBundleParentIndex(rawNode.parentIndex, decodedNodes.length)
  if (!Number.isInteger(rawNode.moveId)) {
    throw new Error("Unsupported opening node move")
  }
  const rawMoveId = Number(rawNode.moveId)
  const move = rawMoveId >= 0 ? cellIdToMove(rawMoveId, boardSize) : ""
  if ((parentIndex === -1 && rawMoveId !== -1) || (parentIndex >= 0 && rawMoveId < 0)) {
    throw new Error("Unsupported opening node move")
  }
  const parentLine = parentIndex >= 0 ? String(decodedNodes[parentIndex]?.line || "") : ""
  const line = parentIndex >= 0 ? appendMoveToLine(parentLine, move) : ""
  const candidateCount = Number(rawNode.candidateCount)
  if (
    !Number.isInteger(candidateCount)
    || candidateStart < 0
    || candidateCount < 0
    || candidateStart + candidateCount > rawCandidates.length
  ) {
    throw new Error("Unsupported opening candidate slice")
  }
  const candidates = []
  for (let idx = candidateStart; idx < candidateStart + candidateCount; idx += 1) {
    const row = rawCandidates[idx]
    if (!Number.isInteger(row.moveId)) {
      throw new Error("Unsupported opening candidate move")
    }
    const moveText = cellIdToMove(row.moveId, boardSize)
    if (!(row.retained === 0 || row.retained === 1)) {
      throw new Error("Unsupported opening candidate retained flag")
    }
    const retained = Number(row.retained) === 1
    candidates.push({
      move: moveText,
      childLine: retained ? appendMoveToLine(line, moveText) : null,
      prior: decodeOptionalBundleMetric(row.prior),
      tree_mover_winrate: decodeOptionalBundleMetric(row.moverWinrate),
      elo_loss: null,
      retained,
    })
  }
  return {
    line,
    parentIndex,
    move,
    importance: decodeThousandths(rawNode.importance) ?? 0,
    candidates,
  }
}

function readBundleMagic(view) {
  return String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  )
}

function normalizeLoadedData(rawBuffer) {
  if (!(rawBuffer instanceof ArrayBuffer)) {
    throw new Error("Unsupported opening data format")
  }
  if (rawBuffer.byteLength < HEADER_SIZE) {
    throw new Error("Unsupported opening data format")
  }
  const view = new DataView(rawBuffer)
  if (readBundleMagic(view) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported opening data format")
  }
  const version = view.getUint16(4, true)
  if (version !== BUNDLE_VERSION) {
    throw new Error("Unsupported opening data format")
  }
  const boardSize = view.getUint16(6, true)
  const nodeCount = view.getUint32(8, true)
  const candidateCount = view.getUint32(12, true)
  const expectedSize = HEADER_SIZE + (nodeCount * NODE_ROW_SIZE) + (candidateCount * CANDIDATE_ROW_SIZE)
  if (rawBuffer.byteLength !== expectedSize) {
    throw new Error("Opening bundle size mismatch")
  }
  const rawNodes = []
  const rawCandidates = []
  let offset = HEADER_SIZE
  for (let idx = 0; idx < nodeCount; idx += 1) {
    rawNodes.push({
      parentIndex: view.getInt32(offset, true),
      moveId: view.getInt16(offset + 4, true),
      importance: view.getUint16(offset + 6, true),
      candidateCount: view.getUint16(offset + 8, true),
    })
    offset += NODE_ROW_SIZE
  }
  for (let idx = 0; idx < candidateCount; idx += 1) {
    rawCandidates.push({
      moveId: view.getUint16(offset, true),
      retained: view.getUint8(offset + 2),
      prior: view.getUint16(offset + 3, true),
      moverWinrate: view.getUint16(offset + 5, true),
    })
    offset += CANDIDATE_ROW_SIZE
  }
  let candidateStart = 0
  const nodes = []
  for (const rawNode of rawNodes) {
    const node = decodeCompactNode(rawNode, rawCandidates, nodes, candidateStart, boardSize)
    nodes.push(node)
    candidateStart += node.candidates.length
  }
  if (candidateStart !== rawCandidates.length) {
    throw new Error("Opening candidate rows were not fully consumed")
  }
  return {
    board_size: boardSize,
    nodes,
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
  state.retainedLeafLines = buildRetainedLeafLines(state.nodesByLine, (node, line) => {
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
  }).filter((line) => String(line || "") !== "")
  state.coreLines = buildCoreLines(state.nodesByLine, RANDOM_CORE_IMPORTANCE_MIN)
    .filter((line) => String(line || "") !== "")
}

function retainedLeafLines() {
  return state.retainedLeafLines
}

function coreLines() {
  return state.coreLines
}

function applyBoardDensity(boardSize) {
  void boardSize
  elements.board.style.setProperty("--opening-cell-font-size", "12px")
  elements.board.style.setProperty("--opening-stack-font-size", "12px")
  elements.board.style.setProperty("--opening-coord-font-size", "13px")
}

function renderBoard() {
  const boardSize = currentBoardSize()
  const node = state.nodesByLine.get(state.lookupLine) || { line: state.lookupLine, candidates: [] }
  return renderMoveTreeBoard({
    boardSvg,
    boardSize,
    currentLine: state.currentLine,
    currentNode: node,
    displayRotation: state.displayRotation,
    applyBoardDensity,
    childLineForCandidate,
    buildOverlay: ({ candidate, displayMove, lookupChildLine, childLine, col, boardRow }) => ({
      ...candidate,
      move: displayMove,
      lookupChildLine,
      col,
      row: boardRow,
      childLine,
      className: [
        "board-hex",
        "candidate",
        "board-hex-face",
        "candidate-retained",
        ...(state.overlayTextMode === "prior" ? ["candidate-prior"] : []),
      ].join(" "),
      stroke: state.overlayTextMode === "prior" ? GRID_EDGE : "none",
      strokeWidth: "0.85",
    }),
    candidateFill: (overlay) => {
      const metric = candidateMetric(overlay)
      return typeof metric === "number" ? resultFill(metric) : rgbText(OFF_WHITE_RGB)
    },
    overlayPrimaryText: (overlay) => overlayText(overlay),
    overlaySecondaryText: (overlay) => (
      state.overlayTextMode === "winrate" ? formatVisits(childSubtreeCount(node.line, overlay)) : ""
    ),
    onGoToLine: (line) => {
      goToLine(line)
    },
    onGoPrevious: () => {
      goPrevious()
    },
  })
}

function overlayText(row) {
  if (state.overlayTextMode === "prior") {
    if (typeof row.prior === "number") {
      return numberText(100 * row.prior)
    }
    return ""
  }
  if (typeof row.tree_mover_winrate === "number") {
    return numberText(100 * row.tree_mover_winrate)
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

function renderMetricLabel() {
  if (!elements.metricLabel) {
    return
  }
  elements.metricLabel.textContent = state.overlayTextMode === "prior" ? "raw-NN priors" : "weighted winrates"
}

function render() {
  syncLookupState()
  const board = renderBoard()
  setTurnStatus(elements.status, board.toPlay)
  elements.currentLine.textContent = lineDisplayBase(state.currentLine, currentBoardSize())
  renderLineMoveList({
    container: elements.moveList,
    currentLine: state.currentLine,
    futureTailLines,
    setCursorLine,
  })
  renderMetricLabel()
  elements.lineMeta.textContent =
    state.dataError && !state.data
      ? `Data load failed: ${state.dataError}`
      : state.isLoadingData && !state.data
        ? "Loading opening data..."
        : lineMetaText(state.currentLine)
  renderHexWorldLink(elements.hexWorldLink, hexWorldUrlForLine(state.currentLine))
  renderBoardSizeButtons()
  renderRandomMode()
}

function renderRandomMode() {
  const coreActive = state.randomMode === "core"
  elements.randomCoreBtn.classList.toggle("is-active", coreActive)
  elements.randomCoreBtn.setAttribute("aria-pressed", coreActive ? "true" : "false")
  elements.randomLeafBtn.classList.toggle("is-active", !coreActive)
  elements.randomLeafBtn.setAttribute("aria-pressed", !coreActive ? "true" : "false")
}

async function currentDataUrl(boardSize, signal = null) {
  let manifest = state.manifestByUrl.get(MANIFEST_URL)
  if (!manifest) {
    const response = await fetch(MANIFEST_URL, { cache: "no-store", signal })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    manifest = await response.json()
    state.manifestByUrl.set(MANIFEST_URL, manifest)
  }
  const bundle = manifest?.bundles?.[String(boardSize)]
  if (typeof bundle !== "string" || !bundle) {
    throw new Error(`Missing openings bundle for board size ${boardSize}`)
  }
  return new URL(`./data/${bundle}`, window.location.href).toString()
}

async function ensureDataLoaded(boardSize = null) {
  const requestedBoardSize = Number(boardSize ?? state.boardSize ?? 11)
  if (Number(state.data?.board_size || 0) === requestedBoardSize) {
    return state.data
  }
  if (state.loadingPromise && Number(state.loadingBoardSize) === requestedBoardSize) {
    return state.loadingPromise
  }
  state.loadAbortController?.abort()
  const abortController = new AbortController()
  const loadGeneration = state.loadGeneration + 1
  state.loadGeneration = loadGeneration
  state.isLoadingData = true
  state.loadingBoardSize = requestedBoardSize
  state.loadAbortController = abortController
  render()
  state.loadingPromise = (async () => {
    const url = await currentDataUrl(requestedBoardSize, abortController.signal)
    let data = state.dataByUrl.get(url)
    if (!data) {
      const response = await fetch(url, { signal: abortController.signal })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      data = normalizeLoadedData(await response.arrayBuffer())
      state.dataByUrl.set(url, data)
    }
    if (loadGeneration !== state.loadGeneration) {
      return
    }
    state.data = data
    rebuildNodeMaps()
    state.dataError = null
    return data
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
    state.loadingBoardSize = null
    state.loadAbortController = null
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

async function loadBoardSize(boardSize) {
  const size = Number(boardSize)
  if (!AVAILABLE_BOARD_SIZES.includes(size)) {
    return
  }
  if (size === Number(state.boardSize) && state.data) {
    if (state.currentLine) {
      jumpToLine("")
    }
    return
  }
  await requestView({ boardSize: size, line: "" })
}

async function requestView({ boardSize, line = "", updateHash = true }) {
  const size = Number(boardSize)
  if (!AVAILABLE_BOARD_SIZES.includes(size)) {
    return
  }
  const requestedLine = sanitizeLineForBoardSize(String(line || "").trim().toLowerCase(), size)
  const viewGeneration = state.viewGeneration + 1
  state.viewGeneration = viewGeneration
  const data = await ensureDataLoaded(size)
  if (viewGeneration !== state.viewGeneration) {
    return
  }
  if (!data || Number(data?.board_size || 0) !== size || Number(state.data?.board_size || 0) !== size) {
    render()
    return
  }
  state.boardSize = size
  state.currentLine = requestedLine
  state.dataError = null
  syncLookupState()
  resetLineHistory(state.currentLine)
  if (updateHash) {
    setHashFromLine(state.currentLine)
  }
  render()
}

function syncFromLocationHash() {
  const parsed = parseHashStateBase({ availableBoardSizes: AVAILABLE_BOARD_SIZES, defaultBoardSize: 11 })
  if (!parsed.valid) {
    clearHash()
    void requestView({ boardSize: 11, line: "", updateHash: false })
    return
  }
  const nextBoardSize = AVAILABLE_BOARD_SIZES.includes(Number(parsed.boardSize)) ? Number(parsed.boardSize) : state.boardSize
  void requestView({
    boardSize: nextBoardSize,
    line: String(parsed.line || ""),
    updateHash: false,
  })
}

async function copyCurrentLine() {
  await copyTextToClipboard(elements.currentLine.textContent || "")
}

async function loadRandomLine() {
  const data = await ensureDataLoaded()
  if (!data) {
    render()
    return
  }
  const lines = (state.randomMode === "leaf" ? retainedLeafLines() : coreLines()).filter((line) => String(line || "") !== "")
  if (lines.length === 0) {
    render()
    return
  }
  const current = String(state.lookupLine || "")
  const choices = lines.length > 1 ? lines.filter((line) => line !== current) : lines
  const nextLine = choices[Math.floor(Math.random() * choices.length)]
  jumpToLine(lookupLineToDisplayLine(nextLine, 0))
}

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
elements.resetBtn.addEventListener("click", () => {
  jumpToLine("")
})
elements.randomBtn.addEventListener("click", () => {
  void loadRandomLine()
})
elements.currentLine.addEventListener("click", () => {
  void copyCurrentLine()
})
window.addEventListener("hashchange", () => {
  syncFromLocationHash()
})
window.addEventListener("keydown", (event) => {
  handleStandardKeydown(event, {
    toggleOverlayMode: () => {
      state.overlayTextMode = state.overlayTextMode === "winrate" ? "prior" : "winrate"
      render()
    },
    goPrevious,
    goNext,
    goFirst,
    goLast,
    canDelete: () => Boolean(state.currentLine || state.lineHistoryIndex + 1 < state.lineHistory.length),
    deleteFromCursor,
  })
})

syncFromLocationHash()
