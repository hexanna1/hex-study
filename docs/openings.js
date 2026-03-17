const AVAILABLE_BOARD_SIZES = document.documentElement.dataset.openingBoardSizes.split(",").map(Number)
const DEFAULT_BOARD_SIZE = AVAILABLE_BOARD_SIZES[0]
const MANIFEST_URL = "./data/openings_current.json"
const {
  copyButtonText,
  createKeyedDataLoader,
  createLineNavigator,
  createModeButtonGroup,
  installSelectorTaps,
  fetchArrayBuffer,
  fetchJson,
  formatVisits,
  handlePageButtonClick,
  handleStandardKeydown,
  installMoveNavigation,
  hexWorldUrlWithCursor,
  analysisHeatFill,
  metricHeatFill,
  renderBoardLinks,
  rgbText,
  setButtonDisabled,
  setCopyButtonValue,
  setTurnStatus,
  shouldIgnoreGlobalKeydown,
  syncPressedButtonGroup,
  THEME,
} = window.HexStudyUI
const {
  GRID_EDGE,
  clearHash,
  createBoardSvg,
  lineDisplay: lineDisplayBase,
  numberText,
  parseHashState: parseHashStateBase,
  renderLineMoveList,
  renderMoveTreeBoard,
  setHashFromLine: setHashFromLineBase,
} = window.HexMoveTree
const {
  compactMoveStreamFromLine,
  formatCell,
  formatLine,
  lineParent,
  linePrefixes,
  lookupLineToDisplayLine: lookupLineToDisplayLineBase,
  normalizeLine,
  parseCell,
  parseMoves,
} = globalThis.HexPosition
const { OFF_WHITE_RGB } = THEME

const {
  normalizeLoadedData,
  openingDescendantCount,
  decodeOpeningNode,
  ensureOpeningRandomIndex,
  openingLineForRandomRank,
} = window.HexOpeningCodec

const elements = {
  board: document.getElementById("board"),
  status: document.getElementById("opening-status"),
  sizeButtons: [...document.querySelectorAll(".opening-size-buttons > button[data-board-size]")],
  resetBtn: document.getElementById("reset-btn"),
  randomBtn: document.getElementById("random-btn"),
  randomCoreBtn: document.getElementById("random-core-btn"),
  randomLeafBtn: document.getElementById("random-leaf-btn"),
  viewWinrateBtn: document.getElementById("view-winrate-btn"),
  viewPriorBtn: document.getElementById("view-prior-btn"),
  currentLine: document.getElementById("current-line"),
  moveList: document.getElementById("move-list"),
  moveNav: document.getElementById("move-nav"),
  lineMeta: document.getElementById("line-meta"),
  boardLinks: document.getElementById("board-links"),
  metricLabel: document.getElementById("metric-label"),
}

const boardSvg = createBoardSvg(elements.board)

const state = {
  data: null,
  dataByUrl: new Map(),
  manifestByUrl: new Map(),
  currentLine: "",
  lookupLine: "",
  displayRotation: 0,
  displaySwap: false,
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
  boardSize: DEFAULT_BOARD_SIZE,
  randomMode: "core",
}

const overlayModeControls = createModeButtonGroup({
  state,
  field: "overlayTextMode",
  rows: [["winrate", elements.viewWinrateBtn], ["prior", elements.viewPriorBtn]],
  render: () => render(),
})

const randomModeControls = createModeButtonGroup({
  state,
  field: "randomMode",
  rows: [["core", elements.randomCoreBtn], ["leaf", elements.randomLeafBtn]],
  render: () => renderRandomMode(),
})

function currentBoardSize() {
  return Number(
    state.data?.board_size ||
      (!state.data && state.isLoadingData ? state.loadingBoardSize : null) ||
      state.boardSize ||
      DEFAULT_BOARD_SIZE,
  )
}

function transposeMove(move) {
  const point = parseCell(move)
  return formatCell(point.row, point.col)
}

function rotate180Move(move, boardSize) {
  const point = parseCell(move)
  const size = Number(boardSize)
  return formatCell(size + 1 - point.col, size + 1 - point.row)
}

function displayMoveFromLookupMove(move, { boardSize, displayRotation, displaySwap = false, isFirstMove = false }) {
  let token = String(move || "") .trim() .toLowerCase()
  if (!token || token === "swap") { return token }
  if (displaySwap && !isFirstMove) { token = transposeMove(token) }
  if (Number(displayRotation) === 180) { token = rotate180Move(token, boardSize) }
  return token
}

function lookupMoveFromDisplayMove(move, { boardSize, displayRotation, displaySwap = false, isFirstMove = false }) {
  let token = String(move || "") .trim() .toLowerCase()
  if (!token || token === "swap") { return token }
  if (Number(displayRotation) === 180) { token = rotate180Move(token, boardSize) }
  if (displaySwap && !isFirstMove) { token = transposeMove(token) }
  return token
}

function displayLineHasSwap(line) {
  const moves = parseMoves(line)
  return moves.length >= 2 && moves[1] === "swap"
}

function displayLineFromLookupLine(line, { displayRotation, displaySwap }) {
  const boardSize = currentBoardSize()
  const lookupMoves = parseMoves(line)
  if (!displaySwap) { return lookupLineToDisplayLineBase(line, { boardSize, displayRotation }) }
  if (lookupMoves.length === 0) { return "" }
  // Virtual swap keeps the underlying opening line unchanged and inserts a
  // display-only swap token after move 1; later moves are shown transposed.
  const displayMoves = [
    displayMoveFromLookupMove(lookupMoves[0], { boardSize, displayRotation, displaySwap: false, isFirstMove: true }),
    "swap",
    ...lookupMoves
      .slice(1)
      .map((move) =>
        displayMoveFromLookupMove(move, { boardSize, displayRotation, displaySwap: true, isFirstMove: false }),
      ),
  ]
  return formatLine(displayMoves)
}

function lookupLineFromDisplayLine(line, { displayRotation, displaySwap }) {
  const boardSize = currentBoardSize()
  const displayMoves = parseMoves(line)
  if (!displaySwap) {
    return formatLine(
      displayMoves.map((move) => lookupMoveFromDisplayMove(move, { boardSize, displayRotation, displaySwap: false })),
    )
  }
  if (displayMoves.length < 2 || displayMoves[1] !== "swap") { return "" }
  const lookupMoves = [
    lookupMoveFromDisplayMove(displayMoves[0], { boardSize, displayRotation, displaySwap: false, isFirstMove: true }),
    ...displayMoves
      .slice(2)
      .map((move) =>
        lookupMoveFromDisplayMove(move, { boardSize, displayRotation, displaySwap: true, isFirstMove: false }),
      ),
  ]
  return formatLine(lookupMoves)
}

function lookupLineToDisplayLine(line, rotation = null, swap = null) {
  const effectiveRotation = rotation === null ? state.displayRotation : rotation
  const effectiveSwap = swap === null ? state.displaySwap : Boolean(swap)
  return displayLineFromLookupLine(line, { displayRotation: effectiveRotation, displaySwap: effectiveSwap })
}

function setHashFromLine(line) {
  setHashFromLineBase(line, {
    boardSize: state.boardSize,
    defaultBoardSize: DEFAULT_BOARD_SIZE,
    futureLines: futureTailLines(),
  })
}

function currentLineText() { return lineDisplayBase(state.currentLine, currentBoardSize()) }

function sanitizeLine(line) {
  const boardSize = state.data?.board_size ?? state.boardSize
  return normalizeLine(line, typeof boardSize === "number" ? Number(boardSize) : null)
}

function sanitizeLineForBoardSize(line, boardSize) {
  return normalizeLine(line, typeof boardSize === "number" ? Number(boardSize) : null)
}

function syncLookupState() {
  const displayLine = sanitizeLine(String(state.currentLine || ""))
  const displaySwap = displayLineHasSwap(displayLine)
  if (!displayLine) {
    state.currentLine = ""
    state.lookupLine = ""
    state.displayRotation = 0
    state.displaySwap = false
    return
  }
  const rotationCandidates = []
  for (const rotation of [state.displayRotation, 0, 180]) {
    if (!rotationCandidates.includes(rotation)) { rotationCandidates.push(rotation) }
  }
  // Resolve the displayed line back to the stored opening-tree line, preserving
  // the current rotation first when both orientations are plausible.
  for (const rotation of rotationCandidates) {
    const lookupLine = lookupLineFromDisplayLine(displayLine, { displayRotation: rotation, displaySwap })
    if (lookupLine && openingNodeForLine(lookupLine)) {
      state.currentLine = displayLine
      state.lookupLine = lookupLine
      state.displayRotation = rotation
      state.displaySwap = displaySwap
      return
    }
  }
  state.currentLine = displayLine
  state.lookupLine = displayLine
  state.displayRotation = 0
  state.displaySwap = false
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

function childLineForCandidate(line, row) {
  if (!row.retained) { return null }
  return String(row.childLine || formatLine([...parseMoves(line), row.move]))
}

function openingNodeForLine(line, data = state.data) {
  if (!data) { return null }
  const lineText = String(line || "")
  const cached = data.nodeByLine.get(lineText)
  if (cached) { return cached }
  const moves = parseMoves(lineText)
  if (lineText && moves.length === 0) { return null }
  let node = decodeOpeningNode(data, 0, "", null, 0)
  let currentLine = ""
  let ply = 0
  for (const move of moves) {
    const candidate = node.candidates.find((row) => row.move === move)
    if (!candidate || candidate.childIndex === null) { return null }
    currentLine = candidate.childLine
    ply += 1
    node = decodeOpeningNode(data, candidate.childIndex, currentLine, candidate.redMetric, ply)
  }
  return node
}

function scheduleOpeningRandomIndex(data) {
  if (!data || data.randomIndexScheduled || (data.subtreeCore && data.subtreeLeaves)) { return }
  data.randomIndexScheduled = true
  setTimeout(() => {
    data.randomIndexScheduled = false
    ensureOpeningRandomIndex(data)
  }, 0)
}

function renderBoard() {
  const boardSize = currentBoardSize()
  const node = openingNodeForLine(state.lookupLine) || { line: state.lookupLine, candidates: [] }
  let topChildSubtreeCount = 0
  for (const row of node.candidates || []) {
    if (!row?.retained) { continue }
    topChildSubtreeCount = Math.max(topChildSubtreeCount, childSubtreeCount(node.line, row))
  }
  return renderMoveTreeBoard({
    boardSvg,
    boardSize,
    currentLine: state.currentLine,
    currentNode: node,
    displayRotation: state.displayRotation,
    childLineForCandidate,
    displayMoveForCandidate: (move, { boardSize: nextBoardSize, displayRotation }) =>
      displayMoveFromLookupMove(move, { boardSize: nextBoardSize, displayRotation, displaySwap: state.displaySwap }),
    displayLineForLookupLine: (line, { displayRotation }) =>
      displayLineFromLookupLine(line, { displayRotation, displaySwap: state.displaySwap }),
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
        ...(state.overlayTextMode === "prior" ? ["candidate-prior"] : []),
      ].join(" "),
      stroke: state.overlayTextMode === "prior" ? GRID_EDGE : "none",
      strokeWidth: "0.85",
    }),
    candidateFill: (overlay) => {
      if (state.overlayTextMode === "prior") {
        return typeof overlay?.prior === "number" ? metricHeatFill(overlay.prior) : rgbText(OFF_WHITE_RGB)
      }
      const winrate = overlay?.tree_mover_winrate
      if (typeof winrate !== "number") { return rgbText(OFF_WHITE_RGB) }
      if (!node.line) { return metricHeatFill(winrate) }
      const count = childSubtreeCount(node.line, overlay)
      return analysisHeatFill({ weight: count, topWeight: topChildSubtreeCount, value: winrate })
    },
    overlayPrimaryText: (overlay) => overlayText(overlay),
    overlaySecondaryText: (overlay) =>
      state.overlayTextMode === "winrate" ? formatVisits(childSubtreeCount(node.line, overlay)) : "",
    onGoToLine: (line) => { goToLine(line) },
    onActivateLastStone: () => {
      const moves = parseMoves(state.currentLine)
      if (!state.displaySwap && moves.length === 1) {
        goToLine(formatLine([...moves, "swap"]))
        return
      }
      deleteFromCursor()
    },
    showMoveNumbers: true,
  })
}

function overlayText(row) {
  if (state.overlayTextMode === "prior") {
    if (typeof row.prior === "number") { return numberText(100 * row.prior) }
    return ""
  }
  if (typeof row.tree_mover_winrate === "number") { return numberText(100 * row.tree_mover_winrate) }
  return ""
}

function childSubtreeCount(line, row) {
  if (!Number.isInteger(row?.childIndex)) { return 1 }
  return 1 + openingDescendantCount(state.data, row.childIndex)
}

function lineMetaText(line) {
  const node = openingNodeForLine(String(state.lookupLine || line || ""))
  const count = node ? openingDescendantCount(state.data, node.index) : 0
  const base = `${formatVisits(count)} continuation${count === 1 ? "" : "s"} in subtree`
  if (!String(state.lookupLine || line || "")) { return `${base} (excl. symmetry)` }
  return base
}

function hexWorldUrlForCurrentPosition() {
  const boardSize = Number(state.data?.board_size || state.boardSize || DEFAULT_BOARD_SIZE)
  const base = `https://hexworld.org/board/#${boardSize}nc1`
  const past = compactMoveStreamFromLine(state.currentLine)
  const futureMoves = futureTailLines()
    .map((line) => {
      const moves = parseMoves(line)
      return moves[moves.length - 1] || ""
    })
    .filter(Boolean)
  const future = compactMoveStreamFromLine(formatLine(futureMoves))
  return hexWorldUrlWithCursor(base, past, future)
}

function renderMetricLabel() {
  if (!elements.metricLabel) { return }
  elements.metricLabel.textContent = state.overlayTextMode === "prior" ? "raw-NN priors" : "weighted winrates"
}

function lineMetaStatusText(line) {
  if (state.dataError) { return `Data load failed: ${state.dataError}` }
  if (state.isLoadingData) { return "Loading opening data..." }
  return lineMetaText(line)
}

function render() {
  const atStart = state.lineHistoryIndex <= 0 && parseMoves(state.currentLine).length === 0
  const atEnd = state.lineHistoryIndex + 1 >= state.lineHistory.length
  moveNavigation.sync({ atStart, atEnd })
  syncLookupState()
  const board = renderBoard()
  setTurnStatus(elements.status, board.toPlay)
  setCopyButtonValue(elements.currentLine, currentLineText())
  renderLineMoveList({ container: elements.moveList, currentLine: state.currentLine, futureTailLines, setCursorLine })
  renderMetricLabel()
  overlayModeControls.sync()
  elements.lineMeta.textContent = lineMetaStatusText(state.currentLine)
  renderBoardLinks(elements.boardLinks, hexWorldUrlForCurrentPosition())
  renderBoardSizeButtons()
  renderRandomMode()
}

function renderRandomMode() { randomModeControls.sync() }

async function currentDataUrl(boardSize, signal = null) {
  let manifest = state.manifestByUrl.get(MANIFEST_URL)
  if (!manifest) {
    manifest = await fetchJson(MANIFEST_URL, { cache: "no-store", signal })
    state.manifestByUrl.set(MANIFEST_URL, manifest)
  }
  const bundle = manifest?.bundles?.[String(boardSize)]
  if (typeof bundle !== "string" || !bundle) { throw new Error(`Missing openings bundle for board size ${boardSize}`) }
  return new URL(bundle, new URL(MANIFEST_URL, window.location.href)).toString()
}

const ensureOpeningDataLoaded = createKeyedDataLoader({
  state,
  loadingKeyField: "loadingBoardSize",
  current: (boardSize) => (Number(state.data?.board_size || 0) === Number(boardSize) ? state.data : null),
  load: async (boardSize, signal) => {
    const url = await currentDataUrl(boardSize, signal)
    let data = state.dataByUrl.get(url)
    if (!data) {
      data = normalizeLoadedData(await fetchArrayBuffer(url, { signal }))
      state.dataByUrl.set(url, data)
    }
    return data
  },
  apply: (data) => { state.data = data },
  render: () => render(),
})

async function ensureDataLoaded(boardSize = null) {
  const requestedBoardSize = Number(boardSize ?? state.boardSize ?? DEFAULT_BOARD_SIZE)
  return ensureOpeningDataLoaded(requestedBoardSize)
}

function renderBoardSizeButtons() {
  syncPressedButtonGroup(
    elements.sizeButtons.map((button) => [Number(button.dataset.boardSize), button]),
    state.boardSize,
    (value, current) => Number(value) === Number(current),
  )
}

async function loadBoardSize(boardSize) {
  const size = Number(boardSize)
  if (!AVAILABLE_BOARD_SIZES.includes(size)) { return }
  if (size === Number(state.boardSize) && state.data && !state.loadingPromise && !state.dataError) {
    if (state.currentLine) { jumpToLine("") }
    return
  }
  await requestView({ boardSize: size, line: "" })
}

async function requestView({ boardSize, line = "", fullLine = null, updateHash = true }) {
  const size = Number(boardSize)
  if (!AVAILABLE_BOARD_SIZES.includes(size)) { return }
  const requestedLine = sanitizeLineForBoardSize(String(line || "") .trim() .toLowerCase(), size)
  const requestedFullLine =
    fullLine === null
      ? requestedLine
      : sanitizeLineForBoardSize(
          String(fullLine || "")
            .trim()
            .toLowerCase(),
          size,
        )
  const viewGeneration = state.viewGeneration + 1
  state.viewGeneration = viewGeneration
  if (!state.data) {
    state.boardSize = size
    renderBoardSizeButtons()
  }
  const data = await ensureDataLoaded(size)
  if (viewGeneration !== state.viewGeneration) { return }
  if (!data || Number(data?.board_size || 0) !== size || Number(state.data?.board_size || 0) !== size) {
    render()
    return
  }
  state.boardSize = size
  state.currentLine = requestedLine
  state.dataError = null
  syncLookupState()
  const history = ["", ...linePrefixes(requestedFullLine)]
  const historyIndex = history.indexOf(state.currentLine)
  if (historyIndex >= 0) {
    state.lineHistory = history
    state.lineHistoryIndex = historyIndex
  } else { resetLineHistory(state.currentLine) }
  if (updateHash) { setHashFromLine(state.currentLine) }
  render()
  scheduleOpeningRandomIndex(data)
}

function syncFromLocationHash() {
  const parsed = parseHashStateBase({
    availableBoardSizes: AVAILABLE_BOARD_SIZES,
    defaultBoardSize: DEFAULT_BOARD_SIZE,
  })
  if (!parsed.valid) {
    clearHash()
    void requestView({ boardSize: DEFAULT_BOARD_SIZE, line: "", updateHash: false })
    return
  }
  const nextBoardSize = AVAILABLE_BOARD_SIZES.includes(Number(parsed.boardSize))
    ? Number(parsed.boardSize)
    : state.boardSize
  void requestView({
    boardSize: nextBoardSize,
    line: String(parsed.line || ""),
    fullLine: String(parsed.fullLine || parsed.line || ""),
    updateHash: false,
  })
}

async function copyCurrentLine() { await copyButtonText(elements.currentLine, currentLineText()) }

function handleSwapShortcut(event) {
  if (shouldIgnoreGlobalKeydown(event)) { return false }
  if (!(event.key === "s" || event.key === "S")) { return false }
  const moves = parseMoves(state.currentLine)
  if (!state.displaySwap && moves.length === 1) {
    event.preventDefault()
    goToLine(formatLine([...moves, "swap"]))
    return true
  }
  if (state.displaySwap && moves.length === 2 && moves[1] === "swap") {
    event.preventDefault()
    goPrevious()
    return true
  }
  return false
}

async function loadRandomLine() {
  const data = await ensureDataLoaded()
  if (!data) {
    render()
    return
  }
  ensureOpeningRandomIndex(data)
  const kind = state.randomMode === "leaf" ? "leaf" : "core"
  const total = kind === "leaf" ? data.subtreeLeaves[0] : data.subtreeCore[0]
  if (total === 0) {
    render()
    return
  }
  const current = String(state.lookupLine || "")
  let nextLine
  do {
    const rank = Math.floor(Math.random() * total)
    nextLine = openingLineForRandomRank(data, kind, rank)
  } while (total > 1 && nextLine === current)
  jumpToLine(lookupLineToDisplayLine(nextLine, 0, false))
}

const moveNavigation = installMoveNavigation({
  container: elements.moveNav,
  goFirst,
  goPrevious,
  goNext,
  goLast,
  setDisabled: setButtonDisabled,
})
elements.randomCoreBtn.addEventListener("click", () => { randomModeControls.set("core") })
elements.randomLeafBtn.addEventListener("click", () => { randomModeControls.set("leaf") })
elements.viewWinrateBtn?.addEventListener("click", () => { overlayModeControls.set("winrate") })
elements.viewPriorBtn?.addEventListener("click", () => { overlayModeControls.set("prior") })
for (const button of elements.sizeButtons) {
  const boardSize = Number(button.dataset.boardSize)
  button.addEventListener("click", (event) => {
    const href = boardSize === DEFAULT_BOARD_SIZE ? "./openings.html" : `./openings.html#${boardSize}`
    handlePageButtonClick(event, href, () => {
      overlayModeControls.set("winrate")
      void loadBoardSize(boardSize)
    })
  })
}
elements.resetBtn.addEventListener("click", () => { jumpToLine("") })
elements.randomBtn.addEventListener("click", () => { void loadRandomLine() })
elements.currentLine.addEventListener("click", () => { void copyCurrentLine() })
window.addEventListener("hashchange", () => { syncFromLocationHash() })
window.addEventListener("keydown", (event) => {
  if (handleSwapShortcut(event)) { return }
  handleStandardKeydown(event, {
    newGame: () => jumpToLine(""),
    toggleOverlayMode: overlayModeControls.toggle,
    goPrevious,
    goNext,
    goFirst,
    goLast,
    canDelete: () => Boolean(state.currentLine || state.lineHistoryIndex + 1 < state.lineHistory.length),
    deleteFromCursor,
  })
})

installSelectorTaps()
syncFromLocationHash()
