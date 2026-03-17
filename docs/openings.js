const AVAILABLE_BOARD_SIZES = [11, 12, 13, 14, 17]
const MANIFEST_URL = "./data/openings_current.json"
const BUNDLE_MAGIC = "HOB1"
const BUNDLE_VERSION = 1
const PACKED_OPTIONAL_NULL = 1023
const HEADER_SIZE = 16
const NODE_ROW_SIZE = 1
const PACKED_NODE_COUNT_BITS = 5
const PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_COUNT_BITS
const PACKED_NODE_COUNT_MASK = (1 << PACKED_NODE_COUNT_BITS) - 1
const PACKED_CANDIDATE_METRIC_BITS = 10
const CANDIDATE_LOW = [244, 232, 250]
const CANDIDATE_HIGH = [170, 125, 210]
const {
  buildCoreLines,
  buildDescendantCounts,
  buildRetainedLeafLines,
  createLineNavigator,
  decodeThousandths,
  formatVisits,
  hexWorldUrlWithCursor,
  hexataCandidateFill,
  makeResultFill,
  rgbText,
  shouldIgnoreGlobalKeydown,
} = window.HexStudyUI
const {
  GRID_EDGE,
  THEME,
  clearHash,
  compactMoveStreamFromLine,
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
  renderSideActionHex,
  setHashFromLine: setHashFromLineBase,
  setTurnStatus,
  swapControlPoint,
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
  viewWinrateBtn: document.getElementById("view-winrate-btn"),
  viewPriorBtn: document.getElementById("view-prior-btn"),
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

function formatCellMove(col, row) {
  let n = Number(col)
  let letters = ""
  while (n > 0) {
    n -= 1
    letters = String.fromCharCode(97 + (n % 26)) + letters
    n = Math.floor(n / 26)
  }
  return `${letters}${Number(row)}`
}

function parseCellMove(move) {
  const match = /^([a-z]+)([1-9][0-9]*)$/.exec(String(move || "").trim().toLowerCase())
  if (!match) {
    throw new Error(`Bad cell '${move}'`)
  }
  let col = 0
  for (const ch of match[1]) {
    col = (26 * col) + (ch.charCodeAt(0) - 96)
  }
  return { col, row: Number(match[2]) }
}

function transposeMove(move) {
  const point = parseCellMove(move)
  return formatCellMove(point.row, point.col)
}

function rotate180Move(move, boardSize) {
  const point = parseCellMove(move)
  const size = Number(boardSize)
  return formatCellMove((size + 1) - point.col, (size + 1) - point.row)
}

function displayMoveFromLookupMove(move, { boardSize, displayRotation, displaySwap = false, isFirstMove = false }) {
  let token = String(move || "").trim().toLowerCase()
  if (!token || token === "swap") {
    return token
  }
  if (displaySwap && !isFirstMove) {
    token = transposeMove(token)
  }
  if (Number(displayRotation) === 180) {
    token = rotate180Move(token, boardSize)
  }
  return token
}

function lookupMoveFromDisplayMove(move, { boardSize, displayRotation, displaySwap = false, isFirstMove = false }) {
  let token = String(move || "").trim().toLowerCase()
  if (!token || token === "swap") {
    return token
  }
  if (Number(displayRotation) === 180) {
    token = rotate180Move(token, boardSize)
  }
  if (displaySwap && !isFirstMove) {
    token = transposeMove(token)
  }
  return token
}

function displayLineHasSwap(line) {
  const moves = parseMoves(line)
  return moves.length >= 2 && moves[1] === "swap"
}

function displayLineFromLookupLine(line, { displayRotation, displaySwap }) {
  const boardSize = currentBoardSize()
  const lookupMoves = parseMoves(line)
  if (!displaySwap) {
    return lookupLineToDisplayLineBase(line, { boardSize, displayRotation })
  }
  if (lookupMoves.length === 0) {
    return ""
  }
  // Virtual swap keeps the underlying opening line unchanged and inserts a
  // display-only swap token after move 1; later moves are shown transposed.
  const displayMoves = [
    displayMoveFromLookupMove(lookupMoves[0], {
      boardSize,
      displayRotation,
      displaySwap: false,
      isFirstMove: true,
    }),
    "swap",
    ...lookupMoves.slice(1).map((move) => displayMoveFromLookupMove(move, {
      boardSize,
      displayRotation,
      displaySwap: true,
      isFirstMove: false,
    })),
  ]
  return formatLine(displayMoves)
}

function lookupLineFromDisplayLine(line, { displayRotation, displaySwap }) {
  const boardSize = currentBoardSize()
  const displayMoves = parseMoves(line)
  if (!displaySwap) {
    return formatLine(displayMoves.map((move) => lookupMoveFromDisplayMove(move, {
      boardSize,
      displayRotation,
      displaySwap: false,
    })))
  }
  if (displayMoves.length < 2 || displayMoves[1] !== "swap") {
    return ""
  }
  const lookupMoves = [
    lookupMoveFromDisplayMove(displayMoves[0], {
      boardSize,
      displayRotation,
      displaySwap: false,
      isFirstMove: true,
    }),
    ...displayMoves.slice(2).map((move) => lookupMoveFromDisplayMove(move, {
      boardSize,
      displayRotation,
      displaySwap: true,
      isFirstMove: false,
    })),
  ]
  return formatLine(lookupMoves)
}

function lookupLineToDisplayLine(line, rotation = null, swap = null) {
  const effectiveRotation = rotation === null ? state.displayRotation : rotation
  const effectiveSwap = swap === null ? state.displaySwap : Boolean(swap)
  return displayLineFromLookupLine(line, {
    displayRotation: effectiveRotation,
    displaySwap: effectiveSwap,
  })
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
    if (!rotationCandidates.includes(rotation)) {
      rotationCandidates.push(rotation)
    }
  }
  // Resolve the displayed line back to the stored opening-tree line, preserving
  // the current rotation first when both orientations are plausible.
  for (const rotation of rotationCandidates) {
    const lookupLine = lookupLineFromDisplayLine(displayLine, {
      displayRotation: rotation,
      displaySwap,
    })
    if (lookupLine && state.nodesByLine.has(lookupLine)) {
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

function decodeOptionalBundleMetric(rawValue) {
  if (rawValue === PACKED_OPTIONAL_NULL) {
    return null
  }
  return decodeThousandths(rawValue)
}

function packedMoveIdBits(boardSize) {
  const size = Number(boardSize)
  if (!Number.isInteger(size) || size <= 0) {
    throw new Error("Unsupported opening board size")
  }
  return Math.floor(Math.log2((size * size) - 1)) + 1
}

function candidateRowBits(boardSize) {
  return packedMoveIdBits(boardSize) + PACKED_CANDIDATE_METRIC_BITS + 7 + 1
}

function singleCandidateRowBits(boardSize) {
  return packedMoveIdBits(boardSize) + PACKED_CANDIDATE_METRIC_BITS + 1
}

function readPackedWordAtBit(view, offset, bitOffset, rowBits) {
  const byteOffset = offset + Math.trunc(bitOffset / 8)
  const shift = bitOffset % 8
  let chunk = 0
  for (let idx = 0; idx < 6 && byteOffset + idx < view.byteLength; idx += 1) {
    chunk += view.getUint8(byteOffset + idx) * (2 ** (8 * idx))
  }
  return Math.trunc(chunk / (2 ** shift)) & ((2 ** rowBits) - 1)
}

function readPackedCandidateWord(view, offset, candidateIndex, rowBits) {
  return readPackedWordAtBit(view, offset, candidateIndex * rowBits, rowBits)
}

function moverMetricFromRedMetric(redMetric, ply) {
  if (redMetric === PACKED_OPTIONAL_NULL || redMetric === null) {
    return PACKED_OPTIONAL_NULL
  }
  return Number(ply) % 2 === 0 ? redMetric : 1000 - redMetric
}

function decodeCompactNodes(rawNodes, view, candidateStreamOffset, exceptionOffset, boardSize, totalCandidateCount) {
  const nodes = []
  let nodeOffset = 0
  let candidateIndex = 0
  let candidateBitOffset = 0
  let exceptionIndex = 0
  const moveIdBits = packedMoveIdBits(boardSize)
  const normalRowBits = candidateRowBits(boardSize)
  const singleRowBits = singleCandidateRowBits(boardSize)
  const deltaBits = 7
  const deltaEscape = (2 ** deltaBits) - 1
  const deltaMaxAbs = Math.floor((deltaEscape - 1) / 2)

  function readExceptionMetric() {
    const value = readPackedCandidateWord(view, exceptionOffset, exceptionIndex, PACKED_CANDIDATE_METRIC_BITS)
    exceptionIndex += 1
    return value
  }

  function decodeNode(line, parentIndex, move, parentEdgeRedMetric, ply) {
    if (nodeOffset >= rawNodes.length) {
      throw new Error("Opening node topology is incomplete")
    }
    const rawNode = rawNodes[nodeOffset]
    const nodeIndex = nodeOffset
    nodeOffset += 1
    const candidateCount = Number(rawNode.candidateCount)
    if (
      !Number.isInteger(candidateCount)
      || candidateIndex < 0
      || candidateCount < 0
      || candidateIndex + candidateCount > totalCandidateCount
    ) {
      throw new Error("Unsupported opening candidate slice")
    }

    const candidates = []
    const childLines = []
    for (let idx = 0; idx < candidateCount; idx += 1) {
      const isSingleCandidateNode = candidateCount === 1
      const rowBits = isSingleCandidateNode ? singleRowBits : normalRowBits
      const word = readPackedWordAtBit(view, candidateStreamOffset, candidateBitOffset, rowBits)
      candidateBitOffset += rowBits
      const moveId = word & ((2 ** moveIdBits) - 1)
      if (!Number.isInteger(moveId)) {
        throw new Error("Unsupported opening candidate move")
      }
      const prior = Math.trunc(word / (2 ** moveIdBits)) & 0x3FF
      let hasChild = false
      let redMetric = parentEdgeRedMetric
      if (isSingleCandidateNode) {
        hasChild = (Math.trunc(word / (2 ** (moveIdBits + PACKED_CANDIDATE_METRIC_BITS))) & 1) !== 0
      } else {
        const deltaCode = Math.trunc(word / (2 ** (moveIdBits + PACKED_CANDIDATE_METRIC_BITS))) & deltaEscape
        hasChild = (Math.trunc(word / (2 ** (moveIdBits + PACKED_CANDIDATE_METRIC_BITS + deltaBits))) & 1) !== 0
        if (deltaCode === deltaEscape || parentEdgeRedMetric === null || parentEdgeRedMetric === PACKED_OPTIONAL_NULL) {
          redMetric = readExceptionMetric()
        } else {
          redMetric = parentEdgeRedMetric + deltaCode - deltaMaxAbs
        }
      }
      const moveText = cellIdToMove(moveId, boardSize)
      const childLine = appendMoveToLine(line, moveText)
      candidates.push({
        move: moveText,
        childLine,
        prior: decodeOptionalBundleMetric(prior),
        tree_mover_winrate: decodeOptionalBundleMetric(moverMetricFromRedMetric(redMetric, ply)),
        elo_loss: null,
        retained: true,
      })
      if (hasChild) {
        childLines.push([childLine, moveText, redMetric])
      }
    }
    candidateIndex += candidateCount

    nodes.push({
      line,
      parentIndex,
      move,
      is_core: Boolean(rawNode.isCore),
      candidates,
    })
    for (const [childLine, childMove, childRedMetric] of childLines) {
      decodeNode(childLine, nodeIndex, childMove, childRedMetric, Number(ply) + 1)
    }
  }

  decodeNode("", -1, "", null, 0)
  if (nodeOffset !== rawNodes.length) {
    throw new Error("Opening node rows were not fully consumed")
  }
  if (candidateIndex !== totalCandidateCount) {
    throw new Error("Opening candidate rows were not fully consumed")
  }
  return { nodes, exceptionIndex, candidateBitOffset }
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
  const rowBits = candidateRowBits(boardSize)
  const singleRowBits = singleCandidateRowBits(boardSize)
  const candidateOffset = HEADER_SIZE + (nodeCount * NODE_ROW_SIZE)
  const rawNodes = []
  let offset = HEADER_SIZE
  let candidateBitLength = 0
  for (let idx = 0; idx < nodeCount; idx += 1) {
    const word = view.getUint8(offset)
    const candidateCountForNode = word & PACKED_NODE_COUNT_MASK
    rawNodes.push({
      candidateCount: candidateCountForNode,
      isCore: Math.trunc(word / (2 ** PACKED_NODE_IS_CORE_SHIFT)) !== 0,
    })
    candidateBitLength += candidateCountForNode * (candidateCountForNode === 1 ? singleRowBits : rowBits)
    offset += NODE_ROW_SIZE
  }
  const exceptionOffset = candidateOffset + Math.ceil(candidateBitLength / 8)
  const decoded = decodeCompactNodes(rawNodes, view, candidateOffset, exceptionOffset, boardSize, candidateCount)
  const usedSize = exceptionOffset + Math.ceil((decoded.exceptionIndex * PACKED_CANDIDATE_METRIC_BITS) / 8)
  if (decoded.candidateBitOffset !== candidateBitLength || rawBuffer.byteLength !== usedSize) {
    throw new Error("Opening bundle size mismatch")
  }
  return {
    board_size: boardSize,
    nodes: decoded.nodes,
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
  state.coreLines = buildCoreLines(state.nodesByLine)
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
  const swapPoint = swapControlPoint(boardSize)
  const node = state.nodesByLine.get(state.lookupLine) || { line: state.lookupLine, candidates: [] }
  let topChildSubtreeCount = 0
  for (const row of node.candidates || []) {
    if (!row?.retained) {
      continue
    }
    topChildSubtreeCount = Math.max(topChildSubtreeCount, childSubtreeCount(node.line, row))
  }
  return renderMoveTreeBoard({
    boardSvg,
    boardSize,
    currentLine: state.currentLine,
    currentNode: node,
    displayRotation: state.displayRotation,
    applyBoardDensity,
    extraViewboxPoints: [swapPoint],
    childLineForCandidate,
    displayMoveForCandidate: (move, { boardSize: nextBoardSize, displayRotation }) => (
      displayMoveFromLookupMove(move, {
        boardSize: nextBoardSize,
        displayRotation,
        displaySwap: state.displaySwap,
      })
    ),
    displayLineForLookupLine: (line, { displayRotation }) => (
      displayLineFromLookupLine(line, {
        displayRotation,
        displaySwap: state.displaySwap,
      })
    ),
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
      if (state.overlayTextMode === "prior") {
        return typeof overlay?.prior === "number" ? resultFill(overlay.prior) : rgbText(OFF_WHITE_RGB)
      }
      const winrate = overlay?.tree_mover_winrate
      if (typeof winrate !== "number") {
        return rgbText(OFF_WHITE_RGB)
      }
      if (!node.line) {
        return resultFill(winrate)
      }
      const count = childSubtreeCount(node.line, overlay)
      return hexataCandidateFill(CANDIDATE_LOW, CANDIDATE_HIGH, count, topChildSubtreeCount, winrate)
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

function renderSwapHex(board) {
  const swapPoint = swapControlPoint(currentBoardSize())
  const moves = parseMoves(state.currentLine)
  if (state.displaySwap || moves.length !== 1) {
    return
  }
  renderSideActionHex({
    boardSvg,
    point: swapPoint,
    toPlay: board.toPlay,
    labelText: "Swap",
    onClick: () => {
      goToLine(formatLine([...moves, "swap"]))
    },
    fill: rgbText(OFF_WHITE_RGB),
    stroke: GRID_EDGE,
    strokeWidth: "0.85",
    className: "board-hex tenuki board-hex-face",
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

function hexWorldUrlForCurrentPosition() {
  const boardSize = Number(state.data?.board_size || state.boardSize || 11)
  const base = `https://hexworld.org/board/#${boardSize}nc1`
  const past = compactMoveStreamFromLine(state.currentLine)
  const futureMoves = futureTailLines().map((line) => {
    const moves = parseMoves(line)
    return moves[moves.length - 1] || ""
  }).filter(Boolean)
  const future = compactMoveStreamFromLine(formatLine(futureMoves))
  return hexWorldUrlWithCursor(base, past, future)
}

function renderMetricLabel() {
  if (!elements.metricLabel) {
    return
  }
  elements.metricLabel.textContent = state.overlayTextMode === "prior" ? "raw-NN priors" : "weighted winrates"
}

function renderOverlayModeToggle() {
  const winrateActive = state.overlayTextMode === "winrate"
  elements.viewWinrateBtn?.classList.toggle("is-active", winrateActive)
  elements.viewWinrateBtn?.setAttribute("aria-pressed", winrateActive ? "true" : "false")
  elements.viewPriorBtn?.classList.toggle("is-active", !winrateActive)
  elements.viewPriorBtn?.setAttribute("aria-pressed", !winrateActive ? "true" : "false")
}

function setOverlayTextMode(mode) {
  const nextMode = mode === "prior" ? "prior" : "winrate"
  if (state.overlayTextMode === nextMode) {
    return
  }
  state.overlayTextMode = nextMode
  render()
}

function render() {
  syncLookupState()
  const board = renderBoard()
  renderSwapHex(board)
  setTurnStatus(elements.status, board.toPlay)
  elements.currentLine.textContent = lineDisplayBase(state.currentLine, currentBoardSize())
  renderLineMoveList({
    container: elements.moveList,
    currentLine: state.currentLine,
    futureTailLines,
    setCursorLine,
  })
  renderMetricLabel()
  renderOverlayModeToggle()
  elements.lineMeta.textContent =
    state.dataError && !state.data
      ? `Data load failed: ${state.dataError}`
      : state.isLoadingData && !state.data
        ? "Loading opening data..."
        : lineMetaText(state.currentLine)
  renderHexWorldLink(elements.hexWorldLink, hexWorldUrlForCurrentPosition())
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

function handleSwapShortcut(event) {
  if (shouldIgnoreGlobalKeydown(event)) {
    return false
  }
  if (!(event.key === "s" || event.key === "S")) {
    return false
  }
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
  const lines = (state.randomMode === "leaf" ? retainedLeafLines() : coreLines()).filter((line) => String(line || "") !== "")
  if (lines.length === 0) {
    render()
    return
  }
  const current = String(state.lookupLine || "")
  const choices = lines.length > 1 ? lines.filter((line) => line !== current) : lines
  const nextLine = choices[Math.floor(Math.random() * choices.length)]
  jumpToLine(lookupLineToDisplayLine(nextLine, 0, false))
}

elements.randomCoreBtn.addEventListener("click", () => {
  state.randomMode = "core"
  renderRandomMode()
})
elements.randomLeafBtn.addEventListener("click", () => {
  state.randomMode = "leaf"
  renderRandomMode()
})
elements.viewWinrateBtn?.addEventListener("click", () => {
  setOverlayTextMode("winrate")
})
elements.viewPriorBtn?.addEventListener("click", () => {
  setOverlayTextMode("prior")
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
  if (handleSwapShortcut(event)) {
    return
  }
  handleStandardKeydown(event, {
    toggleOverlayMode: () => {
      setOverlayTextMode(state.overlayTextMode === "winrate" ? "prior" : "winrate")
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
