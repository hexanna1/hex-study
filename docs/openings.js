const AVAILABLE_BOARD_SIZES = [11, 12, 13, 14, 17]
const MANIFEST_URL = "./data/openings_current.json"
const BUNDLE_MAGIC = "HOB1"
const BUNDLE_VERSION = 1
const PACKED_OPTIONAL_NULL = 1023
const HEADER_SIZE = 16
const NODE_ROW_SIZE = 1
const PACKED_NODE_COUNT_BITS = 5
const PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_COUNT_BITS
const PACKED_NODE_HAS_CHILDREN_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
const PACKED_NODE_COUNT_MASK = (1 << PACKED_NODE_COUNT_BITS) - 1
const PACKED_CANDIDATE_METRIC_BITS = 10
const {
  buildCoreLines,
  buildDescendantCounts,
  buildRetainedLeafLines,
  copyButtonText,
  createKeyedDataLoader,
  createLineNavigator,
  createModeButtonGroup,
  decodeOptionalThousandths,
  fetchArrayBuffer,
  fetchJson,
  formatVisits,
  handleStandardKeydown,
  hexWorldUrlWithCursor,
  analysisHeatFill,
  metricHeatFill,
  readAsciiMagic,
  readPackedWordAtBit: readPackedWordAtBitBase,
  rgbText,
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
  renderHexWorldLink,
  renderLineMoveList,
  renderMoveTreeBoard,
  setHashFromLine: setHashFromLineBase,
} = window.HexMoveTree
const {
  appendMoveToLine,
  cellIdToMove,
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

const overlayModeControls = createModeButtonGroup({
  state,
  field: "overlayTextMode",
  values: ["winrate", "prior"],
  rows: [
    ["winrate", elements.viewWinrateBtn],
    ["prior", elements.viewPriorBtn],
  ],
  defaultValue: "winrate",
  render: () => render(),
})

const randomModeControls = createModeButtonGroup({
  state,
  field: "randomMode",
  values: ["core", "leaf"],
  rows: [
    ["core", elements.randomCoreBtn],
    ["leaf", elements.randomLeafBtn],
  ],
  defaultValue: "core",
  render: () => renderRandomMode(),
})

function currentBoardSize() {
  return Number(
    state.data?.board_size
    || ((!state.data && state.isLoadingData) ? state.loadingBoardSize : null)
    || state.boardSize
    || 11
  )
}

function transposeMove(move) {
  const point = parseCell(move)
  return formatCell(point.row, point.col)
}

function rotate180Move(move, boardSize) {
  const point = parseCell(move)
  const size = Number(boardSize)
  return formatCell((size + 1) - point.col, (size + 1) - point.row)
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
  setHashFromLineBase(line, {
    boardSize: state.boardSize,
    defaultBoardSize: 11,
    futureLines: futureTailLines(),
  })
}

function currentLineText() {
  return lineDisplayBase(state.currentLine, currentBoardSize())
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

function decodeOptionalBundleMetric(rawValue) {
  return decodeOptionalThousandths(rawValue, PACKED_OPTIONAL_NULL)
}

function packedMoveIdBits(boardSize) {
  const size = Number(boardSize)
  if (!Number.isInteger(size) || size <= 0) {
    throw new Error("Unsupported opening board size")
  }
  return Math.floor(Math.log2((size * size) - 1)) + 1
}

function readPackedWordAtBit(view, offset, bitOffset, rowBits) {
  return readPackedWordAtBitBase(view, offset, bitOffset, rowBits, 6)
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

function decodeCompactNodes(rawNodes, view, moveLowOffset, moveHighOffset, priorLowOffset, priorHighOffset, deltaStreamOffset, exceptionOffset, boardSize, totalCandidateCount, firstPriorCount) {
  const nodes = []
  let nodeOffset = 0
  let candidateIndex = 0
  let firstPriorIndex = 0
  let dropPriorIndex = 0
  let deltaBitOffset = 0
  let exceptionIndex = 0
  const moveIdBits = packedMoveIdBits(boardSize)
  const moveHighBits = Math.max(0, moveIdBits - 8)
  const deltaBits = 8
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
    let previousPrior = null
    for (let idx = 0; idx < candidateCount; idx += 1) {
      const isSingleCandidateNode = candidateCount === 1
      const candidateRowIndex = candidateIndex + idx
      const moveLow = view.getUint8(moveLowOffset + candidateRowIndex)
      const moveHigh = moveHighBits
        ? readPackedCandidateWord(view, moveHighOffset, candidateRowIndex, moveHighBits)
        : 0
      const moveId = moveLow + (moveHigh * 256)
      if (!Number.isInteger(moveId)) {
        throw new Error("Unsupported opening candidate move")
      }
      const priorStreamIndex = idx === 0
        ? firstPriorIndex++
        : firstPriorCount + dropPriorIndex++
      const priorLow = readPackedCandidateWord(view, priorLowOffset, priorStreamIndex, 2)
      const priorHigh = view.getUint8(priorHighOffset + priorStreamIndex)
      const priorValue = priorLow + (priorHigh * 4)
      const prior = idx === 0 ? priorValue : previousPrior - priorValue
      if (!Number.isInteger(prior) || prior < 0 || prior >= (2 ** PACKED_CANDIDATE_METRIC_BITS)) {
        throw new Error("Unsupported opening candidate prior")
      }
      previousPrior = prior
      let redMetric = parentEdgeRedMetric
      if (!isSingleCandidateNode) {
        const deltaCode = readPackedWordAtBit(view, deltaStreamOffset, deltaBitOffset, deltaBits)
        deltaBitOffset += deltaBits
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
      if (rawNode.hasChildren) {
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
  return { nodes, exceptionIndex, firstPriorIndex, dropPriorIndex, deltaBitOffset }
}

function normalizeLoadedData(rawBuffer) {
  if (!(rawBuffer instanceof ArrayBuffer)) {
    throw new Error("Unsupported opening data format")
  }
  if (rawBuffer.byteLength < HEADER_SIZE) {
    throw new Error("Unsupported opening data format")
  }
  const view = new DataView(rawBuffer)
  if (readAsciiMagic(view) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported opening data format")
  }
  const version = view.getUint16(4, true)
  if (version !== BUNDLE_VERSION) {
    throw new Error("Unsupported opening data format")
  }
  const boardSize = view.getUint16(6, true)
  const nodeCount = view.getUint32(8, true)
  const candidateCount = view.getUint32(12, true)
  const moveIdBits = packedMoveIdBits(boardSize)
  const moveHighBits = Math.max(0, moveIdBits - 8)
  const moveLowOffset = HEADER_SIZE + (nodeCount * NODE_ROW_SIZE)
  const rawNodes = []
  let offset = HEADER_SIZE
  let deltaBitLength = 0
  let firstPriorCount = 0
  for (let idx = 0; idx < nodeCount; idx += 1) {
    const word = view.getUint8(offset)
    const candidateCountForNode = word & PACKED_NODE_COUNT_MASK
    rawNodes.push({
      candidateCount: candidateCountForNode,
      isCore: (Math.trunc(word / (2 ** PACKED_NODE_IS_CORE_SHIFT)) & 0x1) !== 0,
      hasChildren: (Math.trunc(word / (2 ** PACKED_NODE_HAS_CHILDREN_SHIFT)) & 0x1) !== 0,
    })
    if (candidateCountForNode > 1) {
      deltaBitLength += candidateCountForNode * 8
    }
    if (candidateCountForNode > 0) {
      firstPriorCount += 1
    }
    offset += NODE_ROW_SIZE
  }
  const moveHighOffset = moveLowOffset + candidateCount
  const priorLowOffset = moveHighOffset + Math.ceil((candidateCount * moveHighBits) / 8)
  const priorHighOffset = priorLowOffset + Math.ceil((candidateCount * 2) / 8)
  const deltaOffset = priorHighOffset + candidateCount
  const exceptionOffset = deltaOffset + Math.ceil(deltaBitLength / 8)
  const decoded = decodeCompactNodes(rawNodes, view, moveLowOffset, moveHighOffset, priorLowOffset, priorHighOffset, deltaOffset, exceptionOffset, boardSize, candidateCount, firstPriorCount)
  const usedSize = exceptionOffset + Math.ceil((decoded.exceptionIndex * PACKED_CANDIDATE_METRIC_BITS) / 8)
  if (
    decoded.firstPriorIndex !== firstPriorCount
    || decoded.dropPriorIndex !== candidateCount - firstPriorCount
    || decoded.deltaBitOffset !== deltaBitLength
    || rawBuffer.byteLength !== usedSize
  ) {
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

function renderBoard() {
  const boardSize = currentBoardSize()
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
      if (typeof winrate !== "number") {
        return rgbText(OFF_WHITE_RGB)
      }
      if (!node.line) {
        return metricHeatFill(winrate)
      }
      const count = childSubtreeCount(node.line, overlay)
      return analysisHeatFill({
        weight: count,
        topWeight: topChildSubtreeCount,
        value: winrate,
      })
    },
    overlayPrimaryText: (overlay) => overlayText(overlay),
    overlaySecondaryText: (overlay) => (
      state.overlayTextMode === "winrate" ? formatVisits(childSubtreeCount(node.line, overlay)) : ""
    ),
    onGoToLine: (line) => {
      goToLine(line)
    },
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

function lineMetaStatusText(line) {
  if (state.dataError && !state.data) {
    return `Data load failed: ${state.dataError}`
  }
  if (state.isLoadingData && !state.data) {
    return "Loading opening data..."
  }
  return lineMetaText(line)
}

function render() {
  syncLookupState()
  const board = renderBoard()
  setTurnStatus(elements.status, board.toPlay)
  setCopyButtonValue(elements.currentLine, currentLineText())
  renderLineMoveList({
    container: elements.moveList,
    currentLine: state.currentLine,
    futureTailLines,
    setCursorLine,
  })
  renderMetricLabel()
  overlayModeControls.sync()
  elements.lineMeta.textContent = lineMetaStatusText(state.currentLine)
  renderHexWorldLink(elements.hexWorldLink, hexWorldUrlForCurrentPosition())
  renderBoardSizeButtons()
  renderRandomMode()
}

function renderRandomMode() {
  randomModeControls.sync()
}

async function currentDataUrl(boardSize, signal = null) {
  let manifest = state.manifestByUrl.get(MANIFEST_URL)
  if (!manifest) {
    manifest = await fetchJson(MANIFEST_URL, { cache: "no-store", signal })
    state.manifestByUrl.set(MANIFEST_URL, manifest)
  }
  const bundle = manifest?.bundles?.[String(boardSize)]
  if (typeof bundle !== "string" || !bundle) {
    throw new Error(`Missing openings bundle for board size ${boardSize}`)
  }
  return new URL(`./data/${bundle}`, window.location.href).toString()
}

const ensureOpeningDataLoaded = createKeyedDataLoader({
  state,
  loadingKeyField: "loadingBoardSize",
  current: (boardSize) => (
    Number(state.data?.board_size || 0) === Number(boardSize) ? state.data : null
  ),
  load: async (boardSize, signal) => {
    const url = await currentDataUrl(boardSize, signal)
    let data = state.dataByUrl.get(url)
    if (!data) {
      data = normalizeLoadedData(await fetchArrayBuffer(url, { signal }))
      state.dataByUrl.set(url, data)
    }
    return data
  },
  apply: (data) => {
    state.data = data
    rebuildNodeMaps()
  },
  render: () => render(),
})

async function ensureDataLoaded(boardSize = null) {
  const requestedBoardSize = Number(boardSize ?? state.boardSize ?? 11)
  return ensureOpeningDataLoaded(requestedBoardSize)
}

function renderBoardSizeButtons() {
  syncPressedButtonGroup([
    [11, elements.size11Btn],
    [12, elements.size12Btn],
    [13, elements.size13Btn],
    [14, elements.size14Btn],
    [17, elements.size17Btn],
  ], state.boardSize, (value, current) => Number(value) === Number(current))
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

async function requestView({ boardSize, line = "", fullLine = null, updateHash = true }) {
  const size = Number(boardSize)
  if (!AVAILABLE_BOARD_SIZES.includes(size)) {
    return
  }
  const requestedLine = sanitizeLineForBoardSize(String(line || "").trim().toLowerCase(), size)
  const requestedFullLine = fullLine === null
    ? requestedLine
    : sanitizeLineForBoardSize(String(fullLine || "").trim().toLowerCase(), size)
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
  const history = ["", ...linePrefixes(requestedFullLine)]
  const historyIndex = history.indexOf(state.currentLine)
  if (historyIndex >= 0) {
    state.lineHistory = history
    state.lineHistoryIndex = historyIndex
  } else {
    resetLineHistory(state.currentLine)
  }
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
    fullLine: String(parsed.fullLine || parsed.line || ""),
    updateHash: false,
  })
}

async function copyCurrentLine() {
  await copyButtonText(elements.currentLine, currentLineText())
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
  randomModeControls.set("core")
})
elements.randomLeafBtn.addEventListener("click", () => {
  randomModeControls.set("leaf")
})
elements.viewWinrateBtn?.addEventListener("click", () => {
  overlayModeControls.set("winrate")
})
elements.viewPriorBtn?.addEventListener("click", () => {
  overlayModeControls.set("prior")
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
    toggleOverlayMode: overlayModeControls.toggle,
    goPrevious,
    goNext,
    goFirst,
    goLast,
    canDelete: () => Boolean(state.currentLine || state.lineHistoryIndex + 1 < state.lineHistory.length),
    deleteFromCursor,
  })
})

syncFromLocationHash()
