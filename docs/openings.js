const AVAILABLE_BOARD_SIZES = document.documentElement.dataset.openingBoardSizes.split(",").map(Number)
const DEFAULT_BOARD_SIZE = AVAILABLE_BOARD_SIZES[0]
const MANIFEST_URL = "./data/openings_current.json"
const BUNDLE_MAGIC = "HOD"
const PACKED_OPTIONAL_NULL = 1023
const HEADER_SIZE = 17
const NODE_ROW_SIZE = 1
const PACKED_NODE_COUNT_BITS = 6
const PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_COUNT_BITS
const PACKED_NODE_HAS_CHILDREN_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
const PACKED_NODE_COUNT_MASK = (1 << PACKED_NODE_COUNT_BITS) - 1
const PACKED_CANDIDATE_METRIC_BITS = 10
const PACKED_CANDIDATE_DELTA_BITS = 8
const PACKED_CANDIDATE_DELTA_ESCAPE = (2 ** PACKED_CANDIDATE_DELTA_BITS) - 1
const PACKED_CANDIDATE_DELTA_MAX_ABS = Math.floor((PACKED_CANDIDATE_DELTA_ESCAPE - 1) / 2)
const OPENING_INDEX_CHECKPOINT_INTERVAL = 256
const {
  copyButtonText,
  createKeyedDataLoader,
  createLineNavigator,
  createModeButtonGroup,
  decodeOptionalThousandths,
  fetchArrayBuffer,
  fetchJson,
  formatVisits,
  handlePageButtonClick,
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
  sizeButtons: [...document.querySelectorAll(".opening-size-buttons > button[data-board-size]")],
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
    || DEFAULT_BOARD_SIZE
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
    defaultBoardSize: DEFAULT_BOARD_SIZE,
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

function openingRedirectAt(data, candidateIndex) {
  const byte = data.view.getUint8(data.redirectBitmapOffset + Math.floor(candidateIndex / 8))
  return ((byte >> (candidateIndex % 8)) & 0x1) !== 0
}

function openingRedirectRankBefore(data, candidateIndex) {
  const checkpointIndex = Math.floor(candidateIndex / OPENING_INDEX_CHECKPOINT_INTERVAL)
  const checkpointCandidate = checkpointIndex * OPENING_INDEX_CHECKPOINT_INTERVAL
  let rank = data.redirectCheckpoints[checkpointIndex]
  for (let index = checkpointCandidate; index < candidateIndex; index += 1) {
    if (openingRedirectAt(data, index)) {
      rank += 1
    }
  }
  return rank
}

function openingRedirectTarget(data, redirectIndex) {
  return readPackedCandidateWord(
    data.view,
    data.redirectTargetOffset,
    redirectIndex,
    data.nodeIdBits,
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
  if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported opening data format")
  }
  const boardSize = view.getUint16(3, true)
  const nodeCount = view.getUint32(5, true)
  const candidateCount = view.getUint32(9, true)
  const redirectCount = view.getUint32(13, true)
  const moveIdBits = packedMoveIdBits(boardSize)
  const moveHighBits = Math.max(0, moveIdBits - 8)
  const nodeIdBits = Math.max(1, Math.ceil(Math.log2(Math.max(1, nodeCount))))
  const moveLowOffset = HEADER_SIZE + (nodeCount * NODE_ROW_SIZE)
  let deltaCount = 0
  let firstPriorCount = 0
  const candidateStarts = new Uint32Array(nodeCount + 1)
  let candidateTotal = 0
  for (let idx = 0; idx < nodeCount; idx += 1) {
    candidateStarts[idx] = candidateTotal
    const word = view.getUint8(HEADER_SIZE + idx)
    const candidateCountForNode = word & PACKED_NODE_COUNT_MASK
    candidateTotal += candidateCountForNode
    if (candidateCountForNode > 1) {
      deltaCount += candidateCountForNode
    }
    if (candidateCountForNode > 0) {
      firstPriorCount += 1
    }
  }
  candidateStarts[nodeCount] = candidateTotal
  const moveHighOffset = moveLowOffset + candidateCount
  const priorLowOffset = moveHighOffset + Math.ceil((candidateCount * moveHighBits) / 8)
  const priorHighOffset = priorLowOffset + Math.ceil((candidateCount * 2) / 8)
  const deltaOffset = priorHighOffset + candidateCount
  const exceptionOffset = deltaOffset + deltaCount
  const checkpointCount = Math.ceil(nodeCount / OPENING_INDEX_CHECKPOINT_INTERVAL) + 1
  const candidateCheckpoints = new Uint32Array(checkpointCount)
  const firstPriorCheckpoints = new Uint32Array(checkpointCount)
  const deltaCheckpoints = new Uint32Array(checkpointCount)
  const exceptionCheckpoints = new Uint32Array(checkpointCount)
  const subtreeNodes = new Uint32Array(nodeCount)
  let candidateCursor = 0
  let firstPriorCursor = 0
  let deltaCursor = 0
  let exceptionCursor = 0
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    if (nodeIndex % OPENING_INDEX_CHECKPOINT_INTERVAL === 0) {
      const checkpointIndex = nodeIndex / OPENING_INDEX_CHECKPOINT_INTERVAL
      candidateCheckpoints[checkpointIndex] = candidateCursor
      firstPriorCheckpoints[checkpointIndex] = firstPriorCursor
      deltaCheckpoints[checkpointIndex] = deltaCursor
      exceptionCheckpoints[checkpointIndex] = exceptionCursor
    }
    const count = view.getUint8(HEADER_SIZE + nodeIndex) & PACKED_NODE_COUNT_MASK
    candidateCursor += count
    if (count > 0) {
      firstPriorCursor += 1
    }
    if (count > 1) {
      for (let candidateIndex = 0; candidateIndex < count; candidateIndex += 1) {
        if (view.getUint8(deltaOffset + deltaCursor + candidateIndex) === PACKED_CANDIDATE_DELTA_ESCAPE) {
          exceptionCursor += 1
        }
      }
      deltaCursor += count
    }
  }
  const endCheckpoint = Math.ceil(nodeCount / OPENING_INDEX_CHECKPOINT_INTERVAL)
  candidateCheckpoints[endCheckpoint] = candidateCursor
  firstPriorCheckpoints[endCheckpoint] = firstPriorCursor
  deltaCheckpoints[endCheckpoint] = deltaCursor
  exceptionCheckpoints[endCheckpoint] = exceptionCursor
  const redirectBitmapOffset = exceptionOffset + Math.ceil((exceptionCursor * PACKED_CANDIDATE_METRIC_BITS) / 8)
  const redirectTargetOffset = redirectBitmapOffset + Math.ceil(candidateCount / 8)
  const usedSize = redirectTargetOffset + Math.ceil((redirectCount * nodeIdBits) / 8)
  if (candidateCursor !== candidateCount || firstPriorCursor !== firstPriorCount || deltaCursor !== deltaCount || rawBuffer.byteLength !== usedSize) {
    throw new Error("Opening bundle size mismatch")
  }
  const redirectCheckpointCount = Math.ceil(candidateCount / OPENING_INDEX_CHECKPOINT_INTERVAL) + 1
  const redirectCheckpoints = new Uint32Array(redirectCheckpointCount)
  let redirectCursor = 0
  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex += 1) {
    if (candidateIndex % OPENING_INDEX_CHECKPOINT_INTERVAL === 0) {
      redirectCheckpoints[candidateIndex / OPENING_INDEX_CHECKPOINT_INTERVAL] = redirectCursor
    }
    const byte = view.getUint8(redirectBitmapOffset + Math.floor(candidateIndex / 8))
    if (((byte >> (candidateIndex % 8)) & 0x1) !== 0) {
      redirectCursor += 1
    }
  }
  redirectCheckpoints[Math.ceil(candidateCount / OPENING_INDEX_CHECKPOINT_INTERVAL)] = redirectCursor
  if (redirectCursor !== redirectCount) {
    throw new Error("Opening redirect count mismatch")
  }

  const topologyData = {
    view,
    redirectBitmapOffset,
  }
  for (let nodeIndex = nodeCount - 1; nodeIndex >= 0; nodeIndex -= 1) {
    const word = view.getUint8(HEADER_SIZE + nodeIndex)
    const count = word & PACKED_NODE_COUNT_MASK
    const hasChildren = (Math.trunc(word / (2 ** PACKED_NODE_HAS_CHILDREN_SHIFT)) & 0x1) !== 0
    let subtreeSize = 1
    if (hasChildren) {
      let childIndex = nodeIndex + 1
      for (let candidateIndex = 0; candidateIndex < count; candidateIndex += 1) {
        const absoluteCandidate = candidateStarts[nodeIndex] + candidateIndex
        if (openingRedirectAt(topologyData, absoluteCandidate)) {
          continue
        }
        if (childIndex >= nodeCount || subtreeNodes[childIndex] === 0) {
          throw new Error("Opening node topology is incomplete")
        }
        subtreeSize += subtreeNodes[childIndex]
        childIndex += subtreeNodes[childIndex]
      }
    }
    subtreeNodes[nodeIndex] = subtreeSize
  }
  if (nodeCount === 0 || subtreeNodes[0] !== nodeCount) {
    throw new Error("Opening node topology is incomplete")
  }

  const data = {
    board_size: boardSize,
    view,
    nodeCount,
    candidateCount,
    redirectCount,
    nodeIdBits,
    moveHighBits,
    moveLowOffset,
    moveHighOffset,
    priorLowOffset,
    priorHighOffset,
    deltaOffset,
    exceptionOffset,
    redirectBitmapOffset,
    redirectTargetOffset,
    firstPriorCount,
    candidateCheckpoints,
    firstPriorCheckpoints,
    deltaCheckpoints,
    exceptionCheckpoints,
    redirectCheckpoints,
    candidateStarts,
    subtreeNodes,
    continuationCounts: null,
    nodeCache: new Map(),
    nodeByLine: new Map(),
    subtreeCore: null,
    subtreeLeaves: null,
    randomIndexScheduled: false,
  }
  const continuationCounts = new Uint32Array(nodeCount)
  const continuationState = new Uint8Array(nodeCount)
  function countContinuations(nodeIndex) {
    if (continuationState[nodeIndex] === 2) {
      return continuationCounts[nodeIndex]
    }
    if (continuationState[nodeIndex] === 1) {
      throw new Error("Opening redirect topology is cyclic")
    }
    continuationState[nodeIndex] = 1
    let count = view.getUint8(HEADER_SIZE + nodeIndex) & PACKED_NODE_COUNT_MASK
    for (const target of openingChildIndices(data, nodeIndex)) {
      if (target !== null) {
        if (target < 0 || target >= nodeCount) {
          throw new Error("Opening redirect topology is invalid")
        }
        count += countContinuations(target)
      }
    }
    if (!Number.isSafeInteger(count) || count > 0xFFFFFFFF) {
      throw new Error("Opening continuation count is too large")
    }
    continuationCounts[nodeIndex] = count
    continuationState[nodeIndex] = 2
    return count
  }
  countContinuations(0)
  for (const visitState of continuationState) {
    if (visitState !== 2) {
      throw new Error("Opening graph contains unreachable nodes")
    }
  }
  data.continuationCounts = continuationCounts
  return data
}

function openingNodeCursors(data, nodeIndex) {
  const checkpointIndex = Math.floor(nodeIndex / OPENING_INDEX_CHECKPOINT_INTERVAL)
  const checkpointNode = checkpointIndex * OPENING_INDEX_CHECKPOINT_INTERVAL
  let candidate = data.candidateCheckpoints[checkpointIndex]
  let firstPrior = data.firstPriorCheckpoints[checkpointIndex]
  let delta = data.deltaCheckpoints[checkpointIndex]
  let exception = data.exceptionCheckpoints[checkpointIndex]
  for (let index = checkpointNode; index < nodeIndex; index += 1) {
    const count = data.view.getUint8(HEADER_SIZE + index) & PACKED_NODE_COUNT_MASK
    candidate += count
    if (count > 0) {
      firstPrior += 1
    }
    if (count > 1) {
      for (let candidateIndex = 0; candidateIndex < count; candidateIndex += 1) {
        if (data.view.getUint8(data.deltaOffset + delta + candidateIndex) === PACKED_CANDIDATE_DELTA_ESCAPE) {
          exception += 1
        }
      }
      delta += count
    }
  }
  return { candidate, firstPrior, delta, exception }
}

function openingChildIndices(data, nodeIndex) {
  const word = data.view.getUint8(HEADER_SIZE + nodeIndex)
  const candidateCount = word & PACKED_NODE_COUNT_MASK
  const hasChildren = (Math.trunc(word / (2 ** PACKED_NODE_HAS_CHILDREN_SHIFT)) & 0x1) !== 0
  if (!hasChildren) {
    return Array(candidateCount).fill(null)
  }
  const candidateStart = data.candidateStarts[nodeIndex]
  let redirectIndex = openingRedirectRankBefore(data, candidateStart)
  let childIndex = nodeIndex + 1
  const targets = []
  for (let index = 0; index < candidateCount; index += 1) {
    const absoluteCandidate = candidateStart + index
    if (openingRedirectAt(data, absoluteCandidate)) {
      targets.push(openingRedirectTarget(data, redirectIndex))
      redirectIndex += 1
      continue
    }
    targets.push(childIndex)
    childIndex += data.subtreeNodes[childIndex]
  }
  return targets
}

function openingDescendantCount(data, nodeIndex) {
  if (!data || !Number.isInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= data.nodeCount) {
    return 0
  }
  return data.continuationCounts[nodeIndex]
}

function decodeOpeningNode(data, nodeIndex, line, parentEdgeRedMetric, ply) {
  let structure = data.nodeCache.get(nodeIndex)
  if (!structure) {
    const word = data.view.getUint8(HEADER_SIZE + nodeIndex)
    const candidateCount = word & PACKED_NODE_COUNT_MASK
    const cursors = openingNodeCursors(data, nodeIndex)
    const childIndices = openingChildIndices(data, nodeIndex)
    const dropPriorStart = data.firstPriorCount + cursors.candidate - cursors.firstPrior
    const candidates = []
    let previousPrior = null
    let exceptionIndex = cursors.exception
    for (let index = 0; index < candidateCount; index += 1) {
      const candidateIndex = cursors.candidate + index
      const moveLow = data.view.getUint8(data.moveLowOffset + candidateIndex)
      const moveHigh = data.moveHighBits
        ? readPackedCandidateWord(data.view, data.moveHighOffset, candidateIndex, data.moveHighBits)
        : 0
      const moveId = moveLow + (moveHigh * 256)
      const priorStreamIndex = index === 0 ? cursors.firstPrior : dropPriorStart + index - 1
      const priorLow = readPackedCandidateWord(data.view, data.priorLowOffset, priorStreamIndex, 2)
      const priorHigh = data.view.getUint8(data.priorHighOffset + priorStreamIndex)
      const priorValue = priorLow + (priorHigh * 4)
      const prior = index === 0 ? priorValue : previousPrior - priorValue
      if (!Number.isInteger(prior) || prior < 0 || prior >= (2 ** PACKED_CANDIDATE_METRIC_BITS)) {
        throw new Error("Unsupported opening candidate prior")
      }
      previousPrior = prior
      let redMetric = parentEdgeRedMetric
      if (candidateCount !== 1) {
        const deltaCode = data.view.getUint8(data.deltaOffset + cursors.delta + index)
        if (deltaCode === PACKED_CANDIDATE_DELTA_ESCAPE) {
          redMetric = readPackedCandidateWord(data.view, data.exceptionOffset, exceptionIndex, PACKED_CANDIDATE_METRIC_BITS)
          exceptionIndex += 1
        } else {
          if (parentEdgeRedMetric === null || parentEdgeRedMetric === PACKED_OPTIONAL_NULL) {
            throw new Error("Unsupported opening candidate metric")
          }
          redMetric = parentEdgeRedMetric + deltaCode - PACKED_CANDIDATE_DELTA_MAX_ABS
        }
      }
      candidates.push({
        move: cellIdToMove(moveId, data.board_size),
        childIndex: childIndices[index],
        redMetric,
        prior: decodeOptionalBundleMetric(prior),
        tree_mover_winrate: decodeOptionalBundleMetric(moverMetricFromRedMetric(redMetric, ply)),
        elo_loss: null,
        retained: true,
      })
    }
    structure = {
      index: nodeIndex,
      is_core: (Math.trunc(word / (2 ** PACKED_NODE_IS_CORE_SHIFT)) & 0x1) !== 0,
      candidates,
    }
    data.nodeCache.set(nodeIndex, structure)
  }
  const node = {
    ...structure,
    line,
    candidates: structure.candidates.map((candidate) => ({
      ...candidate,
      childLine: appendMoveToLine(line, candidate.move),
    })),
  }
  data.nodeByLine.set(line, node)
  return node
}

function openingNodeForLine(line, data = state.data) {
  if (!data) {
    return null
  }
  const lineText = String(line || "")
  const cached = data.nodeByLine.get(lineText)
  if (cached) {
    return cached
  }
  const moves = parseMoves(lineText)
  if (lineText && moves.length === 0) {
    return null
  }
  let node = decodeOpeningNode(data, 0, "", null, 0)
  let currentLine = ""
  let ply = 0
  for (const move of moves) {
    const candidate = node.candidates.find((row) => row.move === move)
    if (!candidate || candidate.childIndex === null) {
      return null
    }
    currentLine = candidate.childLine
    ply += 1
    node = decodeOpeningNode(data, candidate.childIndex, currentLine, candidate.redMetric, ply)
  }
  return node
}

function ensureOpeningRandomIndex(data) {
  if (!data || (data.subtreeCore && data.subtreeLeaves)) {
    return
  }
  const subtreeCore = new Uint32Array(data.nodeCount)
  const subtreeLeaves = new Uint32Array(data.nodeCount)
  const visited = new Uint8Array(data.nodeCount)
  function indexNode(nodeIndex) {
    if (visited[nodeIndex]) {
      return
    }
    visited[nodeIndex] = 1
    const word = data.view.getUint8(HEADER_SIZE + nodeIndex)
    const candidateCount = word & PACKED_NODE_COUNT_MASK
    const hasChildren = (Math.trunc(word / (2 ** PACKED_NODE_HAS_CHILDREN_SHIFT)) & 0x1) !== 0
    let coreCount = nodeIndex > 0 && (Math.trunc(word / (2 ** PACKED_NODE_IS_CORE_SHIFT)) & 0x1) !== 0 ? 1 : 0
    let leafCount = hasChildren ? 0 : Math.max(1, candidateCount)
    if (hasChildren) {
      for (const childIndex of openingChildIndices(data, nodeIndex)) {
        indexNode(childIndex)
        coreCount += subtreeCore[childIndex]
        leafCount += subtreeLeaves[childIndex]
      }
    }
    if (coreCount > 0xFFFFFFFF || leafCount > 0xFFFFFFFF) {
      throw new Error("Opening random index is too large")
    }
    subtreeCore[nodeIndex] = coreCount
    subtreeLeaves[nodeIndex] = leafCount
  }
  indexNode(0)
  data.subtreeCore = subtreeCore
  data.subtreeLeaves = subtreeLeaves
}

function scheduleOpeningRandomIndex(data) {
  if (!data || data.randomIndexScheduled || (data.subtreeCore && data.subtreeLeaves)) {
    return
  }
  data.randomIndexScheduled = true
  setTimeout(() => {
    data.randomIndexScheduled = false
    ensureOpeningRandomIndex(data)
  }, 0)
}

function openingLineForRandomRank(data, kind, rank) {
  let nodeIndex = 0
  let line = ""
  let parentEdgeRedMetric = null
  let ply = 0
  while (true) {
    const node = decodeOpeningNode(data, nodeIndex, line, parentEdgeRedMetric, ply)
    if (kind === "core" && nodeIndex > 0 && node.is_core) {
      if (rank === 0) {
        return line
      }
      rank -= 1
    }
    if (kind === "leaf" && node.candidates.length === 0) {
      return line
    }
    let selected = false
    for (const candidate of node.candidates) {
      const weight = candidate.childIndex === null
        ? (kind === "leaf" ? 1 : 0)
        : (kind === "core" ? data.subtreeCore[candidate.childIndex] : data.subtreeLeaves[candidate.childIndex])
      if (rank < weight) {
        if (candidate.childIndex === null) {
          return candidate.childLine
        }
        nodeIndex = candidate.childIndex
        line = candidate.childLine
        parentEdgeRedMetric = candidate.redMetric
        ply += 1
        selected = true
        break
      }
      rank -= weight
    }
    if (!selected) {
      throw new Error("Unsupported opening random index")
    }
  }
}

function renderBoard() {
  const boardSize = currentBoardSize()
  const node = openingNodeForLine(state.lookupLine) || { line: state.lookupLine, candidates: [] }
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
  if (!Number.isInteger(row?.childIndex)) {
    return 1
  }
  return 1 + openingDescendantCount(state.data, row.childIndex)
}

function lineMetaText(line) {
  const node = openingNodeForLine(String(state.lookupLine || line || ""))
  const count = node ? openingDescendantCount(state.data, node.index) : 0
  const base = `${formatVisits(count)} continuation${count === 1 ? "" : "s"} in subtree`
  if (!String(state.lookupLine || line || "")) {
    return `${base} (excl. symmetry)`
  }
  return base
}

function hexWorldUrlForCurrentPosition() {
  const boardSize = Number(state.data?.board_size || state.boardSize || DEFAULT_BOARD_SIZE)
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
  },
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
  if (!state.data) {
    state.boardSize = size
    renderBoardSizeButtons()
  }
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
  scheduleOpeningRandomIndex(data)
}

function syncFromLocationHash() {
  const parsed = parseHashStateBase({ availableBoardSizes: AVAILABLE_BOARD_SIZES, defaultBoardSize: DEFAULT_BOARD_SIZE })
  if (!parsed.valid) {
    clearHash()
    void requestView({ boardSize: DEFAULT_BOARD_SIZE, line: "", updateHash: false })
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
for (const button of elements.sizeButtons) {
  const boardSize = Number(button.dataset.boardSize)
  button.addEventListener("click", (event) => {
    const href = boardSize === DEFAULT_BOARD_SIZE ? "./openings.html" : `./openings.html#${boardSize}`
    handlePageButtonClick(event, href, () => void loadBoardSize(boardSize))
  })
}
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
