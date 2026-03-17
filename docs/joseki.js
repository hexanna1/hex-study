const MANIFEST_URL = "./data/joseki_current.json"
const FAMILY_LABELS = Object.fromEntries(
  document.documentElement.dataset.josekiFamilies.split(",").map((entry) => entry.split(":")),
)
const AVAILABLE_FAMILIES = Object.keys(FAMILY_LABELS)
const DEFAULT_FAMILY = AVAILABLE_FAMILIES[0]
const BUNDLE_MAGIC = "HJB"
const FAMILY_CODE_TO_NAME = {
  1: "A",
  2: "O",
}
const HEADER_SIZE = 17
const JOSEKI_INDEX_CHECKPOINT_INTERVAL = 256
const PACKED_NODE_LOCAL_COUNT_BITS = 4
const PACKED_NODE_LOCAL_COUNT_MASK = (1 << PACKED_NODE_LOCAL_COUNT_BITS) - 1
const PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_LOCAL_COUNT_BITS
const PACKED_NODE_TENUKI_RETAINED_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
const PACKED_NODE_TENUKI_CHILD_SHIFT = PACKED_NODE_TENUKI_RETAINED_SHIFT + 1
const PACKED_NODE_LOCAL_CHILDREN_SHIFT = PACKED_NODE_TENUKI_CHILD_SHIFT + 1
const PACKED_TENUKI_DROP_HIGH_BITS = 2
const HEX_SIZE = 26
const VIEW_PADDING = 40
const LOCAL_BOARD_SIZE = 10
const LOCAL_DELTA_MAX = {
  A: 127,
  O: 64,
}
const TENUKI_POINT = { col: -1, row: Math.round((2 * LOCAL_BOARD_SIZE) / 3) }

const {
  copyButtonText,
  createKeyedDataLoader,
  createLineNavigator,
  createModeButtonGroup,
  createSvgTools,
  decodeThousandths,
  fetchArrayBuffer,
  fetchJson,
  formatVisits,
  handlePageButtonClick,
  handleStandardKeydown,
  hexWorldUrlWithCursor,
  analysisHeatFill,
  lerpRgb,
  percentText,
  readAsciiMagic,
  readPackedWordAtBit,
  replaceHash,
  renderExternalLink,
  renderMoveList: renderSharedMoveList,
  rgbText,
  decodeLocationHash,
  setCopyButtonValue,
  setTurnStatus,
  setSvgViewBoxFromPixels,
  shouldIgnoreGlobalKeydown,
  syncPressedButtonGroup,
  turnRgbaText,
  THEME,
} = window.HexStudyUI
const {
  BLUE_RGB,
  GRID_EDGE,
  OFF_WHITE_RGB,
  RED_RGB,
  TEXT_ON_DARK_RGB,
} = THEME

const elements = {
  board: document.getElementById("board"),
  status: document.getElementById("joseki-status"),
  familyButtons: [...document.querySelectorAll("button[data-family]")],
  randomCoreBtn: document.getElementById("random-core-btn"),
  randomLeafBtn: document.getElementById("random-leaf-btn"),
  resetBtn: document.getElementById("reset-btn"),
  randomBtn: document.getElementById("random-btn"),
  currentLine: document.getElementById("current-line"),
  moveList: document.getElementById("move-list"),
  lineMeta: document.getElementById("line-meta"),
  hexWorldLink: document.getElementById("hexworld-link"),
}

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
  defaultStrokeWidth: "0.75",
})

const state = {
  data: null,
  dataByUrl: new Map(),
  manifestByUrl: new Map(),
  currentLine: "",
  family: DEFAULT_FAMILY,
  dataError: null,
  isLoadingData: false,
  loadingPromise: null,
  loadingFamily: null,
  loadAbortController: null,
  loadGeneration: 0,
  viewGeneration: 0,
  randomMode: "core",
  lineHistory: [""],
  lineHistoryIndex: 0,
}

function localDelta(local, family) {
  const dq = Number(local[0]) - 1
  const dr = Number(local[1]) - 1
  if (family === "O") {
    return (dq * dq) - (dq * dr) + (dr * dr)
  }
  return (dq * dq) + (dq * dr) + (dr * dr)
}

function normalizedFamily(family) {
  const familyValue = String(family || "").trim().toUpperCase()
  if (!AVAILABLE_FAMILIES.includes(familyValue)) {
    return null
  }
  return familyValue
}

function parseLineForFamily(line, family, { validateRegion = false, rejectDuplicates = false } = {}) {
  const familyValue = normalizedFamily(family)
  if (!familyValue) {
    return null
  }
  const raw = String(line || "").replace(/\s+/g, "")
  if (!raw) {
    return null
  }
  const match = /^([AO])\[(.*)\]$/.exec(raw)
  if (!match || match[1] !== familyValue) {
    return null
  }
  const entries = []
  const occupied = new Set()
  for (const token of match[2].split(":")) {
    if (token === "") {
      entries.push(null)
      continue
    }
    const parts = /^([1-9][0-9]*),([1-9][0-9]*)$/.exec(token)
    if (!parts) {
      return null
    }
    const entry = [Number(parts[1]), Number(parts[2])]
    if (validateRegion) {
      const point = localToDisplay(entry, familyValue)
      if (!displayCellInLocalRegion(point.col, point.row, familyValue)) {
        return null
      }
    }
    const key = `${entry[0]},${entry[1]}`
    if (rejectDuplicates && occupied.has(key)) {
      return null
    }
    occupied.add(key)
    entries.push(entry)
  }
  return { family: familyValue, entries }
}

function parseEntries(line, family) {
  return parseLineForFamily(line, family)?.entries || []
}

function normalizeLineForFamily(line, family) {
  const parsed = parseLineForFamily(line, family, {
    validateRegion: true,
    rejectDuplicates: true,
  })
  if (!parsed || parsed.entries.length === 0) {
    return ""
  }
  return formatLine(parsed.family, parsed.entries)
}

function inferFamilyFromLine(line) {
  const match = /^([AO])\[/.exec(String(line || "").trim())
  return match ? match[1] : null
}

function rootFamilyFromLine(line) {
  const match = /^([AO])$/.exec(String(line || "").trim().toUpperCase())
  return match ? match[1] : null
}

function currentFamily() {
  return (
    inferFamilyFromLine(state.currentLine)
    || ((!state.data && state.isLoadingData) ? state.loadingFamily : null)
    || state.family
    || DEFAULT_FAMILY
  )
}

function normalizeRequestedLineForFamily(line, family) {
  const raw = String(line || "").trim()
  if (!raw) {
    return ""
  }
  if (rootFamilyFromLine(raw)) {
    return ""
  }
  return normalizeLineForFamily(raw, family)
}

async function currentDataUrl(family = currentFamily(), signal = null) {
  let manifest = state.manifestByUrl.get(MANIFEST_URL)
  if (!manifest) {
    manifest = await fetchJson(MANIFEST_URL, { cache: "no-store", signal })
    state.manifestByUrl.set(MANIFEST_URL, manifest)
  }
  const bundle = manifest?.bundles?.[family]
  if (typeof bundle !== "string" || !bundle) {
    throw new Error(`Missing joseki bundle for family ${family}`)
  }
  return new URL(bundle, new URL(MANIFEST_URL, window.location.href)).toString()
}

function decodeLocalMove(rawMove) {
  if (!Number.isInteger(rawMove) || rawMove < 0 || rawMove >= 100) {
    throw new Error("Unsupported joseki local move")
  }
  return [Math.floor(rawMove / 10) + 1, (rawMove % 10) + 1]
}

function readBitplaneValue(view, offset, valueCount, valueIndex, bits) {
  let value = 0
  for (let bit = 0; bit < bits; bit += 1) {
    value = (value * 2) + readPackedWordAtBit(view, offset, (bit * valueCount) + valueIndex, 1)
  }
  return value
}

function josekiRedirectAt(data, edgeIndex) {
  const byte = data.view.getUint8(data.redirectBitmapOffset + Math.floor(edgeIndex / 8))
  return ((byte >> (edgeIndex % 8)) & 0x1) !== 0
}

function josekiRedirectRankBefore(data, edgeIndex) {
  const checkpointIndex = Math.floor(edgeIndex / JOSEKI_INDEX_CHECKPOINT_INTERVAL)
  const checkpointEdge = checkpointIndex * JOSEKI_INDEX_CHECKPOINT_INTERVAL
  let rank = data.redirectCheckpoints[checkpointIndex]
  for (let index = checkpointEdge; index < edgeIndex; index += 1) {
    if (josekiRedirectAt(data, index)) {
      rank += 1
    }
  }
  return rank
}

function josekiRedirectTarget(data, redirectIndex) {
  return readPackedWordAtBit(
    data.view,
    data.redirectTargetOffset,
    redirectIndex * data.nodeIdBits,
    data.nodeIdBits,
  )
}

function josekiChildIndices(data, nodeIndex) {
  const rawNode = data.rawNodes[nodeIndex]
  const edgeStart = data.edgeStarts[nodeIndex]
  let redirectIndex = josekiRedirectRankBefore(data, edgeStart)
  let childIndex = nodeIndex + 1
  const targets = []
  const childPresence = [
    ...Array(rawNode.localCount).fill(rawNode.localChildren),
    ...(rawNode.tenukiRetained ? [rawNode.tenukiChildPresent] : []),
  ]
  for (let index = 0; index < childPresence.length; index += 1) {
    const edgeIndex = edgeStart + index
    if (josekiRedirectAt(data, edgeIndex)) {
      targets.push(josekiRedirectTarget(data, redirectIndex))
      redirectIndex += 1
    } else if (childPresence[index]) {
      targets.push(childIndex)
      childIndex += data.subtreeNodes[childIndex]
    } else {
      targets.push(null)
    }
  }
  return targets
}

function normalizeLoadedData(rawBuffer) {
  if (!(rawBuffer instanceof ArrayBuffer)) {
    throw new Error("Unsupported joseki data format")
  }
  if (rawBuffer.byteLength < HEADER_SIZE) {
    throw new Error("Unsupported joseki data format")
  }
  const view = new DataView(rawBuffer)
  if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported joseki data format")
  }
  const family = FAMILY_CODE_TO_NAME[view.getUint8(3)]
  if (!AVAILABLE_FAMILIES.includes(family)) {
    throw new Error("Unsupported joseki data format")
  }
  const boardSize = view.getUint8(4)
  const nodeCount = view.getUint32(5, true)
  const localRowCount = view.getUint32(9, true)
  const redirectCount = view.getUint32(13, true)
  const nodeControlOffset = HEADER_SIZE
  const tenukiDropLowOffset = nodeControlOffset + nodeCount
  const tenukiDropHighOffset = tenukiDropLowOffset + nodeCount
  const localMoveOffset = tenukiDropHighOffset + Math.ceil((nodeCount * PACKED_TENUKI_DROP_HIGH_BITS) / 8)
  const firstLocalDropOffset = localMoveOffset + localRowCount
  const rawNodes = []
  const localStarts = new Uint32Array(nodeCount + 1)
  const firstLocalStarts = new Uint32Array(nodeCount + 1)
  const edgeStarts = new Uint32Array(nodeCount + 1)
  let firstLocalDropCount = 0
  let localTotal = 0
  let edgeTotal = 0
  for (let idx = 0; idx < nodeCount; idx += 1) {
    localStarts[idx] = localTotal
    firstLocalStarts[idx] = firstLocalDropCount
    edgeStarts[idx] = edgeTotal
    const word = view.getUint8(nodeControlOffset + idx)
    const tenukiDrop = view.getUint8(tenukiDropLowOffset + idx) + (readBitplaneValue(
      view,
      tenukiDropHighOffset,
      nodeCount,
      idx,
      PACKED_TENUKI_DROP_HIGH_BITS,
    ) * 256)
    const tenukiPresent = idx > 0
    if (!tenukiPresent && tenukiDrop !== 0) {
      throw new Error("Unsupported joseki root tenuki drop")
    }
    const tenukiStoneFraction = tenukiPresent ? 1000 - tenukiDrop : 0
    if (tenukiStoneFraction < 0 || tenukiStoneFraction > 1000) {
      throw new Error("Unsupported joseki tenuki drop")
    }
    const localCount = word & PACKED_NODE_LOCAL_COUNT_MASK
    if (localCount > 0) {
      firstLocalDropCount += 1
    }
    const rawNode = {
      localCount,
      isCore: (Math.trunc(word / (2 ** PACKED_NODE_IS_CORE_SHIFT)) & 0x1) !== 0,
      tenukiRetained: (Math.trunc(word / (2 ** PACKED_NODE_TENUKI_RETAINED_SHIFT)) & 0x1) !== 0,
      tenukiPresent,
      tenukiChildPresent: (Math.trunc(word / (2 ** PACKED_NODE_TENUKI_CHILD_SHIFT)) & 0x1) !== 0,
      tenukiStoneFraction,
      localChildren: (Math.trunc(word / (2 ** PACKED_NODE_LOCAL_CHILDREN_SHIFT)) & 0x1) !== 0,
    }
    rawNodes.push(rawNode)
    localTotal += localCount
    edgeTotal += localCount + (rawNode.tenukiRetained ? 1 : 0)
  }
  localStarts[nodeCount] = localTotal
  firstLocalStarts[nodeCount] = firstLocalDropCount
  edgeStarts[nodeCount] = edgeTotal
  const siblingLocalDropOffset = firstLocalDropOffset + firstLocalDropCount
  const redirectBitmapOffset = siblingLocalDropOffset + localRowCount - firstLocalDropCount
  const redirectTargetOffset = redirectBitmapOffset + Math.ceil(edgeTotal / 8)
  const nodeIdBits = Math.max(1, Math.ceil(Math.log2(Math.max(1, nodeCount))))
  const expectedSize = redirectTargetOffset + Math.ceil((redirectCount * nodeIdBits) / 8)
  if (localTotal !== localRowCount || rawBuffer.byteLength !== expectedSize) {
    throw new Error("Joseki bundle size mismatch")
  }
  const redirectCheckpoints = new Uint32Array(
    Math.ceil(edgeTotal / JOSEKI_INDEX_CHECKPOINT_INTERVAL) + 1,
  )
  const topologyData = { view, redirectBitmapOffset }
  let redirectCursor = 0
  for (let edgeIndex = 0; edgeIndex < edgeTotal; edgeIndex += 1) {
    if (edgeIndex % JOSEKI_INDEX_CHECKPOINT_INTERVAL === 0) {
      redirectCheckpoints[edgeIndex / JOSEKI_INDEX_CHECKPOINT_INTERVAL] = redirectCursor
    }
    if (josekiRedirectAt(topologyData, edgeIndex)) {
      redirectCursor += 1
    }
  }
  redirectCheckpoints[Math.ceil(edgeTotal / JOSEKI_INDEX_CHECKPOINT_INTERVAL)] = redirectCursor
  if (redirectCursor !== redirectCount) {
    throw new Error("Joseki redirect count mismatch")
  }

  const subtreeNodes = new Uint32Array(nodeCount)
  const data = {
    family,
    board_size: Number(boardSize || 19),
    view,
    nodeCount,
    nodeIdBits,
    rawNodes,
    localStarts,
    firstLocalStarts,
    edgeStarts,
    localMoveOffset,
    firstLocalDropOffset,
    siblingLocalDropOffset,
    redirectBitmapOffset,
    redirectTargetOffset,
    redirectCheckpoints,
    subtreeNodes,
    continuationCounts: null,
    nodeCache: new Map(),
    nodeByLine: new Map(),
    subtreeCore: null,
    subtreeCoreLocal: null,
    subtreeLeaves: null,
    subtreeLeavesLocal: null,
    randomIndexScheduled: false,
  }
  for (let nodeIndex = nodeCount - 1; nodeIndex >= 0; nodeIndex -= 1) {
    const rawNode = rawNodes[nodeIndex]
    let subtreeSize = 1
    let childIndex = nodeIndex + 1
    const childPresence = [
      ...Array(rawNode.localCount).fill(rawNode.localChildren),
      ...(rawNode.tenukiRetained ? [rawNode.tenukiChildPresent] : []),
    ]
    for (let edgeOffset = 0; edgeOffset < childPresence.length; edgeOffset += 1) {
      if (!childPresence[edgeOffset] || josekiRedirectAt(data, edgeStarts[nodeIndex] + edgeOffset)) {
        continue
      }
      if (childIndex >= nodeCount || subtreeNodes[childIndex] === 0) {
        throw new Error("Joseki node topology is incomplete")
      }
      subtreeSize += subtreeNodes[childIndex]
      childIndex += subtreeNodes[childIndex]
    }
    subtreeNodes[nodeIndex] = subtreeSize
  }
  if (nodeCount === 0 || subtreeNodes[0] !== nodeCount) {
    throw new Error("Joseki node topology is incomplete")
  }

  const continuationCounts = new Uint32Array(nodeCount)
  const continuationState = new Uint8Array(nodeCount)
  function countContinuations(nodeIndex) {
    if (continuationState[nodeIndex] === 2) {
      return continuationCounts[nodeIndex]
    }
    if (continuationState[nodeIndex] === 1) {
      throw new Error("Joseki redirect topology is cyclic")
    }
    continuationState[nodeIndex] = 1
    let count = edgeStarts[nodeIndex + 1] - edgeStarts[nodeIndex]
    for (const target of josekiChildIndices(data, nodeIndex)) {
      if (target !== null) {
        if (target < 0 || target >= nodeCount) {
          throw new Error("Joseki redirect topology is invalid")
        }
        count += countContinuations(target)
      }
    }
    if (!Number.isSafeInteger(count) || count > 0xFFFFFFFF) {
      throw new Error("Joseki continuation count is too large")
    }
    continuationCounts[nodeIndex] = count
    continuationState[nodeIndex] = 2
    return count
  }
  countContinuations(0)
  if ([...continuationState].some((value) => value !== 2)) {
    throw new Error("Joseki graph contains unreachable nodes")
  }
  data.continuationCounts = continuationCounts
  return data
}

function decodeJosekiNode(data, nodeIndex, line) {
  let structure = data.nodeCache.get(nodeIndex)
  if (!structure) {
    const rawNode = data.rawNodes[nodeIndex]
    const childIndices = josekiChildIndices(data, nodeIndex)
    const candidates = []
    let childOffset = 0
    const localStart = data.localStarts[nodeIndex]
    const firstLocalStart = data.firstLocalStarts[nodeIndex]
    const siblingLocalStart = localStart - firstLocalStart
    let previousDrop = 0
    for (let index = 0; index < rawNode.localCount; index += 1) {
      if (index === 0) {
        previousDrop = data.view.getUint8(data.firstLocalDropOffset + firstLocalStart)
      } else {
        previousDrop = (
          previousDrop
          + data.view.getUint8(data.siblingLocalDropOffset + siblingLocalStart + index - 1)
        ) & 0xFF
      }
      candidates.push({
        kind: "local",
        local: decodeLocalMove(data.view.getUint8(data.localMoveOffset + localStart + index)),
        stone_fraction: decodeThousandths(1000 - previousDrop),
        retained: true,
        childIndex: childIndices[childOffset],
      })
      childOffset += 1
    }
    if (rawNode.tenukiPresent) {
      candidates.push({
        kind: "tenuki",
        stone_fraction: decodeThousandths(rawNode.tenukiStoneFraction),
        retained: rawNode.tenukiRetained,
        childIndex: rawNode.tenukiRetained ? childIndices[childOffset] : null,
      })
    }
    structure = { index: nodeIndex, is_core: rawNode.isCore, candidates }
    data.nodeCache.set(nodeIndex, structure)
  }
  const family = data.family
  const entries = line ? parseEntries(line, family) : []
  const candidates = structure.candidates.map((candidate) => ({
    ...candidate,
    childLine: candidate.retained
      ? formatLine(family, [
          ...entries,
          candidate.kind === "local" ? candidate.local : null,
        ])
      : null,
  }))
  const node = {
    ...structure,
    line,
    candidates,
  }
  data.nodeByLine.set(line, node)
  return node
}

function josekiNodeForLine(line, data = state.data) {
  if (!data) {
    return null
  }
  const lineText = String(line || "")
  const cached = data.nodeByLine.get(lineText)
  if (cached) {
    return cached
  }
  const entries = lineText ? parseEntries(lineText, data.family) : []
  let node = decodeJosekiNode(data, 0, "")
  let currentLine = ""
  for (const entry of entries) {
    const candidate = node.candidates.find((row) => (
      entry === null ? row.kind === "tenuki" : row.kind === "local" && entriesEqual(row.local, entry)
    ))
    if (!candidate || !candidate.retained || candidate.childIndex === null) {
      return null
    }
    currentLine = candidate.childLine
    node = decodeJosekiNode(data, candidate.childIndex, currentLine)
  }
  return node
}

function lineMetaText(line) {
  const node = josekiNodeForLine(line)
  const count = node ? Number(state.data.continuationCounts[node.index] || 0) : 0
  return `${formatVisits(count)} position${count === 1 ? "" : "s"} in subtree`
}

function childSubtreeCount(childLine) {
  if (!childLine) {
    return 1
  }
  const node = josekiNodeForLine(childLine)
  return 1 + (node ? Number(state.data.continuationCounts[node.index] || 0) : 0)
}

function lineParent(line, family = null) {
  const familyValue = family || inferFamilyFromLine(line)
  if (!familyValue) {
    return ""
  }
  const entries = parseEntries(line, familyValue)
  if (entries.length === 0) {
    return ""
  }
  return formatLine(familyValue, entries.slice(0, -1))
}

function entriesEqual(a, b) {
  if (a === null || b === null) {
    return a === b
  }
  return Array.isArray(a) && Array.isArray(b) && Number(a[0]) === Number(b[0]) && Number(a[1]) === Number(b[1])
}

function formatLine(family, entries) {
  if (!entries.length) {
    return ""
  }
  return `${family}[${entries.map((entry) => (entry ? `${entry[0]},${entry[1]}` : "")).join(":")}]`
}

function familyMoveToCell(family, move, boardSize) {
  const x = Number(move[0])
  const y = Number(move[1])
  const size = Number(boardSize)
  if (family === "O") {
    return `${String.fromCharCode(96 + y)}${size - x + 1}`
  }
  return `${String.fromCharCode(96 + (size - y + 1))}${size - x + 1}`
}

function hexWorldMoveStream(entries, family, boardSize) {
  return entries.map((entry) => (entry ? familyMoveToCell(family, entry, boardSize) : ":p")).join("")
}

function hexWorldUrlForCurrentPosition() {
  const family = currentFamily()
  if (!family) {
    return null
  }
  const boardSize = Number(state.data?.board_size || 19)
  const currentEntries = parseEntries(String(state.currentLine || ""), family)
  const futureEntries = futureTailLines().map((line) => {
    const entries = parseEntries(String(line || ""), family)
    return entries[entries.length - 1] ?? null
  })
  const past = hexWorldMoveStream(currentEntries, family, boardSize)
  const future = hexWorldMoveStream(futureEntries, family, boardSize)
  return hexWorldUrlWithCursor(`https://hexworld.org/board/#${boardSize}c1`, past, future)
}

function displayLineText(node) {
  const family = String(inferFamilyFromLine(node?.line || "") || node?.family || currentFamily() || "")
  if (!state.currentLine) {
    return family || "—"
  }
  return state.currentLine
}

function moveText(entry) {
  return entry ? `${entry[0]}-${entry[1]}` : "tenuki"
}

function linePrefixes(line, family = null) {
  const familyValue = family || inferFamilyFromLine(line)
  if (!familyValue) {
    return []
  }
  const entries = parseEntries(String(line || ""), familyValue)
  const prefixes = []
  for (let i = 1; i <= entries.length; i += 1) {
    prefixes.push(formatLine(familyValue, entries.slice(0, i)))
  }
  return prefixes
}

function lineEntries(line) {
  const family = inferFamilyFromLine(line) || currentFamily()
  if (!family) {
    return []
  }
  return parseEntries(String(line || ""), family)
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
  parseLine: lineEntries,
  linePrefixes,
  lineParent,
  sanitizeLine: (line) => normalizeLineForFamily(line, currentFamily()),
  setHashFromLine,
  render: () => render(),
  entryEquals: entriesEqual,
  canFollowLine: (previousLine, nextLine) => {
    const previousFamily = inferFamilyFromLine(previousLine) || currentFamily()
    const nextFamily = inferFamilyFromLine(nextLine) || previousFamily
    return previousFamily === nextFamily
  },
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

function renderMoveList() {
  const family = currentFamily()
  if (!family) {
    elements.moveList.replaceChildren()
    return
  }
  const currentLine = String(state.currentLine || "")
  const currentEntries = parseEntries(currentLine, family)
  const currentMoveCount = currentEntries.length
  const futureLines = futureTailLines()
  const parts = [
    ...currentEntries.map((entry, index) => ({
      text: moveText(entry),
      isFuture: false,
      line: formatLine(family, currentEntries.slice(0, index + 1)),
    })),
    ...futureLines.map((line) => {
      const entries = parseEntries(line, family)
      return {
        text: moveText(entries[entries.length - 1] || null),
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

function childLineForCandidate(node, row) {
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || "")
  const entries = parseEntries(String(node.line || ""), family)
  if (row.kind === "local" && Array.isArray(row.local) && row.local.length === 2) {
    return formatLine(family, [...entries, [Number(row.local[0]), Number(row.local[1])]])
  }
  if (row.kind === "tenuki" && entries.length > 0) {
    return formatLine(family, [...entries, null])
  }
  return null
}

function localToDisplay(local, family) {
  const x = Number(local[0])
  const y = Number(local[1])
  if (family === "O") {
    return {
      col: y,
      row: LOCAL_BOARD_SIZE - x + 1,
    }
  }
  return {
    col: LOCAL_BOARD_SIZE - y + 1,
    row: LOCAL_BOARD_SIZE - x + 1,
  }
}

function displayToLocal(col, row, family) {
  if (family === "O") {
    return [
      LOCAL_BOARD_SIZE - Number(row) + 1,
      Number(col),
    ]
  }
  return [
    LOCAL_BOARD_SIZE - Number(row) + 1,
    LOCAL_BOARD_SIZE - Number(col) + 1,
  ]
}

function displayCellInLocalRegion(col, row, family) {
  if (Number(col) < 1 || Number(col) > LOCAL_BOARD_SIZE || Number(row) < 1 || Number(row) > LOCAL_BOARD_SIZE) {
    return false
  }
  return localDelta(displayToLocal(col, row, family), family) <= Number(LOCAL_DELTA_MAX[family] || LOCAL_DELTA_MAX.A)
}

function boardPointsForNode(node) {
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || DEFAULT_FAMILY)
  const entries = parseEntries(String(node.line || ""), family)
  const stones = []
  const currentOccupiedPly = entries.length > 0 && entries[entries.length - 1] ? entries.length : null
  let tenukiStone = null
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i]
    if (!entry) {
      const color = i % 2 === 0 ? "red" : "blue"
      const base = color === "red" ? RED_RGB : BLUE_RGB
      tenukiStone = {
        color,
        ply: i + 1,
        isLast: entries.length === i + 1,
        textColor: rgbText((entries.length === i + 1) ? TEXT_ON_DARK_RGB : lerpRgb(base, TEXT_ON_DARK_RGB, 0.45)),
      }
      continue
    }
    const point = localToDisplay(entry, family)
    const color = i % 2 === 0 ? "red" : "blue"
    const base = color === "red" ? RED_RGB : BLUE_RGB
    stones.push({
      col: point.col,
      row: point.row,
      color,
      ply: i + 1,
      isLast: currentOccupiedPly === i + 1,
      textColor: rgbText((currentOccupiedPly === i + 1) ? TEXT_ON_DARK_RGB : lerpRgb(base, TEXT_ON_DARK_RGB, 0.45)),
    })
  }
  const overlays = []
  let tenuki = null
  for (const row of node.candidates || []) {
    if (row.kind === "local" && Array.isArray(row.local) && row.local.length === 2 && typeof row.stone_fraction === "number") {
      const childLine = childLineForCandidate(node, row)
      if (!row.retained || !childLine) {
        continue
      }
      const point = localToDisplay(row.local, family)
      overlays.push({
        col: point.col,
        row: point.row,
        stoneFraction: Number(row.stone_fraction),
        childLine,
      })
    } else if (row.kind === "tenuki" && typeof row.stone_fraction === "number") {
      const childLine = childLineForCandidate(node, row)
      tenuki = {
        stoneFraction: Number(row.stone_fraction),
        childLine: row.retained ? childLine : null,
      }
    }
  }
  return { stones, overlays, tenuki, tenukiStone }
}

function setupViewBox(family) {
  const pixels = [pointToPixel(TENUKI_POINT.col, TENUKI_POINT.row)]
  for (let row = 1; row <= LOCAL_BOARD_SIZE; row += 1) {
    for (let col = 1; col <= LOCAL_BOARD_SIZE; col += 1) {
      if (!displayCellInLocalRegion(col, row, family)) {
        continue
      }
      pixels.push(pointToPixel(col, row))
    }
  }
  setSvgViewBoxFromPixels(elements.board, pixels, VIEW_PADDING)
}

function renderBoard() {
  clearSvg()
  const node = josekiNodeForLine(state.currentLine) || (() => {
    const family = currentFamily()
    return family ? { family, line: state.currentLine, candidates: [] } : null
  })()
  if (!node) {
    return
  }
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || DEFAULT_FAMILY)
  const entries = parseEntries(String(node.line || ""), family)
  setupViewBox(family)
  const toPlay = entries.length % 2 === 0 ? "red" : "blue"
  const hoverColor = toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
  const hoverFill = turnRgbaText(toPlay, 0.12)
  const { stones, overlays, tenuki, tenukiStone } = boardPointsForNode(node)
  let topChildSubtreeCount = 0
  for (const overlay of overlays) {
    if (!overlay?.childLine) {
      continue
    }
    topChildSubtreeCount = Math.max(topChildSubtreeCount, childSubtreeCount(overlay.childLine))
  }
  const stoneByKey = new Map(stones.map((stone) => [`${stone.col},${stone.row}`, stone]))
  const overlayByKey = new Map(overlays.map((overlay) => [`${overlay.col},${overlay.row}`, overlay]))

  function candidateFill(stoneFraction, childLine) {
    const count = childSubtreeCount(childLine)
    return analysisHeatFill({
      weight: count,
      topWeight: topChildSubtreeCount,
      value: stoneFraction,
    })
  }

  for (let row = 1; row <= LOCAL_BOARD_SIZE; row += 1) {
    for (let col = 1; col <= LOCAL_BOARD_SIZE; col += 1) {
      if (!displayCellInLocalRegion(col, row, family)) {
        continue
      }
      const key = `${col},${row}`
      const stone = stoneByKey.get(key) || null
      const overlay = overlayByKey.get(key) || null
      let fill = rgbText(OFF_WHITE_RGB)
      if (overlay) {
        fill = candidateFill(overlay.stoneFraction, overlay.childLine)
      }
      if (stone) {
        fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
      }
      const hitClasses = ["board-hover-hit"]
      const onClick = overlay && overlay.childLine
        ? () => {
            goToLine(overlay.childLine)
          }
        : stone && stone.isLast && state.currentLine
          ? () => {
              goPrevious()
            }
          : null
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
        onClick,
      })
      hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
      const hex = appendHex(col, row, {
        fill,
        className: `${overlay ? "board-hex candidate" : "board-hex"} board-hex-face`,
        stroke: overlay ? "none" : GRID_EDGE,
      })
      hex.polygon.style.setProperty("--hover-outline", hoverColor)
      if (overlay && !stone) {
        appendStackedText(hex.cx, hex.cy, percentText(overlay.stoneFraction), formatVisits(childSubtreeCount(overlay.childLine)))
      }
      if (stone) {
        appendText(hex.cx, hex.cy, String(stone.ply), "cell-text", stone.textColor)
      }
    }
  }
  const tenukiOnClick = tenukiStone && tenukiStone.isLast
    ? () => {
        goPrevious()
      }
    : (tenuki && tenuki.childLine ? () => {
        goToLine(tenuki.childLine)
      } : null)
  const tenukiHitClasses = ["board-hover-hit"]
  if (tenukiOnClick) {
    tenukiHitClasses.push("clickable")
  }
  if (!tenukiStone && tenuki && tenuki.childLine) {
    tenukiHitClasses.push("hoverable")
  }
  const tenukiHoverHex = appendHex(TENUKI_POINT.col, TENUKI_POINT.row, {
    fill: "transparent",
    stroke: "none",
    className: tenukiHitClasses.join(" "),
    size: HEX_SIZE,
    onClick: tenukiOnClick,
  })
  tenukiHoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
  const tenukiHex = appendHex(TENUKI_POINT.col, TENUKI_POINT.row, {
    fill: tenukiStone
      ? (tenukiStone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB))
      : (tenuki && tenuki.childLine ? candidateFill(tenuki.stoneFraction, tenuki.childLine) : rgbText(OFF_WHITE_RGB)),
    className: `${tenukiStone
      ? "board-hex"
      : (tenuki && tenuki.childLine ? "board-hex candidate" : "board-hex")} board-hex-face`,
    stroke: tenukiStone ? "none" : GRID_EDGE,
    onClick: tenukiOnClick,
  })
  tenukiHex.polygon.style.setProperty("--hover-outline", hoverColor)
  appendText(tenukiHex.cx, tenukiHex.cy - (HEX_SIZE * 1.28), "Tenuki", "tenuki-label")
  if (tenukiStone) {
    appendText(tenukiHex.cx, tenukiHex.cy, String(tenukiStone.ply), "cell-text", tenukiStone.textColor)
  } else if (tenuki) {
    if (tenuki.childLine) {
      appendStackedText(
        tenukiHex.cx,
        tenukiHex.cy,
        percentText(tenuki.stoneFraction),
        formatVisits(childSubtreeCount(tenuki.childLine)),
      )
    } else {
      appendText(tenukiHex.cx, tenukiHex.cy, percentText(tenuki.stoneFraction))
    }
  }
  const borderWidth = 4
  const red = rgbText(RED_RGB)
  const blue = rgbText(BLUE_RGB)
  if (family === "O") {
    for (let col = 1; col <= LOCAL_BOARD_SIZE; col += 1) {
      if (!displayCellInLocalRegion(col, LOCAL_BOARD_SIZE, family)) {
        continue
      }
      const [cx, cy] = pointToPixel(col, LOCAL_BOARD_SIZE)
      const c3 = hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
      const c2 = hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
      const c1 = hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
      appendLine(c3[0], c3[1], c2[0], c2[1], red, borderWidth)
      appendLine(c2[0], c2[1], c1[0], c1[1], red, borderWidth)
    }
    for (let row = 1; row <= LOCAL_BOARD_SIZE; row += 1) {
      if (!displayCellInLocalRegion(1, row, family)) {
        continue
      }
      const [cx, cy] = pointToPixel(1, row)
      const c2 = hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
      const c3 = hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
      const c4 = hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
      appendLine(c2[0], c2[1], c3[0], c3[1], blue, borderWidth)
      appendLine(c3[0], c3[1], c4[0], c4[1], blue, borderWidth)
    }
  } else {
    for (let col = 1; col <= LOCAL_BOARD_SIZE; col += 1) {
      const [cx, cy] = pointToPixel(col, LOCAL_BOARD_SIZE)
      const c3 = hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
      const c2 = hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
      const c1 = hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
      appendLine(c3[0], c3[1], c2[0], c2[1], red, borderWidth)
      appendLine(c2[0], c2[1], c1[0], c1[1], red, borderWidth)
    }
    for (let row = 1; row <= LOCAL_BOARD_SIZE; row += 1) {
      const [cx, cy] = pointToPixel(LOCAL_BOARD_SIZE, row)
      const c5 = hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
      const c0 = hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
      const c1 = hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
      appendLine(c5[0], c5[1], c0[0], c0[1], blue, borderWidth)
      appendLine(c0[0], c0[1], c1[0], c1[1], blue, borderWidth)
    }
  }
}

function setHashFromLine(line) {
  const family = inferFamilyFromLine(line) || state.family || DEFAULT_FAMILY
  const hash = line ? `#${line}` : (family === DEFAULT_FAMILY ? "" : `#${family}`)
  replaceHash(hash)
}

function renderHexWorldLink() {
  elements.hexWorldLink.replaceChildren()
  renderExternalLink(elements.hexWorldLink, hexWorldUrlForCurrentPosition())
}

function renderFamilyButtons() {
  syncPressedButtonGroup(
    elements.familyButtons.map((button) => [button.dataset.family, button]),
    currentFamily(),
  )
}

function lineMetaStatusText(line) {
  if (state.dataError && !state.data) {
    return `Data load failed: ${state.dataError}`
  }
  if (state.isLoadingData && !state.data) {
    return "Loading joseki data..."
  }
  return lineMetaText(line)
}

function render() {
  const node = josekiNodeForLine(state.currentLine)
  if (!node) {
    const family = currentFamily()
    const entries = family ? parseEntries(state.currentLine, family) : []
    const toPlay = entries.length % 2 === 0 ? "red" : "blue"
    setTurnStatus(elements.status, family ? toPlay : null)
    setCopyButtonValue(elements.currentLine, displayLineText({ family }))
    renderMoveList()
    elements.lineMeta.textContent = lineMetaStatusText(state.currentLine)
    renderHexWorldLink()
    renderFamilyButtons()
    renderRandomMode()
    renderBoard()
    return
  }
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || DEFAULT_FAMILY)
  const toPlay = parseEntries(String(node.line || ""), family).length % 2 === 0 ? "red" : "blue"
  setTurnStatus(elements.status, toPlay)
  setCopyButtonValue(elements.currentLine, displayLineText(node))
  renderMoveList()
  elements.lineMeta.textContent = lineMetaStatusText(node.line)
  renderHexWorldLink()
  renderFamilyButtons()
  renderRandomMode()
  renderBoard()
}

const ensureJosekiDataLoaded = createKeyedDataLoader({
  state,
  loadingKeyField: "loadingFamily",
  current: (family) => (
    String(state.data?.family || "") === family
      ? state.data
      : null
  ),
  load: async (family, signal) => {
    const dataUrl = await currentDataUrl(family, signal)
    const cached = state.dataByUrl.get(dataUrl)
    if (cached) {
      return cached
    }
    const data = normalizeLoadedData(await fetchArrayBuffer(dataUrl, { signal }))
    state.dataByUrl.set(dataUrl, data)
    return data
  },
  apply: (data) => {
    state.data = data
  },
  render: () => render(),
})

async function ensureDataLoaded(family = null) {
  const requestedFamily = String(family || state.family || DEFAULT_FAMILY).trim().toUpperCase()
  return ensureJosekiDataLoaded(requestedFamily)
}

function requestedViewFromHash() {
  const line = decodeLocationHash()
  if (line === null) {
    return {
      valid: false,
      family: DEFAULT_FAMILY,
      line: "",
    }
  }
  if (!String(line || "").trim()) {
    return {
      valid: true,
      family: DEFAULT_FAMILY,
      line: "",
    }
  }
  const rootFamily = rootFamilyFromLine(line)
  if (rootFamily) {
    return {
      valid: true,
      family: rootFamily,
      line: "",
    }
  }
  const family = inferFamilyFromLine(line)
  if (!family || !normalizeRequestedLineForFamily(line, family)) {
    return {
      valid: false,
      family: DEFAULT_FAMILY,
      line: "",
    }
  }
  return {
    valid: true,
    family,
    line,
  }
}

function syncFromLocationHash() {
  const requested = requestedViewFromHash()
  if (!requested.valid) {
    replaceHash("")
    void requestView({ family: DEFAULT_FAMILY, line: "", updateHash: false })
    return
  }
  void requestView({ family: requested.family, line: requested.line, updateHash: false })
}

async function requestView({ family, line = "", updateHash = true }) {
  const requestedFamily = String(family || state.family || DEFAULT_FAMILY).trim().toUpperCase()
  if (!AVAILABLE_FAMILIES.includes(requestedFamily)) {
    return
  }
  const requestedLine = normalizeRequestedLineForFamily(line, requestedFamily)
  const viewGeneration = state.viewGeneration + 1
  state.viewGeneration = viewGeneration
  const loaded = await ensureDataLoaded(requestedFamily)
  if (viewGeneration !== state.viewGeneration) {
    return
  }
  if (!loaded || String(loaded?.family || "") !== requestedFamily || String(state.data?.family || "") !== requestedFamily) {
    render()
    return
  }
  state.family = requestedFamily
  state.currentLine = requestedLine
  state.dataError = null
  resetLineHistory(state.currentLine)
  if (updateHash) {
    setHashFromLine(state.currentLine)
  }
  render()
  scheduleJosekiRandomIndex(loaded)
}

async function copyCurrentLine() {
  const text = String(elements.currentLine.textContent || "").trim()
  if (!text || text === "—") {
    return
  }
  await copyButtonText(elements.currentLine, text)
}

function ensureJosekiRandomIndex(data) {
  if (!data || (
    data.subtreeCore
    && data.subtreeCoreLocal
    && data.subtreeLeaves
    && data.subtreeLeavesLocal
  )) {
    return
  }
  const subtreeCore = new Uint32Array(data.nodeCount)
  const subtreeCoreLocal = new Uint32Array(data.nodeCount)
  const subtreeLeaves = new Uint32Array(data.nodeCount)
  const subtreeLeavesLocal = new Uint32Array(data.nodeCount)
  const endingKind = new Int8Array(data.nodeCount)
  const visited = new Uint8Array(data.nodeCount)
  endingKind[0] = -1

  function indexNode(nodeIndex) {
    if (visited[nodeIndex] === 2) {
      return
    }
    if (visited[nodeIndex] === 1) {
      throw new Error("Joseki redirect topology is cyclic")
    }
    visited[nodeIndex] = 1
    const rawNode = data.rawNodes[nodeIndex]
    const targets = josekiChildIndices(data, nodeIndex)
    const edgeCount = targets.length
    let coreCount = nodeIndex > 0 && rawNode.isCore ? 1 : 0
    let coreLocalCount = coreCount && endingKind[nodeIndex] === 1 ? 1 : 0
    let leafCount = edgeCount === 0 ? 1 : 0
    let leafLocalCount = leafCount && endingKind[nodeIndex] === 1 ? 1 : 0
    for (let edgeOffset = 0; edgeOffset < edgeCount; edgeOffset += 1) {
      const target = targets[edgeOffset]
      const isLocal = edgeOffset < rawNode.localCount
      if (target === null) {
        leafCount += 1
        leafLocalCount += isLocal ? 1 : 0
        continue
      }
      const targetKind = isLocal ? 1 : 2
      if (endingKind[target] !== 0 && endingKind[target] !== targetKind) {
        throw new Error("Joseki node has inconsistent incoming move kinds")
      }
      endingKind[target] = targetKind
      indexNode(target)
      coreCount += subtreeCore[target]
      coreLocalCount += subtreeCoreLocal[target]
      leafCount += subtreeLeaves[target]
      leafLocalCount += subtreeLeavesLocal[target]
    }
    for (const count of [coreCount, coreLocalCount, leafCount, leafLocalCount]) {
      if (!Number.isSafeInteger(count) || count > 0xFFFFFFFF) {
        throw new Error("Joseki random index is too large")
      }
    }
    subtreeCore[nodeIndex] = coreCount
    subtreeCoreLocal[nodeIndex] = coreLocalCount
    subtreeLeaves[nodeIndex] = leafCount
    subtreeLeavesLocal[nodeIndex] = leafLocalCount
    visited[nodeIndex] = 2
  }
  indexNode(0)
  data.subtreeCore = subtreeCore
  data.subtreeCoreLocal = subtreeCoreLocal
  data.subtreeLeaves = subtreeLeaves
  data.subtreeLeavesLocal = subtreeLeavesLocal
}

function scheduleJosekiRandomIndex(data) {
  if (!data || data.randomIndexScheduled || (
    data.subtreeCore
    && data.subtreeCoreLocal
    && data.subtreeLeaves
    && data.subtreeLeavesLocal
  )) {
    return
  }
  data.randomIndexScheduled = true
  setTimeout(() => {
    data.randomIndexScheduled = false
    ensureJosekiRandomIndex(data)
  }, 0)
}

function josekiLineForRandomRank(data, kind, localOnly, rank) {
  let nodeIndex = 0
  let line = ""
  while (true) {
    const node = decodeJosekiNode(data, nodeIndex, line)
    const entries = line ? parseEntries(line, data.family) : []
    const endsLocal = entries.length > 0 && entries[entries.length - 1] !== null
    if (kind === "core" && nodeIndex > 0 && node.is_core && (!localOnly || endsLocal)) {
      if (rank === 0) {
        return line
      }
      rank -= 1
    }
    const retained = node.candidates.filter((candidate) => candidate.retained)
    if (kind === "leaf" && retained.length === 0) {
      return line
    }
    let selected = false
    for (const candidate of retained) {
      const weight = candidate.childIndex === null
        ? (kind === "leaf" && (!localOnly || candidate.kind === "local") ? 1 : 0)
        : (kind === "core"
            ? (localOnly ? data.subtreeCoreLocal[candidate.childIndex] : data.subtreeCore[candidate.childIndex])
            : (localOnly ? data.subtreeLeavesLocal[candidate.childIndex] : data.subtreeLeaves[candidate.childIndex]))
      if (rank < weight) {
        if (candidate.childIndex === null) {
          return candidate.childLine
        }
        nodeIndex = candidate.childIndex
        line = candidate.childLine
        selected = true
        break
      }
      rank -= weight
    }
    if (!selected) {
      throw new Error("Unsupported joseki random index")
    }
  }
}

async function goRandom() {
  const loaded = await ensureDataLoaded()
  if (!loaded) {
    render()
    return
  }
  const data = loaded
  ensureJosekiRandomIndex(data)
  const kind = state.randomMode === "leaf" ? "leaf" : "core"
  const allCount = kind === "leaf" ? data.subtreeLeaves[0] : data.subtreeCore[0]
  const localCount = kind === "leaf" ? data.subtreeLeavesLocal[0] : data.subtreeCoreLocal[0]
  const count = localCount || allCount
  if (!count) {
    render()
    return
  }
  let line
  do {
    const rank = Math.floor(Math.random() * count)
    line = josekiLineForRandomRank(data, kind, localCount > 0, rank)
  } while (count > 1 && line === state.currentLine)
  jumpToLine(line)
}

function handleFamilyButtonClick(family) {
  if (currentFamily() === family && state.data) {
    if (state.currentLine) {
      jumpToLine("")
    }
    return
  }
  void requestView({ family, line: "" })
}

function renderRandomMode() {
  randomModeControls.sync()
}

for (const button of elements.familyButtons) {
  const family = button.dataset.family
  button.addEventListener("click", (event) => {
    const href = family === DEFAULT_FAMILY ? "./joseki.html" : `./joseki.html#${family}`
    handlePageButtonClick(event, href, () => handleFamilyButtonClick(family))
  })
}
elements.randomCoreBtn.addEventListener("click", () => {
  randomModeControls.set("core")
})
elements.randomLeafBtn.addEventListener("click", () => {
  randomModeControls.set("leaf")
})
elements.resetBtn.addEventListener("click", () => {
  jumpToLine("")
})
elements.randomBtn.addEventListener("click", () => {
  void goRandom()
})
elements.currentLine.addEventListener("click", () => {
  void copyCurrentLine()
})
window.addEventListener("hashchange", () => {
  syncFromLocationHash()
})
window.addEventListener("keydown", (event) => {
  if (shouldIgnoreGlobalKeydown(event)) {
    return
  }
  const node = josekiNodeForLine(state.currentLine) || { line: state.currentLine || formatLine(currentFamily(), []) }
  const { tenuki, tenukiStone } = boardPointsForNode(node)
  const tenukiOnClick = tenukiStone && tenukiStone.isLast
    ? () => {
        goPrevious()
      }
    : (tenuki && tenuki.childLine ? () => {
        goToLine(tenuki.childLine)
      } : null)
  if (event.key === "t" || event.key === "T") {
    if (tenukiOnClick) {
      event.preventDefault()
      tenukiOnClick()
    }
    return
  }
  handleStandardKeydown(event, {
    goPrevious,
    goNext,
    goFirst,
    goLast,
    deleteFromCursor,
  })
})

async function main() {
  syncFromLocationHash()
}

void main()
