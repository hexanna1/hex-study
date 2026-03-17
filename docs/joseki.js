const MANIFEST_URL = "./data/joseki_current.json"
const BUNDLE_MAGIC = "HJB1"
const BUNDLE_VERSION = 1
const FAMILY_CODE_TO_NAME = {
  1: "A",
  2: "O",
}
const HEADER_SIZE = 16
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
  buildCoreLines,
  buildDescendantCounts,
  buildRetainedLeafLines,
  copyButtonText,
  createKeyedDataLoader,
  createLineNavigator,
  createModeButtonGroup,
  createSvgTools,
  decodeThousandths,
  fetchArrayBuffer,
  fetchJson,
  formatVisits,
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
  familyABtn: document.getElementById("family-a-btn"),
  familyOBtn: document.getElementById("family-o-btn"),
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
  defaultStrokeWidth: "0.9",
})

const state = {
  data: null,
  nodesByLine: new Map(),
  descendantCountsByLine: new Map(),
  dataByUrl: new Map(),
  manifestByUrl: new Map(),
  currentLine: "",
  family: "A",
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
  if (!(familyValue === "A" || familyValue === "O")) {
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
    || "A"
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

function decodeCompactNode(rawNode, rawLocalRows, family, entries, cursors, nodes) {
  const line = formatLine(family, entries)
  const retained_lines = []
  const candidates = []
  const localCount = rawNode.localCount
  const localRowStart = cursors.localRow
  if (!Number.isInteger(localCount) || localCount < 0 || localRowStart < 0 || localRowStart + localCount > rawLocalRows.length) {
    throw new Error("Unsupported joseki local child slice")
  }
  const node = { line, retained_lines, candidates, is_core: rawNode.isCore }
  nodes.push(node)
  cursors.node += 1
  const localChildren = []
  for (let idx = localRowStart; idx < localRowStart + localCount; idx += 1) {
    const row = rawLocalRows[idx]
    const local = decodeLocalMove(row.moveCode)
    candidates.push({
      kind: "local",
      local,
      stone_fraction: decodeThousandths(row.stoneFraction),
    })
    const childEntries = [...entries, local]
    retained_lines.push(formatLine(family, childEntries))
    if (rawNode.localChildren) {
      localChildren.push(childEntries)
    }
  }
  cursors.localRow += localCount
  for (const childEntries of localChildren) {
    if (cursors.node >= cursors.nodeCount) {
      throw new Error("Unsupported joseki preorder child")
    }
    decodeCompactNode(cursors.rawNodes[cursors.node], rawLocalRows, family, childEntries, cursors, nodes)
  }
  if (rawNode.tenukiPresent) {
    candidates.push({
      kind: "tenuki",
      stone_fraction: decodeThousandths(rawNode.tenukiStoneFraction),
    })
    if (rawNode.tenukiRetained && entries.length > 0) {
      const childEntries = [...entries, null]
      retained_lines.push(formatLine(family, childEntries))
      if (rawNode.tenukiChildPresent) {
        if (cursors.node >= cursors.nodeCount) {
          throw new Error("Unsupported joseki preorder tenuki child")
        }
        decodeCompactNode(cursors.rawNodes[cursors.node], rawLocalRows, family, childEntries, cursors, nodes)
      }
    }
  }
}

function readBitplaneValue(view, offset, valueCount, valueIndex, bits) {
  let value = 0
  for (let bit = 0; bit < bits; bit += 1) {
    value = (value * 2) + readPackedWordAtBit(view, offset, (bit * valueCount) + valueIndex, 1)
  }
  return value
}

function normalizeLoadedData(rawBuffer) {
  if (!(rawBuffer instanceof ArrayBuffer)) {
    throw new Error("Unsupported joseki data format")
  }
  if (rawBuffer.byteLength < HEADER_SIZE) {
    throw new Error("Unsupported joseki data format")
  }
  const view = new DataView(rawBuffer)
  if (readAsciiMagic(view) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported joseki data format")
  }
  const version = view.getUint16(4, true)
  if (version !== BUNDLE_VERSION) {
    throw new Error("Unsupported joseki data format")
  }
  const family = FAMILY_CODE_TO_NAME[view.getUint8(6)]
  if (!(family === "A" || family === "O")) {
    throw new Error("Unsupported joseki data format")
  }
  const boardSize = view.getUint8(7)
  const nodeCount = view.getUint32(8, true)
  const localRowCount = view.getUint32(12, true)
  const nodeControlOffset = HEADER_SIZE
  const tenukiDropLowOffset = nodeControlOffset + nodeCount
  const tenukiDropHighOffset = tenukiDropLowOffset + nodeCount
  const localMoveOffset = tenukiDropHighOffset + Math.ceil((nodeCount * PACKED_TENUKI_DROP_HIGH_BITS) / 8)
  const firstLocalDropOffset = localMoveOffset + localRowCount
  const expectedSize = firstLocalDropOffset + localRowCount
  if (rawBuffer.byteLength !== expectedSize) {
    throw new Error("Joseki bundle size mismatch")
  }
  const rawNodes = []
  const rawLocalRows = []
  let firstLocalDropCount = 0
  for (let idx = 0; idx < nodeCount; idx += 1) {
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
    rawNodes.push({
      localCount,
      isCore: (Math.trunc(word / (2 ** PACKED_NODE_IS_CORE_SHIFT)) & 0x1) !== 0,
      tenukiRetained: (Math.trunc(word / (2 ** PACKED_NODE_TENUKI_RETAINED_SHIFT)) & 0x1) !== 0,
      tenukiPresent,
      tenukiChildPresent: (Math.trunc(word / (2 ** PACKED_NODE_TENUKI_CHILD_SHIFT)) & 0x1) !== 0,
      tenukiStoneFraction,
      localChildren: (Math.trunc(word / (2 ** PACKED_NODE_LOCAL_CHILDREN_SHIFT)) & 0x1) !== 0,
    })
  }
  const siblingLocalDropOffset = firstLocalDropOffset + firstLocalDropCount
  let localRowIndex = 0
  let firstLocalDropIndex = 0
  let siblingLocalDropIndex = 0
  for (const rawNode of rawNodes) {
    let previousDrop = 0
    for (let idx = 0; idx < rawNode.localCount; idx += 1) {
      if (idx === 0) {
        previousDrop = view.getUint8(firstLocalDropOffset + firstLocalDropIndex)
        firstLocalDropIndex += 1
      } else {
        previousDrop = (previousDrop + view.getUint8(siblingLocalDropOffset + siblingLocalDropIndex)) & 0xFF
        siblingLocalDropIndex += 1
      }
      rawLocalRows.push({
        moveCode: view.getUint8(localMoveOffset + localRowIndex),
        stoneFraction: 1000 - previousDrop,
      })
      localRowIndex += 1
    }
  }
  if (
    localRowIndex !== localRowCount
    || firstLocalDropIndex !== firstLocalDropCount
    || siblingLocalDropIndex !== localRowCount - firstLocalDropCount
  ) {
    throw new Error("Joseki local drop streams were not fully consumed")
  }
  const nodes = []
  const cursors = { node: 0, localRow: 0, nodeCount, rawNodes }
  if (nodeCount > 0) {
    decodeCompactNode(rawNodes[0], rawLocalRows, family, [], cursors, nodes)
  }
  if (cursors.localRow !== rawLocalRows.length) {
    throw new Error("Joseki local rows were not fully consumed")
  }
  if (cursors.node !== rawNodes.length) {
    throw new Error("Joseki preorder nodes were not fully consumed")
  }
  return {
    family,
    board_size: Number(boardSize || 19),
    nodes,
  }
}

function lineMetaText(line) {
  const count = Number(state.descendantCountsByLine.get(String(line || "")) || 0)
  return `${formatVisits(count)} position${count === 1 ? "" : "s"} in subtree`
}

function childSubtreeCount(childLine) {
  if (!childLine) {
    return 1
  }
  return 1 + Number(state.descendantCountsByLine.get(String(childLine || "")) || 0)
}

function retainedLeafLines() {
  return buildRetainedLeafLines(
    state.nodesByLine,
    (node) => (Array.isArray(node?.retained_lines) ? node.retained_lines : []),
  )
}

function coreLines() {
  return buildCoreLines(state.nodesByLine)
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
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || "A")
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
  const retained = new Set(Array.isArray(node.retained_lines) ? node.retained_lines : [])
  for (const row of node.candidates || []) {
    if (row.kind === "local" && Array.isArray(row.local) && row.local.length === 2 && typeof row.stone_fraction === "number") {
      const childLine = childLineForCandidate(node, row)
      if (!childLine || !retained.has(childLine)) {
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
        childLine: childLine && retained.has(childLine) ? childLine : null,
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
  const node = state.nodesByLine.get(state.currentLine) || (() => {
    const family = currentFamily()
    return family ? { family, line: state.currentLine, candidates: [], retained_lines: [] } : null
  })()
  if (!node) {
    return
  }
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || "A")
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
      const isRealBorder = col === LOCAL_BOARD_SIZE || row === LOCAL_BOARD_SIZE
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
        strokeWidth: isRealBorder ? "1.35" : "0.85",
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
    strokeWidth: tenukiStone ? "0" : "1.0",
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
  const family = inferFamilyFromLine(line) || state.family || "A"
  const hash = line ? `#${line}` : (family === "O" ? "#O" : "")
  replaceHash(hash)
}

function renderHexWorldLink(node) {
  void node
  elements.hexWorldLink.replaceChildren()
  renderExternalLink(elements.hexWorldLink, hexWorldUrlForCurrentPosition())
}

function renderFamilyButtons() {
  syncPressedButtonGroup([
    ["A", elements.familyABtn],
    ["O", elements.familyOBtn],
  ], currentFamily())
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
  const node = state.nodesByLine.get(state.currentLine)
  if (!node) {
    const family = currentFamily()
    const entries = family ? parseEntries(state.currentLine, family) : []
    const toPlay = entries.length % 2 === 0 ? "red" : "blue"
    setTurnStatus(elements.status, family ? toPlay : null)
    setCopyButtonValue(elements.currentLine, displayLineText({ family }))
    renderMoveList()
    elements.lineMeta.textContent = lineMetaStatusText(state.currentLine)
    renderHexWorldLink({ line: state.currentLine || formatLine(family, []) })
    renderFamilyButtons()
    renderRandomMode()
    renderBoard()
    return
  }
  const family = String(inferFamilyFromLine(node.line || "") || currentFamily() || "A")
  const toPlay = parseEntries(String(node.line || ""), family).length % 2 === 0 ? "red" : "blue"
  setTurnStatus(elements.status, toPlay)
  setCopyButtonValue(elements.currentLine, displayLineText(node))
  renderMoveList()
  elements.lineMeta.textContent = lineMetaStatusText(node.line)
  renderHexWorldLink(node)
  renderFamilyButtons()
  renderRandomMode()
  renderBoard()
}

const ensureJosekiDataLoaded = createKeyedDataLoader({
  state,
  loadingKeyField: "loadingFamily",
  current: (family) => (
    String(state.data?.family || "") === family
      ? {
          data: state.data,
          nodesByLine: state.nodesByLine,
          descendantCountsByLine: state.descendantCountsByLine,
        }
      : null
  ),
  load: async (family, signal) => {
    const dataUrl = await currentDataUrl(family, signal)
    const cached = state.dataByUrl.get(dataUrl)
    if (cached) {
      return cached
    }
    const data = normalizeLoadedData(await fetchArrayBuffer(dataUrl, { cache: "no-store", signal }))
    const nodes = Array.isArray(data.nodes) ? data.nodes : []
    const nodesByLine = new Map(nodes.map((node) => [String(node.line || ""), node]))
    const descendantCountsByLine = buildDescendantCounts(
      nodesByLine,
      (node) => (Array.isArray(node?.retained_lines) ? node.retained_lines : []),
    )
    const loaded = { data, nodesByLine, descendantCountsByLine }
    state.dataByUrl.set(dataUrl, loaded)
    return loaded
  },
  apply: (loaded) => {
    state.data = loaded.data
    state.nodesByLine = loaded.nodesByLine
    state.descendantCountsByLine = loaded.descendantCountsByLine
  },
  render: () => render(),
})

async function ensureDataLoaded(family = null) {
  const requestedFamily = String(family || state.family || "A").trim().toUpperCase()
  return ensureJosekiDataLoaded(requestedFamily)
}

function requestedViewFromHash() {
  const line = decodeLocationHash()
  if (line === null) {
    return {
      valid: false,
      family: "A",
      line: "",
    }
  }
  if (!String(line || "").trim()) {
    return {
      valid: true,
      family: "A",
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
      family: "A",
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
    void requestView({ family: "A", line: "", updateHash: false })
    return
  }
  void requestView({ family: requested.family, line: requested.line, updateHash: false })
}

async function requestView({ family, line = "", updateHash = true }) {
  const requestedFamily = String(family || state.family || "A").trim().toUpperCase()
  if (!(requestedFamily === "A" || requestedFamily === "O")) {
    return
  }
  const requestedLine = normalizeRequestedLineForFamily(line, requestedFamily)
  const viewGeneration = state.viewGeneration + 1
  state.viewGeneration = viewGeneration
  const loaded = await ensureDataLoaded(requestedFamily)
  if (viewGeneration !== state.viewGeneration) {
    return
  }
  if (!loaded || String(loaded?.data?.family || "") !== requestedFamily || String(state.data?.family || "") !== requestedFamily) {
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
}

async function copyCurrentLine() {
  const text = String(elements.currentLine.textContent || "").trim()
  if (!text || text === "—") {
    return
  }
  await copyButtonText(elements.currentLine, text)
}

async function goRandom() {
  const loaded = await ensureDataLoaded()
  if (!loaded) {
    render()
    return
  }
  const allLines = (state.randomMode === "leaf" ? retainedLeafLines() : coreLines())
    .filter((line) => String(line || "") !== "")
  if (!allLines.length) {
    render()
    return
  }
  const localLines = allLines.filter((line) => {
    const family = inferFamilyFromLine(line) || currentFamily()
    const entries = line ? parseEntries(line, family) : []
    return entries.length > 0 && entries[entries.length - 1] !== null
  })
  const eligible = localLines.length ? localLines : allLines
  const filtered =
    eligible.length > 1 && state.currentLine
      ? eligible.filter((line) => String(line || "") !== state.currentLine)
      : eligible
  const pool = filtered.length ? filtered : eligible
  const line = String(pool[Math.floor(Math.random() * pool.length)] || "")
  if (!line && !pool.includes("")) {
    return
  }
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

elements.familyABtn.addEventListener("click", () => {
  handleFamilyButtonClick("A")
})
elements.familyOBtn.addEventListener("click", () => {
  handleFamilyButtonClick("O")
})
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
  const node = state.nodesByLine.get(state.currentLine) || { line: state.currentLine || formatLine(currentFamily(), []) }
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
