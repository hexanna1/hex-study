const CURRENT_URL = "./data/patterns_current.json"
const BUNDLE_MAGIC = "HPB1"
const BUNDLE_VERSION = 1
const HEADER_SIZE = 20
const PAIR_COUNT_SIZE = 2
const PAIR_ROW_SIZE = 2
const PACKED_STONE_FRACTION_BITS = 10
const PACKED_OPTIONAL_NULL = 1023
const FRACTION_MODE_COORD_MAJOR_BITPLANE = 2
const PRESENCE_MODE_ALL = 0
const PRESENCE_MODE_BITMAP = 1
const PRESENCE_MODE_PRESENT_GAPS = 2
const PRESENCE_MODE_ABSENT_GAPS = 3
const PACKED_KEY_COORD_MIN = -8
const BOARD_RADIUS = 6
const HEX_SIZE = 24
const VIEW_PADDING = 38
const CENTER_POINT = [0, 0]
const TENUKI_POINT = [-BOARD_RADIUS - 2, Math.round((2 * BOARD_RADIUS) / 3)]

const RED_RGB = [220, 60, 60]
const BLUE_RGB = [40, 100, 220]
const OFF_WHITE_RGB = [246, 241, 232]
const GRID_EDGE = "rgb(182, 182, 182)"
const CANDIDATE_LOW = [244, 232, 250]
const CANDIDATE_HIGH = [170, 125, 210]

const {
  createSvgTools,
  decodeThousandths,
  makeResultFill,
  rgbText,
  shouldIgnoreGlobalKeydown,
} = window.HexStudyUI

const elements = {
  board: document.getElementById("board"),
  clearBtn: document.getElementById("clear-btn"),
  randomBtn: document.getElementById("random-btn"),
  randomLe4Btn: document.getElementById("random-le4-btn"),
  random5Btn: document.getElementById("random-5-btn"),
  symmetryButtons: Array.from(document.querySelectorAll(".symmetry-btn")),
  turnIndicator: document.getElementById("turn-indicator"),
  patternInput: document.getElementById("pattern-input"),
  canonicalPattern: document.getElementById("canonical-pattern"),
  patternsCount: document.getElementById("patterns-count"),
  lookupStatus: document.getElementById("lookup-status"),
}

const resultFill = makeResultFill(CANDIDATE_LOW, CANDIDATE_HIGH)
const {
  appendText,
  clear: clearSvg,
  createNode: createSvgNode,
  hexPolygonPoints,
  pointToPixel: rawPointToPixel,
} = createSvgTools({
  board: elements.board,
  hexSize: HEX_SIZE,
  defaultFill: rgbText(OFF_WHITE_RGB),
  defaultStroke: GRID_EDGE,
  defaultStrokeWidth: "1",
})

const state = {
  data: null,
  dataError: null,
  isLoadingData: false,
  loadingPromise: null,
  inputError: null,
  board: {
    red: [CENTER_POINT],
    blue: [],
    toPlay: "blue",
    tenukiOwner: null,
  },
  overlay: [],
  tenukiEntry: null,
  canonical: null,
  drag: null,
  randomBucket: "le4",
}

function pointKey(q, r) {
  return `${q},${r}`
}

function pointsEqual(a, b) {
  return comparePoints(a, b) === 0
}

function comparePoints(a, b) {
  if (a[0] !== b[0]) {
    return a[0] - b[0]
  }
  return a[1] - b[1]
}

function sortPoints(points) {
  return [...points].sort(comparePoints)
}

function comparePointLists(a, b) {
  const limit = Math.min(a.length, b.length)
  for (let i = 0; i < limit; i += 1) {
    const cmp = comparePoints(a[i], b[i])
    if (cmp !== 0) {
      return cmp
    }
  }
  return a.length - b.length
}

function comparePatterns(aPlus, aMinus, bPlus, bMinus) {
  const plusCmp = comparePointLists(aPlus, bPlus)
  if (plusCmp !== 0) {
    return plusCmp
  }
  return comparePointLists(aMinus, bMinus)
}

function oppositeColor(color) {
  return color === "red" ? "blue" : "red"
}

function rgbaText(rgb, alpha) {
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`
}

function parseInteger(text) {
  if (!/^-?[0-9]+$/.test(text)) {
    throw new Error(`Bad integer '${text}'`)
  }
  return Number.parseInt(text, 10)
}

function parsePointBlock(body) {
  if (body === "") {
    return []
  }
  const points = []
  for (const token of body.split(":")) {
    if (!token.includes(",")) {
      throw new Error(`Bad point '${token}'`)
    }
    const [qText, rText] = token.split(",", 2)
    const point = [parseInteger(qText), parseInteger(rText)]
    if (points.some((other) => comparePoints(other, point) === 0)) {
      throw new Error("Duplicate point in block")
    }
    points.push(point)
  }
  return sortPoints(points)
}

function readDelimitedBlock(text, startIndex, openChar, closeChar) {
  if (text[startIndex] !== openChar) {
    throw new Error(`Expected '${openChar}' at position ${startIndex}`)
  }
  const closeIndex = text.indexOf(closeChar, startIndex + 1)
  if (closeIndex < 0) {
    throw new Error(`Missing closing '${closeChar}'`)
  }
  return {
    points: parsePointBlock(text.slice(startIndex + 1, closeIndex)),
    nextIndex: closeIndex + 1,
  }
}

function assertNoCrossBlockOverlap(plusPoints, minusPoints) {
  const seen = new Set()
  for (const point of [...plusPoints, ...minusPoints]) {
    const key = pointKey(point[0], point[1])
    if (seen.has(key)) {
      throw new Error("Cross-block overlap is not allowed")
    }
    seen.add(key)
  }
}

function parsePattern(rawText) {
  const text = String(rawText || "").replace(/\s+/g, "")
  if (!text) {
    throw new Error("Empty pattern")
  }
  if (!text.startsWith("+")) {
    throw new Error("Pattern must use labeled notation and start with '+'")
  }
  const plusBlock = readDelimitedBlock(text, 1, "[", "]")
  if (text[plusBlock.nextIndex] !== "-") {
    throw new Error("Expected '-' after plus block")
  }
  const minusBlock = readDelimitedBlock(text, plusBlock.nextIndex + 1, "[", "]")
  if (minusBlock.nextIndex !== text.length) {
    throw new Error("Trailing characters after pattern")
  }
  assertNoCrossBlockOverlap(plusBlock.points, minusBlock.points)
  return { plus: plusBlock.points, minus: minusBlock.points }
}

function parseHashPattern(rawHash) {
  const text = String(rawHash || "").replace(/\s+/g, "")
  const hashText = text.startsWith("#") ? text.slice(1) : text
  if (!hashText) {
    throw new Error("Empty hash pattern")
  }
  const plusBlock = readDelimitedBlock(hashText, 0, "(", ")")
  const minusBlock = readDelimitedBlock(hashText, plusBlock.nextIndex, "(", ")")
  if (minusBlock.nextIndex !== hashText.length) {
    throw new Error("Trailing characters after hash pattern")
  }
  assertNoCrossBlockOverlap(plusBlock.points, minusBlock.points)
  return { plus: plusBlock.points, minus: minusBlock.points }
}

function moveCountForPattern(patternText) {
  const parsed = parsePattern(patternText)
  const diff = labeledFamilyDiff(parsed.plus.length, parsed.minus.length)
  if (diff === null) {
    throw new Error("Pattern is not a supported labeled family under red-first play with at most one tenuki")
  }
  return parsed.plus.length + parsed.minus.length + (diff === -1 || diff === 2 ? 1 : 0)
}

function axToCube(point) {
  const [q, r] = point
  return [q, -q - r, r]
}

function cubeToAx(cube) {
  return [cube[0], cube[2]]
}

function rotateCube(cube, turns) {
  let [x, y, z] = cube
  for (let i = 0; i < turns; i += 1) {
    ;[x, y, z] = [-z, -x, -y]
  }
  return [x, y, z]
}

function reflectCube(cube) {
  return [cube[0], cube[2], cube[1]]
}

function applyTransformAx(point, transformId) {
  if (!(transformId >= 0 && transformId < 12)) {
    throw new Error(`Bad transform id: ${transformId}`)
  }
  const cube = axToCube(point)
  const transformed = transformId < 6 ? rotateCube(cube, transformId) : reflectCube(rotateCube(cube, transformId - 6))
  return cubeToAx(transformed)
}

function inverseTransformId(transformId) {
  if (transformId >= 0 && transformId < 6) {
    return (6 - transformId) % 6
  }
  if (transformId >= 6 && transformId < 12) {
    return transformId
  }
  throw new Error(`Bad transform id: ${transformId}`)
}

function normalizeLabeledPoints(plus, minus) {
  const all = [...plus, ...minus]
  if (all.length === 0) {
    throw new Error("Pattern has no stones")
  }
  const anchor = all.reduce((best, point) => (comparePoints(point, best) < 0 ? point : best))
  const shift = (point) => [point[0] - anchor[0], point[1] - anchor[1]]
  return {
    plus: sortPoints(plus.map(shift)),
    minus: sortPoints(minus.map(shift)),
    anchor: [anchor[0], anchor[1]],
  }
}

function formatPointList(points) {
  return points.map((point) => `${point[0]},${point[1]}`).join(":")
}

function formatPattern(plus, minus) {
  return `+[${formatPointList(plus)}]-[${formatPointList(minus)}]`
}

function formatHashPattern(plus, minus) {
  return `(${formatPointList(plus)})(${formatPointList(minus)})`
}

function canonicalizeLabeledPattern(plus, minus) {
  let best = null
  for (let transformId = 0; transformId < 12; transformId += 1) {
    const plusT = plus.map((point) => applyTransformAx(point, transformId))
    const minusT = minus.map((point) => applyTransformAx(point, transformId))
    const normalized = normalizeLabeledPoints(plusT, minusT)
    if (best === null || comparePatterns(normalized.plus, normalized.minus, best.plus, best.minus) < 0) {
      best = {
        pattern: formatPattern(normalized.plus, normalized.minus),
        plus: normalized.plus,
        minus: normalized.minus,
        transformId,
        anchor: normalized.anchor,
      }
    }
  }
  return best
}

function shiftPoints(points, dq, dr) {
  return points.map((point) => [point[0] + dq, point[1] + dr])
}

function labeledFamilyDiff(plusCount, minusCount) {
  const diff = minusCount - plusCount
  return diff >= -1 && diff <= 2 ? diff : null
}

function inferBoardFromLabeledPattern(plus, minus) {
  const diff = labeledFamilyDiff(plus.length, minus.length)
  if (diff !== null) {
    return {
      toPlay: diff <= 0 ? "red" : "blue",
      red: sortPoints(diff <= 0 ? plus : minus),
      blue: sortPoints(diff <= 0 ? minus : plus),
      tenukiOwner: diff === -1 || diff === 2 ? "blue" : null,
    }
  }
  throw new Error("Pattern is not a supported labeled family under red-first play with at most one tenuki")
}

function shiftBoardToCenter(redPoints, bluePoints) {
  const reds = sortPoints(redPoints)
  if (reds.length === 0) {
    throw new Error("Pattern must contain at least one red stone")
  }
  const anchorRed = reds[0]
  const dq = CENTER_POINT[0] - anchorRed[0]
  const dr = CENTER_POINT[1] - anchorRed[1]
  const shiftedRed = shiftPoints(reds, dq, dr)
  const shiftedBlue = shiftPoints(sortPoints(bluePoints), dq, dr)
  if (!shiftedRed.some((point) => comparePoints(point, CENTER_POINT) === 0)) {
    throw new Error("Editor board must contain the fixed center red stone")
  }
  if (shiftedBlue.some((point) => comparePoints(point, CENTER_POINT) === 0)) {
    throw new Error("Blue stone cannot occupy the fixed center")
  }
  return {
    red: shiftedRed,
    blue: shiftedBlue,
  }
}

function pointAtCenter(point) {
  return comparePoints(point, CENTER_POINT) === 0
}

function pointAtTenuki(point) {
  return comparePoints(point, TENUKI_POINT) === 0
}

function setBoardState({ red, blue, toPlay, tenukiOwner = null }) {
  state.board = {
    red: sortPoints(red),
    blue: sortPoints(blue),
    toPlay,
    tenukiOwner,
  }
}

function coloredStones() {
  const out = [{ point: CENTER_POINT, color: "red", fixed: true }]
  for (const point of state.board.red) {
    if (!pointAtCenter(point)) {
      out.push({ point, color: "red", fixed: false })
    }
  }
  for (const point of state.board.blue) {
    out.push({ point, color: "blue", fixed: false })
  }
  return out
}

function occupiedMap() {
  const out = new Map()
  for (const stone of coloredStones()) {
    out.set(pointKey(stone.point[0], stone.point[1]), stone)
  }
  return out
}

function boardPatternState() {
  const redSorted = sortPoints(state.board.red)
  const blueSorted = sortPoints(state.board.blue)
  const toPlay = state.board.toPlay
  const plus = toPlay === "red" ? redSorted : blueSorted
  const minus = toPlay === "red" ? blueSorted : redSorted
  return {
    red: redSorted,
    blue: blueSorted,
    toPlay,
    tenukiOwner: state.board.tenukiOwner,
    plus,
    minus,
    rawPattern: formatPattern(plus, minus),
  }
}

function lookupToPlayForBoardState(boardState) {
  const diff = labeledFamilyDiff(boardState.plus.length, boardState.minus.length)
  return diff !== null ? (diff <= 0 ? "red" : "blue") : boardState.toPlay
}

function lastMoveColorForBoardState(boardState) {
  return oppositeColor(boardState.toPlay)
}

function removableStoneColor() {
  const color = lastMoveColorForBoardState(state.board)
  if (color === "red") {
    return state.board.red.some((point) => !pointAtCenter(point)) ? "red" : null
  }
  return state.board.blue.length > 0 ? "blue" : null
}

function canToggleTenuki(boardState) {
  const diff = boardState.red.length - boardState.blue.length
  if (boardState.tenukiOwner !== null) {
    return boardState.tenukiOwner === lastMoveColorForBoardState(boardState)
  }
  return (
    (diff === 0 && boardState.toPlay === "red")
    || (diff === 1 && boardState.toPlay === "blue")
  )
}

function toggleTenukiState(boardState) {
  if (!canToggleTenuki(boardState)) {
    return null
  }
  if (boardState.tenukiOwner !== null) {
    return {
      red: boardState.red,
      blue: boardState.blue,
      toPlay: oppositeColor(boardState.toPlay),
      tenukiOwner: null,
    }
  }
  return {
    red: boardState.red,
    blue: boardState.blue,
    toPlay: oppositeColor(boardState.toPlay),
    tenukiOwner: boardState.toPlay,
  }
}

function fractionPercent(stoneFraction) {
  return (100 * Number(stoneFraction)).toFixed(1)
}

function readBundleMagic(view) {
  return String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  )
}

function readPackedWordAtBit(view, offset, bitOffset, bits) {
  if (bits === 0) {
    return 0
  }
  const byteOffset = offset + Math.trunc(bitOffset / 8)
  const shift = bitOffset % 8
  let chunk = 0
  for (let idx = 0; idx < 4 && byteOffset + idx < view.byteLength; idx += 1) {
    chunk += view.getUint8(byteOffset + idx) * (2 ** (8 * idx))
  }
  return Math.trunc(chunk / (2 ** shift)) & ((2 ** bits) - 1)
}

function assertPatternBundleSize(condition) {
  if (!condition) {
    throw new Error("Pattern bundle size mismatch")
  }
}

function compactKeyCoordText(word) {
  const q = (word & 0x0F) + PACKED_KEY_COORD_MIN
  const r = ((word >> 4) & 0x0F) + PACKED_KEY_COORD_MIN
  return `${q},${r}`
}

function readUvarintAt(bytes, offset) {
  let value = 0
  let shift = 0
  let nextOffset = Number(offset)
  while (nextOffset < bytes.length) {
    const byte = bytes[nextOffset]
    nextOffset += 1
    value += (byte & 0x7F) * (2 ** shift)
    if (byte < 0x80) {
      return { value, offset: nextOffset }
    }
    shift += 7
  }
  assertPatternBundleSize(false)
}

function decodeCompactPatternKey(keyBlob, keyOffset, keyLength) {
  if (keyLength < 1) {
    throw new Error("Unsupported pattern key slice")
  }
  const countByte = keyBlob[keyOffset]
  const plusCount = countByte & 0x0F
  const minusCount = (countByte >> 4) & 0x0F
  if (keyLength !== 1 + plusCount + minusCount) {
    throw new Error("Unsupported pattern key slice")
  }
  const plus = []
  const minus = []
  let offset = keyOffset + 1
  for (let idx = 0; idx < plusCount; idx += 1) {
    plus.push(compactKeyCoordText(keyBlob[offset]))
    offset += 1
  }
  for (let idx = 0; idx < minusCount; idx += 1) {
    minus.push(compactKeyCoordText(keyBlob[offset]))
    offset += 1
  }
  return `+[${plus.join(":")}]-[${minus.join(":")}]`
}

function compactPatternKeyToPlay(keyBytes) {
  if (!keyBytes || keyBytes.length < 1) {
    throw new Error("Unsupported pattern key slice")
  }
  const plusCount = keyBytes[0] & 0x0F
  const minusCount = (keyBytes[0] >> 4) & 0x0F
  return minusCount - plusCount <= 0 ? "red" : "blue"
}

function concatPatternKeyPrefix(prevKey, prefixLength, suffix) {
  if (prefixLength < 0 || prefixLength > prevKey.length) {
    throw new Error("Unsupported pattern key slice")
  }
  const key = new Uint8Array(prefixLength + suffix.length)
  key.set(prevKey.slice(0, prefixLength), 0)
  key.set(suffix, prefixLength)
  return key
}

function readBitplaneValue(view, offset, valueCount, valueIndex, bits) {
  let value = 0
  for (let bit = 0; bit < bits; bit += 1) {
    value = (value * 2) + readPackedWordAtBit(view, offset, (bit * valueCount) + valueIndex, 1)
  }
  return value
}

function prefixCountsForPresence(present) {
  const prefix = new Uint32Array(present.length + 1)
  for (let idx = 0; idx < present.length; idx += 1) {
    prefix[idx + 1] = prefix[idx] + present[idx]
  }
  return prefix
}

function decodePresenceRow(bytes, offset, patternCount) {
  assertPatternBundleSize(offset < bytes.length)
  const mode = bytes[offset]
  let nextOffset = offset + 1
  const present = new Uint8Array(patternCount)
  if (mode === PRESENCE_MODE_ALL) {
    present.fill(1)
    return {
      present,
      prefix: prefixCountsForPresence(present),
      offset: nextOffset,
    }
  }
  if (mode === PRESENCE_MODE_BITMAP) {
    const bitmapByteLength = Math.ceil(patternCount / 8)
    assertPatternBundleSize(nextOffset + bitmapByteLength <= bytes.length)
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    for (let idx = 0; idx < patternCount; idx += 1) {
      present[idx] = readPackedWordAtBit(view, nextOffset, idx, 1)
    }
    nextOffset += bitmapByteLength
    return {
      present,
      prefix: prefixCountsForPresence(present),
      offset: nextOffset,
    }
  }
  if (mode === PRESENCE_MODE_PRESENT_GAPS || mode === PRESENCE_MODE_ABSENT_GAPS) {
    if (mode === PRESENCE_MODE_ABSENT_GAPS) {
      present.fill(1)
    }
    const count = readUvarintAt(bytes, nextOffset)
    nextOffset = count.offset
    let idx = -1
    for (let row = 0; row < count.value; row += 1) {
      const gap = readUvarintAt(bytes, nextOffset)
      nextOffset = gap.offset
      idx += gap.value + 1
      assertPatternBundleSize(idx >= 0 && idx < patternCount)
      present[idx] = mode === PRESENCE_MODE_PRESENT_GAPS ? 1 : 0
    }
    return {
      present,
      prefix: prefixCountsForPresence(present),
      offset: nextOffset,
    }
  }
  throw new Error("Unsupported pattern presence stream")
}

function normalizeLoadedData(rawBuffer) {
  if (!(rawBuffer instanceof ArrayBuffer) || rawBuffer.byteLength < HEADER_SIZE) {
    throw new Error("Unsupported pattern data format")
  }
  const view = new DataView(rawBuffer)
  if (readBundleMagic(view) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported pattern data format")
  }
  const version = view.getUint16(4, true)
  if (version !== BUNDLE_VERSION) {
    throw new Error("Unsupported pattern data format")
  }
  const fractionMode = view.getUint16(6, true)
  if (fractionMode !== FRACTION_MODE_COORD_MAJOR_BITPLANE) {
    throw new Error("Unsupported pattern fraction stream")
  }
  const patternCount = view.getUint32(8, true)
  const cellCount = view.getUint32(12, true)
  const keyStreamSize = view.getUint32(16, true)
  const tenukiOffset = HEADER_SIZE
  const pairCountOffset = tenukiOffset + Math.ceil((patternCount * PACKED_STONE_FRACTION_BITS) / 8)
  assertPatternBundleSize(rawBuffer.byteLength >= pairCountOffset + PAIR_COUNT_SIZE)
  const pairCount = view.getUint16(pairCountOffset, true)
  if (cellCount > 0 && pairCount === 0) {
    throw new Error("Unsupported pattern pair table")
  }
  const pairTableOffset = pairCountOffset + PAIR_COUNT_SIZE
  const presenceSizeOffset = pairTableOffset + (pairCount * PAIR_ROW_SIZE)
  assertPatternBundleSize(rawBuffer.byteLength >= presenceSizeOffset + 4)
  const presenceByteLength = view.getUint32(presenceSizeOffset, true)
  const presenceOffset = presenceSizeOffset + 4
  const fractionOffset = presenceOffset + presenceByteLength
  const fractionByteLength = Math.ceil((cellCount * PACKED_STONE_FRACTION_BITS) / 8)
  const keyStreamOffset = fractionOffset + fractionByteLength
  const expectedSize = keyStreamOffset + keyStreamSize
  assertPatternBundleSize(rawBuffer.byteLength === expectedSize)
  const pairTable = []
  for (let idx = 0; idx < pairCount; idx += 1) {
    const offset = pairTableOffset + (idx * PAIR_ROW_SIZE)
    pairTable.push([view.getInt8(offset), view.getInt8(offset + 1)])
  }
  const presenceBytes = new Uint8Array(rawBuffer, presenceOffset, presenceByteLength)
  const coordRows = []
  let presenceCursor = 0
  let fractionStart = 0
  for (let idx = 0; idx < pairCount; idx += 1) {
    const row = decodePresenceRow(presenceBytes, presenceCursor, patternCount)
    presenceCursor = row.offset
    const rowCount = row.prefix[patternCount]
    coordRows.push({
      pair: pairTable[idx],
      present: row.present,
      prefix: row.prefix,
      fractionStart,
    })
    fractionStart += rowCount
  }
  assertPatternBundleSize(presenceCursor === presenceBytes.length && fractionStart === cellCount)
  const keyStream = new Uint8Array(rawBuffer, keyStreamOffset, keyStreamSize)
  const patterns = Object.create(null)
  const entries = []
  const entryCache = new Map()
  let keyOffset = 0
  let prevKey = new Uint8Array()
  for (let idx = 0; idx < patternCount; idx += 1) {
    const tenukiStoneFraction = readBitplaneValue(
      view,
      tenukiOffset,
      patternCount,
      idx,
      PACKED_STONE_FRACTION_BITS,
    )
    const prefix = readUvarintAt(keyStream, keyOffset)
    const suffixLength = readUvarintAt(keyStream, prefix.offset)
    keyOffset = suffixLength.offset
    if (keyOffset + suffixLength.value > keyStream.length) {
      throw new Error("Unsupported pattern key slice")
    }
    const suffix = keyStream.slice(keyOffset, keyOffset + suffixLength.value)
    keyOffset += suffixLength.value
    const keyBytes = concatPatternKeyPrefix(prevKey, prefix.value, suffix)
    const pattern = decodeCompactPatternKey(keyBytes, 0, keyBytes.length)
    entries.push({
      keyBytes,
      tenukiStoneFraction,
    })
    patterns[pattern] = idx
    prevKey = keyBytes
  }
  assertPatternBundleSize(keyOffset === keyStream.length)
  return {
    version,
    pattern_count: patternCount,
    rawBuffer,
    coordRows,
    fractionOffset,
    cellCount,
    entries,
    entryCache,
    patterns,
  }
}

function patternEntryForLookup(data, pattern) {
  if (!data || !data.patterns || !Object.prototype.hasOwnProperty.call(data.patterns, pattern)) {
    return null
  }
  const entryIndex = data.patterns[pattern]
  if (data.entryCache.has(entryIndex)) {
    return data.entryCache.get(entryIndex)
  }
  const rawEntry = data.entries[entryIndex]
  if (!rawEntry) {
    throw new Error("Unsupported pattern entry")
  }
  const view = new DataView(data.rawBuffer)
  const cells = []
  for (const row of data.coordRows) {
    if (row.present[entryIndex] === 0) {
      continue
    }
    const fractionIndex = row.fractionStart + row.prefix[entryIndex]
    assertPatternBundleSize(fractionIndex >= 0 && fractionIndex < data.cellCount)
    const stoneFraction = readBitplaneValue(
      view,
      data.fractionOffset,
      data.cellCount,
      fractionIndex,
      PACKED_STONE_FRACTION_BITS,
    )
    const pair = row.pair
    cells.push([pair[0], pair[1], stoneFraction])
  }
  const entry = {
    p: compactPatternKeyToPlay(rawEntry.keyBytes),
    c: cells,
  }
  if (rawEntry.tenukiStoneFraction !== PACKED_OPTIONAL_NULL) {
    entry.t = rawEntry.tenukiStoneFraction
  }
  data.entryCache.set(entryIndex, entry)
  return entry
}

function candidateOverlayForLookup(entry, canonical) {
  const overlay = []
  const inverseId = inverseTransformId(canonical.transformId)
  for (const cell of entry.c || []) {
    if (!Array.isArray(cell) || cell.length !== 3) {
      throw new Error("Unsupported pattern cell format")
    }
    const transformed = [cell[0] + canonical.anchor[0], cell[1] + canonical.anchor[1]]
    const editorPoint = applyTransformAx(transformed, inverseId)
    overlay.push({
      stoneFraction: decodeThousandths(cell[2]),
      point: [editorPoint[0], editorPoint[1]],
    })
  }
  overlay.sort((a, b) => Number(b.stoneFraction) - Number(a.stoneFraction) || comparePoints(a.point, b.point))
  const tenukiValue = entry.t
  return {
    overlay,
    tenukiStoneFraction: decodeThousandths(tenukiValue),
  }
}

function loadedPatternsLabel() {
  const count = Number(state.data?.pattern_count || 0)
  return `${count} pattern${count === 1 ? "" : "s"} loaded`
}

function syncHashFromBoard(boardState) {
  const nextHash = boardState.rawPattern === "+[]-[0,0]" ? "" : `#${formatHashPattern(boardState.plus, boardState.minus)}`
  const nextUrl = `${window.location.pathname}${window.location.search}${nextHash}`
  const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`
  if (currentUrl === nextUrl) {
    return
  }
  window.history.replaceState(null, "", nextUrl)
}

function syncFromBoard({ rewriteInput = false } = {}) {
  const boardState = boardPatternState()
  state.overlay = []
  state.tenukiEntry = null

  if (rewriteInput) {
    elements.patternInput.value = boardState.rawPattern
    state.inputError = null
  }

  const canonical = canonicalizeLabeledPattern(boardState.plus, boardState.minus)
  const lookupToPlay = lookupToPlayForBoardState(boardState)
  state.canonical = {
    ...canonical,
    toPlay: lookupToPlay,
    rawPattern: boardState.rawPattern,
  }
  elements.canonicalPattern.textContent = canonical.pattern
  elements.patternsCount.textContent = state.data ? loadedPatternsLabel() : ""
  elements.turnIndicator.textContent = `Turn: ${boardState.toPlay === "red" ? "Red" : "Blue"}`
  elements.turnIndicator.className = `turn-indicator ${boardState.toPlay === "red" ? "turn-red" : "turn-blue"}`
  if (!state.inputError) {
    syncHashFromBoard(boardState)
  }

  if (state.inputError) {
    elements.lookupStatus.textContent = state.inputError
    renderBoard()
    return
  }

  const entry = patternEntryForLookup(state.data, canonical.pattern)
  if (entry) {
    if (entry.p === lookupToPlay) {
      const mapped = candidateOverlayForLookup(entry, canonical)
      state.overlay = mapped.overlay
      state.tenukiEntry = mapped.tenukiStoneFraction
      elements.lookupStatus.textContent = "Matched precomputed pattern."
    } else {
      elements.lookupStatus.textContent = "Found the pattern, but the side to play did not match."
    }
  } else if (state.dataError) {
    elements.lookupStatus.textContent = "Pattern bundle failed to load."
  } else if (state.data) {
    elements.lookupStatus.textContent = "No precomputed pattern data for this pattern."
  } else if (state.isLoadingData) {
    elements.lookupStatus.textContent = "Loading patterns…"
  } else {
    elements.lookupStatus.textContent = ""
  }

  renderBoard()
}

function pointToPixel(point) {
  const [q, r] = point
  return rawPointToPixel(q, r)
}

function hexBallPoints(radius) {
  const out = []
  for (let q = -radius; q <= radius; q += 1) {
    for (let r = -radius; r <= radius; r += 1) {
      const s = -q - r
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) <= radius) {
        out.push([q, r])
      }
    }
  }
  return out
}

const FIXED_BOARD_POINTS = hexBallPoints(BOARD_RADIUS)

function setupViewBox() {
  const points = [...FIXED_BOARD_POINTS, TENUKI_POINT]
  const pixels = points.map((point) => pointToPixel(point))
  const xs = pixels.map((point) => point[0])
  const ys = pixels.map((point) => point[1])
  const minX = Math.min(...xs) - VIEW_PADDING
  const maxX = Math.max(...xs) + VIEW_PADDING
  const minY = Math.min(...ys) - VIEW_PADDING
  const maxY = Math.max(...ys) + VIEW_PADDING
  elements.board.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`)
}

function pointFromHexElement(element) {
  if (!(element instanceof Element)) {
    return null
  }
  const hex = element.closest("[data-board-point='1']")
  if (!(hex instanceof Element)) {
    return null
  }
  const qText = hex.getAttribute("data-q")
  const rText = hex.getAttribute("data-r")
  if (qText === null || rText === null) {
    return null
  }
  return [Number(qText), Number(rText)]
}

function pointFromClientPosition(clientX, clientY) {
  return pointFromHexElement(document.elementFromPoint(clientX, clientY))
}

function appendHex(point, options = {}) {
  const [cx, cy] = pointToPixel(point)
  const polygon = createSvgNode("polygon")
  polygon.setAttribute("points", hexPolygonPoints(cx, cy, options.size || (HEX_SIZE - 1.5)))
  polygon.setAttribute("class", options.className || "board-hex")
  polygon.setAttribute("fill", options.fill || rgbText(OFF_WHITE_RGB))
  polygon.setAttribute("stroke", options.stroke || GRID_EDGE)
  if (options.boardPoint) {
    polygon.setAttribute("data-board-point", "1")
    polygon.setAttribute("data-q", String(point[0]))
    polygon.setAttribute("data-r", String(point[1]))
  }
  elements.board.appendChild(polygon)
  return { cx, cy, polygon }
}

function beginInteraction(pointerId, point, stone) {
  state.drag = {
    pointerId,
    startPoint: [point[0], point[1]],
    sourceColor: stone ? stone.color : null,
    sourceRemovable: Boolean(stone && stone.color === removableStoneColor()),
    targetPoint: null,
  }
}

function dragTargetFromPoint(sourcePoint, point, occupied) {
  if (point === null) {
    return null
  }
  if (pointAtCenter(point) || pointAtTenuki(point) || pointsEqual(point, sourcePoint)) {
    return null
  }
  if (occupied.has(pointKey(point[0], point[1]))) {
    return null
  }
  return [point[0], point[1]]
}

function updateDragTargetFromClientPosition(clientX, clientY) {
  if (!state.drag || state.drag.sourceColor === null) {
    return
  }
  const occupied = occupiedMap()
  const nextTarget = dragTargetFromPoint(state.drag.startPoint, pointFromClientPosition(clientX, clientY), occupied)
  const targetUnchanged =
    (state.drag.targetPoint === null && nextTarget === null)
    || (state.drag.targetPoint !== null && nextTarget !== null && pointsEqual(state.drag.targetPoint, nextTarget))
  if (targetUnchanged) {
    return
  }
  state.drag.targetPoint = nextTarget
  renderBoard()
}

function rewriteDraggedStone(sourcePoint, targetPoint, color) {
  const boardState = boardPatternState()
  if (color === "red") {
    const nextRed = boardState.red.filter((point) => !pointsEqual(point, sourcePoint))
    nextRed.push(targetPoint)
    return {
      red: sortPoints(nextRed),
      blue: boardState.blue,
      toPlay: boardState.toPlay,
      tenukiOwner: boardState.tenukiOwner,
    }
  }
  const nextBlue = boardState.blue.filter((point) => !pointsEqual(point, sourcePoint))
  nextBlue.push(targetPoint)
  return {
    red: boardState.red,
    blue: sortPoints(nextBlue),
    toPlay: boardState.toPlay,
    tenukiOwner: boardState.tenukiOwner,
  }
}

function removeStoneAtPoint(point) {
  const boardState = boardPatternState()
  if (boardState.red.some((candidate) => pointsEqual(candidate, point)) && !pointAtCenter(point)) {
    setBoardState({
      red: boardState.red.filter((candidate) => !pointsEqual(candidate, point)),
      blue: boardState.blue,
      toPlay: "red",
      tenukiOwner: boardState.tenukiOwner,
    })
    return
  }
  if (boardState.blue.some((candidate) => pointsEqual(candidate, point))) {
    setBoardState({
      red: boardState.red,
      blue: boardState.blue.filter((candidate) => !pointsEqual(candidate, point)),
      toPlay: "blue",
      tenukiOwner: boardState.tenukiOwner,
    })
  }
}

function commitBoardChange() {
  syncFromBoard({ rewriteInput: true })
  void ensureDataLoaded()
}

function renderBoard() {
  clearSvg()
  setupViewBox()
  elements.board.classList.toggle("dragging", Boolean(state.drag && state.drag.sourceColor !== null))

  const boardState = boardPatternState()
  const tenukiMarked = boardState.tenukiOwner !== null
  const occupied = occupiedMap()
  const overlayByKey = new Map(state.overlay.map((row) => [pointKey(row.point[0], row.point[1]), row]))
  const nextColor = boardState.toPlay
  const hoverColor = nextColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
  const hoverFill = nextColor === "red" ? rgbaText(RED_RGB, 0.12) : rgbaText(BLUE_RGB, 0.12)
  const dragSourceKey =
    state.drag && state.drag.sourceColor !== null && state.drag.targetPoint
      ? pointKey(state.drag.startPoint[0], state.drag.startPoint[1])
      : null
  const dragTargetKey = state.drag && state.drag.targetPoint ? pointKey(state.drag.targetPoint[0], state.drag.targetPoint[1]) : null

  for (const point of FIXED_BOARD_POINTS) {
    const key = pointKey(point[0], point[1])
    const stone = occupied.get(key) || null
    const overlay = overlayByKey.get(key) || null
    let fill = rgbText(OFF_WHITE_RGB)
    const classes = ["board-hex"]
    if (overlay) {
      fill = resultFill(overlay.stoneFraction)
      classes.push("candidate")
    }
    if (stone) {
      fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    }
    if (dragSourceKey === key) {
      classes.push("drag-source")
    }
    const hitClasses = ["board-hover-hit"]
    if (stone ? !stone.fixed : !pointAtCenter(point)) {
      hitClasses.push("clickable")
    }
    if (!stone && !pointAtCenter(point)) {
      hitClasses.push("hoverable")
    }
    const hoverHex = appendHex(point, {
      fill: "transparent",
      stroke: "none",
      className: hitClasses.join(" "),
      boardPoint: true,
      size: HEX_SIZE,
    })
    hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
    const hex = appendHex(point, {
      fill,
      className: `${classes.join(" ")} board-hex-face`,
      stroke: overlay ? "none" : GRID_EDGE,
    })
    hex.polygon.style.setProperty("--hover-outline", hoverColor)
    if (overlay && !stone && dragTargetKey !== key) {
      appendText(hex.cx, hex.cy, fractionPercent(overlay.stoneFraction))
    }
  }

  const tenukiHoverHex = appendHex(TENUKI_POINT, {
    fill: "transparent",
    stroke: "none",
    className:
      canToggleTenuki(boardState)
        ? (tenukiMarked ? "board-hover-hit clickable" : "board-hover-hit clickable hoverable")
        : "board-hover-hit",
    boardPoint: true,
    size: HEX_SIZE,
  })
  tenukiHoverHex.polygon.style.setProperty(
    "--hover-fill",
    boardState.toPlay === "red" ? rgbaText(RED_RGB, 0.12) : rgbaText(BLUE_RGB, 0.12),
  )

  const tenukiShaded = !tenukiMarked && state.tenukiEntry !== null
  const tenukiHex = appendHex(TENUKI_POINT, {
    fill:
      tenukiMarked
        ? rgbText(boardState.tenukiOwner === "red" ? RED_RGB : BLUE_RGB)
        : (tenukiShaded ? resultFill(state.tenukiEntry) : rgbText(OFF_WHITE_RGB)),
    className:
      tenukiMarked || tenukiShaded
        ? "board-hex tenuki candidate board-hex-face"
        : "board-hex tenuki board-hex-face",
    stroke: tenukiShaded ? "none" : GRID_EDGE,
  })
  tenukiHex.polygon.style.setProperty(
    "--hover-outline",
    boardState.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB),
  )
  appendText(tenukiHex.cx, tenukiHex.cy - HEX_SIZE * 1.28, "Tenuki", "tenuki-label")
  if (state.tenukiEntry !== null) {
    appendText(
      tenukiHex.cx,
      tenukiHex.cy,
      fractionPercent(state.tenukiEntry),
      "cell-text tenuki-text",
      tenukiMarked ? rgbText(OFF_WHITE_RGB) : null,
    )
  }

  if (state.drag && state.drag.sourceColor !== null && state.drag.targetPoint) {
    const ghostFill = state.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    appendHex(state.drag.startPoint, {
      fill: ghostFill,
      stroke: "none",
      className: "board-ghost board-ghost-source",
    })
    appendHex(state.drag.targetPoint, {
      fill: ghostFill,
      stroke: "none",
      className: "board-ghost board-ghost-target",
    })
  }
}

function releaseBoardPointer(pointerId) {
  if (elements.board.hasPointerCapture(pointerId)) {
    elements.board.releasePointerCapture(pointerId)
  }
}

function handleBoardPointerDown(event) {
  if (state.drag) {
    return
  }
  if (event.button !== 0) {
    return
  }
  const point = pointFromHexElement(event.target)
  if (point === null || pointAtCenter(point)) {
    return
  }
  if (pointAtTenuki(point) && !canToggleTenuki(boardPatternState())) {
    return
  }
  const stone = occupiedMap().get(pointKey(point[0], point[1])) || null
  if (stone && stone.fixed) {
    return
  }
  beginInteraction(event.pointerId, point, stone)
  if (stone && event.pointerType !== "touch") {
    renderBoard()
    elements.board.setPointerCapture(event.pointerId)
  }
}

function handleBoardPointerMove(event) {
  if (!state.drag || event.pointerId !== state.drag.pointerId || state.drag.sourceColor === null || event.pointerType === "touch") {
    return
  }
  updateDragTargetFromClientPosition(event.clientX, event.clientY)
}

function handleBoardPointerUp(event) {
  if (!state.drag || event.pointerId !== state.drag.pointerId) {
    return
  }
  const interaction = state.drag
  const releasePoint = pointFromClientPosition(event.clientX, event.clientY)
  releaseBoardPointer(event.pointerId)
  let changed = false

  if (interaction.sourceColor !== null) {
    let targetPoint = null
    if (event.pointerType !== "touch") {
      const occupied = occupiedMap()
      targetPoint = dragTargetFromPoint(interaction.startPoint, releasePoint, occupied)
    }
    state.drag = null
    if (targetPoint) {
      setBoardState(rewriteDraggedStone(interaction.startPoint, targetPoint, interaction.sourceColor))
      changed = true
    } else if (releasePoint !== null && pointsEqual(releasePoint, interaction.startPoint) && interaction.sourceRemovable) {
      removeStoneAtPoint(interaction.startPoint)
      changed = true
    }
  } else {
    state.drag = null
    if (
      releasePoint !== null
      && pointsEqual(releasePoint, interaction.startPoint)
    ) {
      const boardState = boardPatternState()
      if (pointAtTenuki(releasePoint)) {
        const toggled = toggleTenukiState(boardState)
        if (toggled) {
          setBoardState(toggled)
          changed = true
        }
      } else if (!occupiedMap().has(pointKey(releasePoint[0], releasePoint[1]))) {
        if (boardState.toPlay === "red") {
          setBoardState({
            red: sortPoints([...boardState.red, [releasePoint[0], releasePoint[1]]]),
            blue: boardState.blue,
            toPlay: "blue",
            tenukiOwner: boardState.tenukiOwner,
          })
        } else {
          setBoardState({
            red: boardState.red,
            blue: sortPoints([...boardState.blue, [releasePoint[0], releasePoint[1]]]),
            toPlay: "red",
            tenukiOwner: boardState.tenukiOwner,
          })
        }
        changed = true
      }
    }
  }

  if (changed) {
    commitBoardChange()
  } else {
    renderBoard()
  }
}

function handleBoardPointerCancel(event) {
  if (!state.drag || event.pointerId !== state.drag.pointerId) {
    return
  }
  releaseBoardPointer(event.pointerId)
  state.drag = null
  renderBoard()
}

elements.board.addEventListener("pointerdown", handleBoardPointerDown)
elements.board.addEventListener("pointermove", handleBoardPointerMove)
elements.board.addEventListener("pointerup", handleBoardPointerUp)
elements.board.addEventListener("pointercancel", handleBoardPointerCancel)

function clearBoard() {
  setBoardState({
    red: [CENTER_POINT],
    blue: [],
    toPlay: "blue",
    tenukiOwner: null,
  })
  syncFromBoard({ rewriteInput: true })
}

function applyParsedPattern(parsed, { rewriteInput = false } = {}) {
  const inferred = inferBoardFromLabeledPattern(parsed.plus, parsed.minus)
  const centered = shiftBoardToCenter(inferred.red, inferred.blue)
  setBoardState({
    red: centered.red,
    blue: centered.blue,
    toPlay: inferred.toPlay,
    tenukiOwner: inferred.tenukiOwner,
  })
  state.inputError = null
  if (rewriteInput) {
    elements.patternInput.value = formatPattern(parsed.plus, parsed.minus)
  }
  syncFromBoard()
  void ensureDataLoaded()
}

function loadPatternFromInput() {
  try {
    applyParsedPattern(parsePattern(elements.patternInput.value))
  } catch (error) {
    state.inputError = String(error instanceof Error ? error.message : error)
    elements.lookupStatus.textContent = state.inputError
  }
}

function syncBoardFromLocationHash() {
  if (window.location.hash && window.location.hash !== "#") {
    const hashText = window.location.hash.slice(1)
    try {
      try {
        applyParsedPattern(parsePattern(hashText), { rewriteInput: true })
      } catch (_error) {
        applyParsedPattern(parsePattern(hashText.includes("%") ? decodeURIComponent(hashText) : hashText), { rewriteInput: true })
      }
    } catch (_error) {
      try {
        applyParsedPattern(parseHashPattern(window.location.hash), { rewriteInput: true })
      } catch (error) {
        clearBoard()
        void ensureDataLoaded()
      }
    }
    return
  }
  clearBoard()
  void ensureDataLoaded()
}

async function loadRandomPattern({ maxMoves = null, minMoves = null } = {}) {
  await ensureDataLoaded()
  const patterns = Object.keys(state.data?.patterns || {})
  if (patterns.length === 0) {
    elements.lookupStatus.textContent = state.dataError ? "Pattern bundle failed to load." : "No precomputed patterns available."
    return
  }
  const eligible = patterns.filter((pattern) => {
    const moveCount = moveCountForPattern(pattern)
    if (minMoves !== null && moveCount < minMoves) {
      return false
    }
    if (maxMoves !== null && moveCount > maxMoves) {
      return false
    }
    return true
  })
  if (eligible.length === 0) {
    elements.lookupStatus.textContent = "No patterns available for that random bucket."
    return
  }
  const currentPattern = String(state.canonical?.pattern || "").trim()
  const choices =
    eligible.length > 1 && currentPattern ? eligible.filter((pattern) => pattern !== currentPattern) : eligible
  const pattern = choices[Math.floor(Math.random() * choices.length)]
  elements.patternInput.value = pattern
  loadPatternFromInput()
}

function syncRandomModeUi() {
  const buttons = [
    ["le4", elements.randomLe4Btn],
    ["5", elements.random5Btn],
  ]
  for (const [bucket, button] of buttons) {
    const active = state.randomBucket === bucket
    button.classList.toggle("is-active", active)
    button.setAttribute("aria-pressed", active ? "true" : "false")
  }
}

async function ensureDataLoaded() {
  if (state.data) {
    return
  }
  if (state.loadingPromise) {
    return state.loadingPromise
  }
  state.isLoadingData = true
  syncFromBoard()
  state.loadingPromise = (async () => {
    const currentResponse = await fetch(CURRENT_URL, { cache: "no-store" })
    if (!currentResponse.ok) {
      throw new Error(`HTTP ${currentResponse.status}`)
    }
    const current = await currentResponse.json()
    const bundle = String(current?.bundle || "").trim()
    if (!bundle) {
      throw new Error("Missing bundle filename")
    }
    const bundleResponse = await fetch(`./data/${bundle}`)
    if (!bundleResponse.ok) {
      throw new Error(`HTTP ${bundleResponse.status}`)
    }
    state.data = normalizeLoadedData(await bundleResponse.arrayBuffer())
    state.dataError = null
  })().catch((error) => {
    state.dataError = String(error instanceof Error ? error.message : error)
  }).finally(() => {
    state.isLoadingData = false
    state.loadingPromise = null
    syncFromBoard()
  })
  return state.loadingPromise
}

async function copyCanonicalPattern() {
  const text = String(elements.canonicalPattern.textContent || "").trim()
  if (!text || text === "—") {
    return
  }
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(text)
    } else {
      throw new Error("Clipboard API unavailable")
    }
  } catch (_error) {}
}

elements.clearBtn.addEventListener("click", () => clearBoard())
elements.randomBtn.addEventListener("click", () => {
  if (state.randomBucket === "le4") {
    void loadRandomPattern({ maxMoves: 4 })
    return
  }
  const moveCount = Number.parseInt(state.randomBucket, 10)
  void loadRandomPattern({ minMoves: moveCount, maxMoves: moveCount })
})
elements.randomLe4Btn.addEventListener("click", () => {
  state.randomBucket = "le4"
  syncRandomModeUi()
})
elements.random5Btn.addEventListener("click", () => {
  state.randomBucket = "5"
  syncRandomModeUi()
})
elements.canonicalPattern.addEventListener("click", () => {
  void copyCanonicalPattern()
})
elements.patternInput.addEventListener("input", () => {
  void loadPatternFromInput()
})
for (const button of elements.symmetryButtons) {
  button.addEventListener("click", () => {
    const pattern = String(button.getAttribute("data-pattern") || "").trim()
    if (!pattern) {
      return
    }
    elements.patternInput.value = pattern
    void loadPatternFromInput()
  })
}

window.addEventListener("keydown", (event) => {
  if (shouldIgnoreGlobalKeydown(event)) {
    return
  }
  if (!(event.key === "t" || event.key === "T")) {
    return
  }
  const toggled = toggleTenukiState(boardPatternState())
  if (!toggled) {
    return
  }
  event.preventDefault()
  setBoardState(toggled)
  syncFromBoard({ rewriteInput: true })
})

syncRandomModeUi()
window.addEventListener("hashchange", syncBoardFromLocationHash)
syncBoardFromLocationHash()
