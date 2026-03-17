const MANIFEST_URL = "./data/patterns_current.json"
const BOARD_RADIUS = 6
const HEX_SIZE = 24
const VIEW_PADDING = 38
const CENTER_POINT = [0, 0]
const TENUKI_POINT = [-BOARD_RADIUS - 2, Math.round((2 * BOARD_RADIUS) / 3)]

const {
  copyButtonText,
  createBoardPointerController,
  createModeButtonGroup,
  installSelectorTaps,
  createSvgTools,
  decodeThousandths,
  fetchArrayBuffer,
  fetchJson,
  formatVisits,
  metricHeatFill,
  percentText,
  replaceHash,
  rgbText,
  setCopyButtonValue,
  setTurnStatus,
  shouldIgnoreGlobalKeydown,
  setSvgViewBoxFromPixels,
  THEME,
  turnRgbaText,
} = window.HexStudyUI
const { BLUE_RGB, GRID_EDGE, OFF_WHITE_RGB, RED_RGB } = THEME

const {
  PACKED_DEAD_CELL,
  PACKED_CAPTURED_CELL,
  pointKey,
  pointsEqual,
  comparePoints,
  sortPoints,
  oppositeColor,
  parsePattern,
  parseHashPattern,
  moveCountForPattern,
  applyTransformAx,
  inverseTransformId,
  formatPattern,
  formatHashPattern,
  canonicalizeLabeledPattern,
  shiftPoints,
  labeledFamilyDiff,
  inferBoardFromLabeledPattern,
  normalizeLoadedData,
  patternEntryForLookupInData,
} = window.HexPatternCodec

const elements = {
  board: document.getElementById("board"),
  clearBtn: document.getElementById("clear-btn"),
  randomBtn: document.getElementById("random-btn"),
  randomLe4Btn: document.getElementById("random-le4-btn"),
  random5Btn: document.getElementById("random-5-btn"),
  random6Btn: document.getElementById("random-6-btn"),
  symmetryButtons: Array.from(document.querySelectorAll(".symmetry-btn")),
  turnIndicator: document.getElementById("turn-indicator"),
  patternInput: document.getElementById("pattern-input"),
  canonicalPattern: document.getElementById("canonical-pattern"),
  patternsCount: document.getElementById("patterns-count"),
  lookupStatus: document.getElementById("lookup-status"),
}

const { appendHex, appendText, clear: clearSvg, pointToPixel: rawPointToPixel } = createSvgTools({
  board: elements.board,
  hexSize: HEX_SIZE,
  defaultFill: rgbText(OFF_WHITE_RGB),
  defaultStroke: GRID_EDGE,
  defaultStrokeWidth: "1",
})

const state = {
  manifest: null,
  dataLayers: [],
  loadingPromiseByLayerId: new Map(),
  errorByLayerId: new Map(),
  dataError: null,
  inputError: null,
  board: { red: [CENTER_POINT], blue: [], toPlay: "blue", tenukiOwner: null },
  overlay: [],
  tenukiEntry: null,
  canonical: null,
  drag: null,
  randomBucket: "le4",
}
let boardPointerController = null

const randomBucketControls = createModeButtonGroup({
  state,
  field: "randomBucket",
  rows: [["le4", elements.randomLe4Btn], ["5", elements.random5Btn], ["6", elements.random6Btn]],
  render: () => syncRandomModeUi(),
})

function shiftBoardToCenter(redPoints, bluePoints) {
  const reds = sortPoints(redPoints)
  if (reds.length === 0) { throw new Error("Pattern must contain at least one red stone") }
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
  return { red: shiftedRed, blue: shiftedBlue }
}

function pointAtCenter(point) { return comparePoints(point, CENTER_POINT) === 0 }

function pointAtTenuki(point) { return comparePoints(point, TENUKI_POINT) === 0 }

function setBoardState({ red, blue, toPlay, tenukiOwner = null }) {
  cancelBoardInteractionForBoardChange()
  state.board = { red: sortPoints(red), blue: sortPoints(blue), toPlay, tenukiOwner }
}

function coloredStones() {
  const out = [{ point: CENTER_POINT, color: "red", fixed: true }]
  for (const point of state.board.red) {
    if (!pointAtCenter(point)) { out.push({ point, color: "red", fixed: false }) }
  }
  for (const point of state.board.blue) { out.push({ point, color: "blue", fixed: false }) }
  return out
}

function occupiedMap() {
  const out = new Map()
  for (const stone of coloredStones()) { out.set(pointKey(stone.point[0], stone.point[1]), stone) }
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

function lastMoveColorForBoardState(boardState) { return oppositeColor(boardState.toPlay) }

function removableStoneColor() {
  const color = lastMoveColorForBoardState(state.board)
  if (color === "red") { return state.board.red.some((point) => !pointAtCenter(point)) ? "red" : null }
  return state.board.blue.length > 0 ? "blue" : null
}

function canToggleTenuki(boardState) {
  const diff = boardState.red.length - boardState.blue.length
  if (boardState.tenukiOwner !== null) { return boardState.tenukiOwner === lastMoveColorForBoardState(boardState) }
  return (diff === 0 && boardState.toPlay === "red") || (diff === 1 && boardState.toPlay === "blue")
}

function toggleTenukiState(boardState) {
  if (!canToggleTenuki(boardState)) { return null }
  if (boardState.tenukiOwner !== null) {
    return { red: boardState.red, blue: boardState.blue, toPlay: oppositeColor(boardState.toPlay), tenukiOwner: null }
  }
  return {
    red: boardState.red,
    blue: boardState.blue,
    toPlay: oppositeColor(boardState.toPlay),
    tenukiOwner: boardState.toPlay,
  }
}

function patternEntryForLookup(pattern, layerId) {
  for (const row of state.dataLayers) {
    if (String(row.id) !== String(layerId)) { continue }
    const entry = patternEntryForLookupInData(row.data, pattern)
    if (entry) { return entry }
  }
  return null
}

function candidateOverlayForLookup(entry, canonical) {
  const overlay = []
  const inverseId = inverseTransformId(canonical.transformId)
  for (const cell of entry.c || []) {
    if (!Array.isArray(cell) || cell.length !== 3) { throw new Error("Unsupported pattern cell format") }
    const transformed = [cell[0] + canonical.anchor[0], cell[1] + canonical.anchor[1]]
    const editorPoint = applyTransformAx(transformed, inverseId)
    const unavailable = cell[2] === PACKED_DEAD_CELL || cell[2] === PACKED_CAPTURED_CELL
    overlay.push({
      stoneFraction: unavailable ? 0 : decodeThousandths(cell[2]),
      point: [editorPoint[0], editorPoint[1]],
    })
  }
  overlay.sort((a, b) => Number(b.stoneFraction) - Number(a.stoneFraction) || comparePoints(a.point, b.point))
  const tenukiValue = entry.t
  return { overlay, tenukiStoneFraction: decodeThousandths(tenukiValue) }
}

function loadedPatternsLabel() {
  const total = totalPatternCount()
  return `${formatVisits(total)} pattern${total === 1 ? "" : "s"} in corpus`
}

function loadedPatternCount() {
  return state.dataLayers.reduce((sum, row) => sum + Number(row?.data?.pattern_count || 0), 0)
}

function totalPatternCount() { return Number(state.manifest?.pattern_count || loadedPatternCount() || 0) }

function isPatternLayerLoaded(layerId) { return state.dataLayers.some((row) => String(row.id) === String(layerId)) }

function patternLayerIdForMoveCount(moveCount) { return moveCount === 6 ? "m6" : "base" }

function patternLayerLoadFailed(layerId) { return Boolean(state.dataError || state.errorByLayerId.has(layerId)) }

function patternLayerLoadingLabel(layerId) {
  return layerId === "m6" ? "Loading 6-move patterns..." : "Loading patterns…"
}

function syncPatternLoadStatus() {
  const count = loadedPatternCount()
  elements.patternsCount.textContent = count > 0 ? loadedPatternsLabel() : ""
}

function syncHashFromBoard(boardState) {
  const nextHash =
    boardState.rawPattern === "+[]-[0,0]" ? "" : `#${formatHashPattern(boardState.plus, boardState.minus)}`
  replaceHash(nextHash)
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
  state.canonical = { ...canonical, toPlay: lookupToPlay, rawPattern: boardState.rawPattern }
  setCopyButtonValue(elements.canonicalPattern, canonical.pattern)
  syncPatternLoadStatus()
  setTurnStatus(elements.turnIndicator, boardState.toPlay)
  if (!state.inputError) { syncHashFromBoard(boardState) }

  if (state.inputError) {
    elements.lookupStatus.textContent = state.inputError
    renderBoard()
    return
  }

  const moveCount = moveCountForPattern(canonical.pattern)
  const layerId = patternLayerIdForMoveCount(moveCount)
  const entry = patternEntryForLookup(canonical.pattern, layerId)
  if (entry) {
    if (entry.p === lookupToPlay) {
      const mapped = candidateOverlayForLookup(entry, canonical)
      state.overlay = mapped.overlay
      state.tenukiEntry = mapped.tenukiStoneFraction
      elements.lookupStatus.textContent = "Matched precomputed pattern."
    } else { elements.lookupStatus.textContent = "Found the pattern, but the side to play did not match." }
  } else if (patternLayerLoadFailed(layerId)) {
    elements.lookupStatus.textContent = "Pattern bundle failed to load."
  } else if (!isPatternLayerLoaded(layerId)) {
    elements.lookupStatus.textContent = patternLayerLoadingLabel(layerId)
  } else { elements.lookupStatus.textContent = "No precomputed pattern data for this pattern." }

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
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) <= radius) { out.push([q, r]) }
    }
  }
  return out
}

const FIXED_BOARD_POINTS = hexBallPoints(BOARD_RADIUS)

function setupViewBox() {
  const points = [...FIXED_BOARD_POINTS, TENUKI_POINT]
  const pixels = points.map((point) => pointToPixel(point))
  setSvgViewBoxFromPixels(elements.board, pixels, VIEW_PADDING)
}

function pointFromHexElement(element) {
  if (!(element instanceof Element)) { return null }
  const hex = element.closest("[data-board-point='1']")
  if (!(hex instanceof Element)) { return null }
  const qText = hex.getAttribute("data-q")
  const rText = hex.getAttribute("data-r")
  if (qText === null || rText === null) { return null }
  return [Number(qText), Number(rText)]
}

function pointFromClientPosition(clientX, clientY) {
  return pointFromHexElement(document.elementFromPoint(clientX, clientY))
}

function patternDragData(point) {
  if (pointAtCenter(point) || pointAtTenuki(point)) { return null }
  const stone = occupiedMap().get(pointKey(point[0], point[1])) || null
  return stone && !stone.fixed ? { sourceColor: stone.color } : null
}

function dragTargetFromPoint(sourcePoint, point, occupied) {
  if (point === null) { return null }
  if (pointAtCenter(point) || pointAtTenuki(point) || pointsEqual(point, sourcePoint)) { return null }
  if (occupied.has(pointKey(point[0], point[1]))) { return null }
  return [point[0], point[1]]
}

function beginPatternDrag(interaction) {
  state.drag = {
    pointerId: interaction.pointerId,
    startPoint: interaction.startPoint,
    sourceColor: interaction.dragData.sourceColor,
    targetPoint: null,
  }
  renderBoard()
}

function movePatternDrag(_interaction, point) {
  if (!state.drag) { return }
  const occupied = occupiedMap()
  const nextTarget = dragTargetFromPoint(state.drag.startPoint, point, occupied)
  const targetUnchanged =
    (state.drag.targetPoint === null && nextTarget === null) ||
    (state.drag.targetPoint !== null && nextTarget !== null && pointsEqual(state.drag.targetPoint, nextTarget))
  if (targetUnchanged) { return }
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
  void preloadAllPatternLayers()
}

function activateBoardPoint(point) {
  if (point === null || pointAtCenter(point)) { return }
  const boardState = boardPatternState()
  if (pointAtTenuki(point)) {
    const toggled = toggleTenukiState(boardState)
    if (toggled) {
      setBoardState(toggled)
      commitBoardChange()
    }
    return
  }

  const stone = occupiedMap().get(pointKey(point[0], point[1])) || null
  if (stone) {
    if (!stone.fixed && stone.color === removableStoneColor()) {
      removeStoneAtPoint(point)
      commitBoardChange()
    }
    return
  }

  if (boardState.toPlay === "red") {
    setBoardState({
      red: sortPoints([...boardState.red, [point[0], point[1]]]),
      blue: boardState.blue,
      toPlay: "blue",
      tenukiOwner: boardState.tenukiOwner,
    })
  } else {
    setBoardState({
      red: boardState.red,
      blue: sortPoints([...boardState.blue, [point[0], point[1]]]),
      toPlay: "red",
      tenukiOwner: boardState.tenukiOwner,
    })
  }
  commitBoardChange()
}

function renderBoard() {
  clearSvg()
  setupViewBox()
  const dragActive = Boolean(state.drag)
  elements.board.classList.toggle("dragging", dragActive)

  const boardState = boardPatternState()
  const tenukiMarked = boardState.tenukiOwner !== null
  const occupied = occupiedMap()
  const overlayByKey = new Map(state.overlay.map((row) => [pointKey(row.point[0], row.point[1]), row]))
  const nextColor = boardState.toPlay
  const hoverColor = nextColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
  const hoverFill = turnRgbaText(nextColor, 0.12)
  const dragSourceKey = dragActive ? pointKey(state.drag.startPoint[0], state.drag.startPoint[1]) : null
  const dragTargetKey =
    dragActive && state.drag.targetPoint ? pointKey(state.drag.targetPoint[0], state.drag.targetPoint[1]) : null

  for (const point of FIXED_BOARD_POINTS) {
    const key = pointKey(point[0], point[1])
    const stone = occupied.get(key) || null
    const overlay = overlayByKey.get(key) || null
    let fill = rgbText(OFF_WHITE_RGB)
    const classes = ["board-hex"]
    if (overlay) {
      fill = metricHeatFill(overlay.stoneFraction)
      classes.push("candidate")
    }
    if (stone) { fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB) }
    if (dragSourceKey === key) { classes.push("drag-source") }
    const hitClasses = ["board-hover-hit"]
    if (stone ? !stone.fixed : !pointAtCenter(point)) { hitClasses.push("clickable") }
    if (stone && !stone.fixed) { hitClasses.push("board-drag-hit") }
    if (!stone && !pointAtCenter(point)) { hitClasses.push("hoverable") }
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
    if (overlay && !stone && dragTargetKey !== key) { appendText(hex.cx, hex.cy, percentText(overlay.stoneFraction)) }
  }

  const tenukiHoverHex = appendHex(TENUKI_POINT, {
    fill: "transparent",
    stroke: "none",
    className: canToggleTenuki(boardState)
      ? tenukiMarked
        ? "board-hover-hit clickable"
        : "board-hover-hit clickable hoverable"
      : "board-hover-hit",
    boardPoint: true,
    size: HEX_SIZE,
  })
  tenukiHoverHex.polygon.style.setProperty("--hover-fill", turnRgbaText(boardState.toPlay, 0.12))

  const tenukiShaded = !tenukiMarked && state.tenukiEntry !== null
  const tenukiHex = appendHex(TENUKI_POINT, {
    fill: tenukiMarked
      ? rgbText(boardState.tenukiOwner === "red" ? RED_RGB : BLUE_RGB)
      : tenukiShaded
        ? metricHeatFill(state.tenukiEntry)
        : rgbText(OFF_WHITE_RGB),
    className:
      tenukiMarked || tenukiShaded ? "board-hex tenuki candidate board-hex-face" : "board-hex tenuki board-hex-face",
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
      percentText(state.tenukiEntry),
      "cell-text tenuki-text",
      tenukiMarked ? rgbText(OFF_WHITE_RGB) : null,
    )
  }

  if (dragActive && state.drag.targetPoint) {
    const ghostFill = state.drag.sourceColor === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    appendHex(state.drag.targetPoint, { fill: ghostFill, stroke: "none", className: "board-ghost" })
  }
}

function cancelBoardInteractionForBoardChange() {
  boardPointerController?.cancel({ notify: false })
  state.drag = null
}

function finishPatternDrag(interaction, releasePoint) {
  const occupied = occupiedMap()
  const targetPoint = dragTargetFromPoint(interaction.startPoint, releasePoint, occupied)
  state.drag = null
  if (targetPoint) {
    setBoardState(rewriteDraggedStone(interaction.startPoint, targetPoint, interaction.dragData.sourceColor))
    commitBoardChange()
  } else { renderBoard() }
}

function cancelPatternPointerInteraction(interaction) {
  if (interaction.dragging) {
    state.drag = null
    renderBoard()
  }
}

boardPointerController = createBoardPointerController({
  board: elements.board,
  pointFromTarget: pointFromHexElement,
  pointFromClientPosition,
  pointsEqual,
  dragDataForPoint: patternDragData,
  onTap: activateBoardPoint,
  onDragStart: beginPatternDrag,
  onDragMove: movePatternDrag,
  onDrop: finishPatternDrag,
  onCancel: cancelPatternPointerInteraction,
})

function clearBoard() {
  setBoardState({ red: [CENTER_POINT], blue: [], toPlay: "blue", tenukiOwner: null })
  syncFromBoard({ rewriteInput: true })
}

function applyParsedPattern(parsed, { rewriteInput = false } = {}) {
  const inferred = inferBoardFromLabeledPattern(parsed.plus, parsed.minus)
  const centered = shiftBoardToCenter(inferred.red, inferred.blue)
  setBoardState({ red: centered.red, blue: centered.blue, toPlay: inferred.toPlay, tenukiOwner: inferred.tenukiOwner })
  state.inputError = null
  if (rewriteInput) { elements.patternInput.value = formatPattern(parsed.plus, parsed.minus) }
  syncFromBoard()
  void preloadAllPatternLayers()
}

function loadPatternFromInput() {
  try { applyParsedPattern(parsePattern(elements.patternInput.value)) } catch (error) {
    state.inputError = String(error instanceof Error ? error.message : error)
    elements.lookupStatus.textContent = state.inputError
  }
}

function syncBoardFromLocationHash() {
  if (window.location.hash && window.location.hash !== "#") {
    const hashText = window.location.hash.slice(1)
    try {
      try { applyParsedPattern(parsePattern(hashText), { rewriteInput: true }) } catch (_error) {
        applyParsedPattern(parsePattern(hashText.includes("%") ? decodeURIComponent(hashText) : hashText), {
          rewriteInput: true,
        })
      }
    } catch (_error) {
      try { applyParsedPattern(parseHashPattern(window.location.hash), { rewriteInput: true }) } catch (error) {
        clearBoard()
        void preloadAllPatternLayers()
      }
    }
    return
  }
  clearBoard()
  void preloadAllPatternLayers()
}

async function loadRandomPattern(bucket = state.randomBucket) {
  await preloadAllPatternLayers()
  const layerId = bucket === "6" ? "m6" : "base"
  await ensurePatternLayerLoaded(layerId)
  const patterns = state.dataLayers.find((row) => row.id === layerId)?.data.randomPatterns[bucket] || []
  if (patterns.length === 0) {
    elements.lookupStatus.textContent = patternLayerLoadFailed(layerId)
      ? "Pattern bundle failed to load."
      : "No patterns available for that random bucket."
    return
  }
  const currentPattern = String(state.canonical?.pattern || "").trim()
  let index = Math.floor(Math.random() * patterns.length)
  if (patterns.length > 1 && patterns[index] === currentPattern) {
    index = (index + 1 + Math.floor(Math.random() * (patterns.length - 1))) % patterns.length
  }
  const pattern = patterns[index]
  elements.patternInput.value = pattern
  loadPatternFromInput()
}

function syncRandomModeUi() { randomBucketControls.sync() }

function manifestLayers() {
  const layers = state.manifest?.layers
  return Array.isArray(layers)
    ? layers
        .filter((layer) => layer && typeof layer.bundle === "string" && typeof layer.id === "string")
        .map((layer) => ({ ...layer, id: String(layer.id) }))
    : []
}

async function ensurePatternManifestLoaded() {
  if (state.manifest) { return state.manifest }
  state.manifest = await fetchJson(MANIFEST_URL, { cache: "no-store" })
  return state.manifest
}

function patternLayerById(layerId) {
  return manifestLayers().find((layer) => String(layer.id) === String(layerId)) || null
}

async function ensurePatternLayerLoaded(layerId) {
  const layerIdText = String(layerId || "").trim()
  if (!layerIdText || isPatternLayerLoaded(layerIdText)) { return }
  const existingPromise = state.loadingPromiseByLayerId.get(layerIdText)
  if (existingPromise) { return existingPromise }
  state.errorByLayerId.delete(layerIdText)
  syncPatternLoadStatus()
  const promise = (async () => {
    await ensurePatternManifestLoaded()
    const layer = patternLayerById(layerIdText)
    if (!layer) { throw new Error(`Missing pattern layer: ${layerIdText}`) }
    const dataUrl = new URL(String(layer.bundle), new URL(MANIFEST_URL, window.location.href)).toString()
    const data = normalizeLoadedData(await fetchArrayBuffer(dataUrl))
    if (!isPatternLayerLoaded(layerIdText)) { state.dataLayers.push({ id: layerIdText, data }) }
    state.errorByLayerId.delete(layerIdText)
    state.dataError = null
  })()
    .catch((error) => { state.errorByLayerId.set(layerIdText, String(error instanceof Error ? error.message : error)) })
    .finally(() => {
      state.loadingPromiseByLayerId.delete(layerIdText)
      syncPatternLoadStatus()
      syncFromBoard()
    })
  state.loadingPromiseByLayerId.set(layerIdText, promise)
  return promise
}

async function preloadAllPatternLayers() {
  const manifest = await ensurePatternManifestLoaded().catch((error) => {
    state.dataError = String(error instanceof Error ? error.message : error)
    return null
  })
  if (manifest) {
    state.dataError = null
    for (const layer of manifestLayers()) { void ensurePatternLayerLoaded(layer.id) }
  }
  syncFromBoard()
}

async function copyCanonicalPattern() {
  const text = String(elements.canonicalPattern.textContent || "").trim()
  if (!text || text === "—") { return }
  await copyButtonText(elements.canonicalPattern, text)
}

elements.clearBtn.addEventListener("click", () => clearBoard())
elements.randomBtn.addEventListener("click", () => { void loadRandomPattern() })
elements.randomLe4Btn.addEventListener("click", () => { randomBucketControls.set("le4") })
elements.random5Btn.addEventListener("click", () => { randomBucketControls.set("5") })
elements.random6Btn.addEventListener("click", () => { randomBucketControls.set("6") })
elements.canonicalPattern.addEventListener("click", () => { void copyCanonicalPattern() })
elements.patternInput.addEventListener("input", () => { void loadPatternFromInput() })
for (const button of elements.symmetryButtons) {
  button.addEventListener("click", () => {
    const pattern = String(button.getAttribute("data-pattern") || "").trim()
    if (!pattern) { return }
    elements.patternInput.value = pattern
    void loadPatternFromInput()
  })
}

window.addEventListener("keydown", (event) => {
  if (shouldIgnoreGlobalKeydown(event)) { return }
  if (!(event.key === "t" || event.key === "T")) { return }
  const toggled = toggleTenukiState(boardPatternState())
  if (!toggled) { return }
  event.preventDefault()
  setBoardState(toggled)
  syncFromBoard({ rewriteInput: true })
})

installSelectorTaps()
syncRandomModeUi()
window.addEventListener("hashchange", syncBoardFromLocationHash)
syncBoardFromLocationHash()
