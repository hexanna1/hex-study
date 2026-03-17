const SVG_NS = "http://www.w3.org/2000/svg"
const MANIFEST_URL = "./data/matches_current.json"
const BUNDLE_MAGIC = "HMB1"
const BUNDLE_VERSION = 1
const BUNDLE_HEADER_SIZE = 12
const PACKED_OPTIONAL_NULL = 1023
const MOVE_PASS = 65534
const MOVE_SWAP = 65535
const {
  appendMoveToLine,
  cellIdToMove,
  createBoardSvg,
  compactMoveStreamFromLine,
  parseMoves,
  formatLine,
  lineParent,
  normalizeLine,
  renderHexWorldLink,
  renderMoveTreeBoard,
  renderSideActionHex,
  swapControlPoint,
  tryParseCell,
} = window.HexMoveTree
const {
  createModeButtonGroup,
  decodeOptionalThousandths,
  fetchArrayBuffer,
  fetchJson,
  makeResultFill,
  formatVisits,
  handleStandardKeydown,
  hexataCandidateFill,
  hexWorldUrlWithCursor,
  installHoldButton,
  navButtonDisabled,
  percentText,
  readAsciiMagic,
  replaceHash,
  renderMoveList,
  rgbText,
  setNavButtonDisabled,
  setTurnStatus,
  shouldIgnoreGlobalKeydown,
} = window.HexStudyUI
const { THEME } = window.HexMoveTree
const { BLUE_RGB, CANDIDATE_HIGH, CANDIDATE_LOW, OFF_WHITE_RGB, RED_RGB } = THEME

const elements = {
  board: document.getElementById("board"),
  status: document.getElementById("match-status"),
  gameSelect: document.getElementById("game-select"),
  viewWinrateBtn: document.getElementById("view-winrate-btn"),
  viewPriorBtn: document.getElementById("view-prior-btn"),
  playerNames: document.getElementById("player-names"),
  moveList: document.getElementById("move-list"),
  matchNav: document.getElementById("match-nav"),
  moveFirstBtn: document.getElementById("move-first-btn"),
  movePrevBtn: document.getElementById("move-prev-btn"),
  moveNextBtn: document.getElementById("move-next-btn"),
  moveLastBtn: document.getElementById("move-last-btn"),
  gamePrevBtn: document.getElementById("game-prev-btn"),
  gameNextBtn: document.getElementById("game-next-btn"),
  shortcutHint: document.getElementById("shortcut-hint"),
  externalLinks: document.getElementById("external-links"),
  evalGraphWrap: document.getElementById("eval-graph-wrap"),
  evalGraph: document.getElementById("eval-graph"),
  evalGraphTooltip: document.getElementById("eval-graph-tooltip"),
}

const boardSvg = createBoardSvg(elements.board)
const resultFill = makeResultFill(CANDIDATE_LOW, CANDIDATE_HIGH)

const state = {
  games: [],
  gamesByIndex: new Map(),
  defaultBoardSize: 14,
  currentGameIndex: null,
  currentStepIndex: 0,
  overlayTextMode: "winrate",
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

function parseOpeningLine(raw) {
  const moves = String(raw || "")
    .split(",")
    .map((token) => String(token || "").trim().toLowerCase())
    .filter(Boolean)
  return formatLine(moves)
}

function createSvgNode(tagName) {
  return document.createElementNS(SVG_NS, tagName)
}

function currentGame() {
  return state.gamesByIndex.get(Number(state.currentGameIndex)) || null
}

function currentBoardSize() {
  return Number(currentGame()?.board_size || state.defaultBoardSize || 14)
}

function sanitizeLine(line, boardSize = currentBoardSize()) {
  return normalizeLine(line, Number(boardSize))
}

function currentStep(game = currentGame()) {
  const steps = Array.isArray(game?.steps) ? game.steps : []
  if (steps.length === 0) {
    return null
  }
  const idx = Math.max(0, Math.min(Number(state.currentStepIndex) || 0, steps.length - 1))
  return steps[idx]
}

function currentLine(game = currentGame()) {
  return currentStep(game)?.line || ""
}

function currentNode(game = currentGame()) {
  const line = currentLine(game)
  return game?.nodesByLine?.get(line) || game?.nodesByLine?.get(game?.finalLine) || { line, candidates: [] }
}

function setHashState({ gameIndex = state.currentGameIndex, stepIndex = state.currentStepIndex } = {}) {
  const normalizedGameIndex = Number(gameIndex)
  const game = state.gamesByIndex.get(normalizedGameIndex) || currentGame()
  if (!game || !Number.isInteger(normalizedGameIndex) || normalizedGameIndex <= 0) {
    return
  }
  const normalizedStepIndex = Number(stepIndex)
  const defaultStepIndex = defaultCursorStepIndex(game)
  const suffix = (
    Number.isInteger(normalizedStepIndex)
    && normalizedStepIndex !== defaultStepIndex
  ) ? `:${normalizedStepIndex}` : ""
  const nextHash = `${normalizedGameIndex}${suffix}`
  replaceHash(nextHash)
}

function parseHashState() {
  const hasHash = Boolean(window.location.hash)
  const raw = hasHash ? decodeURIComponent(window.location.hash.slice(1)).trim().toLowerCase() : ""
  const match = /^([1-9][0-9]*)(?::([0-9]+))?$/.exec(raw)
  if (!match) {
    return {
      gameIndex: null,
      stepIndex: null,
      hasHash,
    }
  }
  return {
    gameIndex: Number(match[1]),
    stepIndex: match[2] === undefined ? null : Number(match[2]),
    hasHash,
  }
}

function moverWinrate(row, side) {
  if (typeof row?.red_winrate !== "number") {
    return null
  }
  return side === "blue" ? (1.0 - row.red_winrate) : row.red_winrate
}

function childLineForCandidate(_line, row) {
  return typeof row?.childLine === "string" && row.childLine ? row.childLine : null
}

function stepIndexForLine(game, line) {
  const normalized = closestGameLine(game, line)
  const lineIndex = game.steps.findIndex((step) => step.kind !== "terminal" && step.line === normalized)
  return lineIndex >= 0 ? lineIndex : 0
}

function setCursorStep(index) {
  const game = currentGame()
  if (!game) {
    return
  }
  const stepIndex = Number(index)
  if (!Number.isInteger(stepIndex) || stepIndex < 0 || stepIndex >= game.steps.length) {
    return
  }
  state.currentStepIndex = stepIndex
  setHashState()
  render()
}

function setCursorLine(line) {
  const game = currentGame()
  if (!game) {
    return
  }
  state.currentStepIndex = stepIndexForLine(game, line)
  setHashState()
  render()
}

function goToStepIndex(index) {
  setCursorStep(index)
}

function stepCursor(delta) {
  const game = currentGame()
  if (!game) {
    return false
  }
  const nextStepIndex = state.currentStepIndex + Number(delta)
  if (!Number.isInteger(nextStepIndex) || nextStepIndex < 0 || nextStepIndex >= game.steps.length) {
    return false
  }
  state.currentStepIndex = nextStepIndex
  setHashState()
  render()
  return true
}

function goPrevious() {
  stepCursor(-1)
}

function goNext() {
  stepCursor(1)
}

function goFirst() {
  const game = currentGame()
  if (!game || game.lineHistory.length === 0) {
    return
  }
  state.currentStepIndex = 0
  setHashState()
  render()
}

function goLast() {
  const game = currentGame()
  if (!game || game.lineHistory.length === 0) {
    return
  }
  state.currentStepIndex = Math.max(0, game.steps.length - 1)
  setHashState()
  render()
}

function goToLine(line) {
  setCursorLine(line)
}

function closestGameLine(game, line) {
  let normalized = sanitizeLine(line, game.board_size)
  while (normalized && !game.nodesByLine.has(normalized) && !game.lineSet.has(normalized)) {
    normalized = sanitizeLine(lineParent(normalized), game.board_size)
  }
  return game.lineSet.has(normalized) ? normalized : game.openingLine
}

function normalizeStepIndex(game, stepIndex) {
  const idx = Number(stepIndex)
  if (!Number.isInteger(idx) || idx < 0 || idx >= game.steps.length) {
    return 0
  }
  return idx
}

function defaultCursorStepIndex(game) {
  const moves = parseMoves(game?.finalLine || "")
  if (moves.length >= 2 && moves[1] === "swap") {
    return normalizeStepIndex(game, 2)
  }
  if (moves.length >= 1) {
    return normalizeStepIndex(game, 1)
  }
  return 0
}

function defaultCursorLine(game) {
  const step = game?.steps?.[defaultCursorStepIndex(game)] || null
  return typeof step?.line === "string" ? step.line : (game?.openingLine || "")
}

function setCurrentGame(gameIndex, line = null, { updateHash = true, stepIndex = null } = {}) {
  const requestedGame = state.gamesByIndex.get(Number(gameIndex)) || null
  const game = requestedGame || state.games[0] || null
  if (!game) {
    return
  }
  state.currentGameIndex = game.game_index
  if (requestedGame && stepIndex !== null && stepIndex !== undefined) {
    state.currentStepIndex = normalizeStepIndex(game, stepIndex)
  } else {
    const targetLine = line === null || line === undefined ? defaultCursorLine(game) : line
    const nextLine = closestGameLine(game, targetLine)
    state.currentStepIndex = stepIndexForLine(game, nextLine)
  }
  if (updateHash) {
    setHashState()
  }
  render()
}

function buildGameFromRecord(record, gameIndex) {
  return buildGame({
    record,
    meta: {
      red: String(record?.red || "").trim(),
      blue: String(record?.blue || "").trim(),
      opening: String(record?.opening || "").trim(),
      result: String(record?.result || "").trim().toLowerCase(),
      url: String(record?.url || "").trim(),
    },
    openingLine: parseOpeningLine(record?.opening),
    moves: Array.isArray(record?.plies) ? record.plies : [],
    final: record?.final || null,
    kind: String(record?.kind || "").trim().toLowerCase(),
    gameIndex,
  })
}

function playedSideForPly(ply) {
  return Number(ply) % 2 === 1 ? "red" : "blue"
}

function playedCandidate(candidates, played) {
  const playedMove = String(played || "").trim().toLowerCase()
  return candidates.find((row) => String(row?.move || "").trim().toLowerCase() === playedMove) || null
}

function appendMatchGraphPoint(graphPoints, moveEntry, candidates, childLine) {
  const row = playedCandidate(candidates, moveEntry?.played)
  if (typeof row?.red_winrate !== "number") {
    return
  }
  const ply = parseMoves(childLine).length
  graphPoints.push({
    ply,
    line: childLine,
    move: String(moveEntry?.played || "").trim().toLowerCase(),
    side: String(moveEntry?.side || "").trim().toLowerCase() || playedSideForPly(ply),
    winrate: row.red_winrate,
  })
}

function appendBatchGraphPoint(graphPoints, analyzeEntry, line) {
  if (!line) {
    return
  }
  const redWinrate = analyzeEntry?.analyze?.best?.red_winrate
  if (typeof redWinrate !== "number") {
    return
  }
  const moves = parseMoves(line)
  const ply = moves.length
  if (ply <= 0) {
    return
  }
  graphPoints.push({
    ply,
    line,
    move: moves[moves.length - 1] || "",
    side: playedSideForPly(ply),
    winrate: redWinrate,
  })
}

function buildGame({ record, meta, openingLine, moves, final, kind, gameIndex }) {
  const boardSize = Number(record?.board_size || state.defaultBoardSize || 14)
  const startLine = ""
  const lineHistory = [startLine]
  const nodes = []
  const lineSet = new Set([startLine])
  const graphPoints = []
  let line = startLine

  for (const moveEntry of moves) {
    const played = String(moveEntry?.played || "").trim().toLowerCase()
    if (!played) {
      continue
    }
    const rawChildren = Array.isArray(moveEntry?.analyze?.moves) ? moveEntry.analyze.moves : []
    const childLine = appendMoveToLine(line, played)
    const candidates = rawChildren.map((row) => ({
      ...row,
      move: String(row?.move || "").trim().toLowerCase(),
      retained: true,
      childLine: String(row?.move || "").trim().toLowerCase() === played ? childLine : null,
    }))
    nodes.push({
      line,
      ply: Number(moveEntry?.ply || (parseMoves(childLine).length)),
      side: String(moveEntry?.side || "").trim().toLowerCase(),
      played,
      playedChildLine: childLine,
      candidates,
    })
    if (kind === "match") {
      appendMatchGraphPoint(graphPoints, moveEntry, candidates, childLine)
    } else {
      appendBatchGraphPoint(graphPoints, moveEntry, line)
    }
    line = childLine
    lineHistory.push(line)
    lineSet.add(line)
  }

  const finalMoves = parseMoves(line)
  const finalAnalyze = final?.analyze || null
  const finalCandidates = Array.isArray(finalAnalyze?.moves) ? finalAnalyze.moves.map((row) => ({
    ...row,
    move: String(row?.move || "").trim().toLowerCase(),
    retained: true,
    childLine: null,
  })) : []
  nodes.push({
    line,
    ply: finalMoves.length,
    side: String(final?.side || "").trim().toLowerCase() || (finalMoves.length % 2 === 0 ? "red" : "blue"),
    candidates: finalCandidates,
  })
  if (finalAnalyze && kind !== "match") {
    appendBatchGraphPoint(graphPoints, final, line)
  }

  const steps = lineHistory.map((historyLine, index) => {
    const historyMoves = parseMoves(historyLine)
    const text = historyMoves[historyMoves.length - 1] || ""
    return {
      index,
      kind: index === 0 ? "root" : "move",
      line: historyLine,
      text,
      ply: historyMoves.length,
      moveListCount: historyMoves.length,
    }
  })
  if (resignedSideForResult(meta)) {
    steps.push({
      index: steps.length,
      kind: "terminal",
      line,
      text: "resign",
      ply: finalMoves.length + 1,
      moveListCount: finalMoves.length + 1,
    })
  }
  for (const point of graphPoints) {
    const pointStep = steps.find((step) => step.kind !== "terminal" && step.line === point.line)
    point.stepIndex = pointStep?.index ?? 0
  }

  const nodesByLine = new Map()
  for (const node of nodes) {
    nodesByLine.set(node.line, node)
  }
  const normalizedOpeningLine = lineSet.has(openingLine) ? openingLine : startLine

  return {
    ...record,
    ...meta,
    game_index: gameIndex,
    board_size: boardSize,
    openingLine: normalizedOpeningLine,
    finalLine: line,
    lineHistory,
    lineSet,
    steps,
    graphPoints,
    nodes,
    nodesByLine,
  }
}

class MatchBundleReader {
  constructor(buffer) {
    this.buffer = buffer
    this.view = new DataView(buffer)
    this.offset = 0
    this.decoder = new TextDecoder()
  }

  requireBytes(byteCount) {
    if (this.offset + byteCount > this.view.byteLength) {
      throw new Error("Match bundle ended unexpectedly")
    }
  }

  readUint8() {
    this.requireBytes(1)
    const value = this.view.getUint8(this.offset)
    this.offset += 1
    return value
  }

  readUint16() {
    this.requireBytes(2)
    const value = this.view.getUint16(this.offset, true)
    this.offset += 2
    return value
  }

  readUint32() {
    this.requireBytes(4)
    const value = this.view.getUint32(this.offset, true)
    this.offset += 4
    return value
  }

  readVaruint() {
    let value = 0
    let shift = 0
    while (shift <= 49) {
      const byte = this.readUint8()
      value += (byte & 0x7F) * (2 ** shift)
      if ((byte & 0x80) === 0) {
        return value
      }
      shift += 7
    }
    throw new Error("Match bundle varuint is too large")
  }

  readString() {
    const byteLength = this.readVaruint()
    this.requireBytes(byteLength)
    const bytes = new Uint8Array(this.buffer, this.offset, byteLength)
    this.offset += byteLength
    return this.decoder.decode(bytes)
  }
}

function decodeOptionalBundleMetric(rawValue) {
  if (rawValue !== PACKED_OPTIONAL_NULL && rawValue > 1000) {
    throw new Error("Unsupported match bundle metric")
  }
  return decodeOptionalThousandths(rawValue, PACKED_OPTIONAL_NULL)
}

function decodeMoveCode(code, boardSize) {
  const raw = Number(code)
  if (raw === 0) {
    return ""
  }
  if (raw === MOVE_PASS) {
    return "pass"
  }
  if (raw === MOVE_SWAP) {
    return "swap"
  }
  const cellId = raw - 1
  const size = Number(boardSize)
  if (!Number.isInteger(size) || size <= 0 || cellId < 0 || cellId >= size * size) {
    throw new Error("Unsupported match bundle move")
  }
  return cellIdToMove(cellId, size)
}

function readRequiredFlag(reader) {
  const flag = reader.readUint8()
  if (flag !== 0 && flag !== 1) {
    throw new Error("Unsupported match bundle flag")
  }
  return flag === 1
}

function decodeAnalysis(reader, boardSize) {
  const bestMove = decodeMoveCode(reader.readUint16(), boardSize)
  const bestRedWinrate = decodeOptionalBundleMetric(reader.readUint16())
  const candidateCount = reader.readVaruint()
  const moves = []
  for (let idx = 0; idx < candidateCount; idx += 1) {
    const move = decodeMoveCode(reader.readUint16(), boardSize)
    if (!move) {
      throw new Error("Match bundle candidate is missing a move")
    }
    const redWinrate = decodeOptionalBundleMetric(reader.readUint16())
    const prior = decodeOptionalBundleMetric(reader.readUint16())
    const visitsPayload = reader.readVaruint()
    const row = { move }
    if (typeof redWinrate === "number") {
      row.red_winrate = redWinrate
    }
    if (visitsPayload > 0) {
      row.visits = visitsPayload - 1
    }
    if (typeof prior === "number") {
      row.prior = prior
    }
    moves.push(row)
  }
  const analysis = { moves }
  if (bestMove && typeof bestRedWinrate === "number") {
    analysis.best = {
      move: bestMove,
      red_winrate: bestRedWinrate,
    }
  }
  return analysis
}

function decodeGameRecord(reader, gameIndex) {
  const boardSize = reader.readUint16()
  const kindCode = reader.readUint8()
  const resultCode = reader.readUint8()
  const kind = {
    1: "batch",
    2: "match",
  }[kindCode]
  const result = {
    0: "",
    1: "red_resigned",
    2: "blue_resigned",
  }[resultCode]
  if (!kind || result === undefined) {
    throw new Error("Unsupported match bundle game header")
  }
  const record = {
    kind,
    board_size: boardSize,
    red: reader.readString(),
    blue: reader.readString(),
    opening: reader.readString(),
    result,
    url: reader.readString(),
    plies: [],
    game_index: gameIndex,
  }
  const plyCount = reader.readVaruint()
  for (let idx = 0; idx < plyCount; idx += 1) {
    const ply = idx + 1
    const played = decodeMoveCode(reader.readUint16(), boardSize)
    if (!played) {
      throw new Error("Match bundle ply is missing a played move")
    }
    const row = {
      ply,
      side: playedSideForPly(ply),
      played,
    }
    if (readRequiredFlag(reader)) {
      row.analyze = decodeAnalysis(reader, boardSize)
    }
    record.plies.push(row)
  }
  if (readRequiredFlag(reader)) {
    record.final = {
      side: record.plies.length % 2 === 0 ? "red" : "blue",
      analyze: decodeAnalysis(reader, boardSize),
    }
  }
  return record
}

function decodeMatchBundle(rawBuffer) {
  if (!(rawBuffer instanceof ArrayBuffer) || rawBuffer.byteLength < BUNDLE_HEADER_SIZE) {
    throw new Error("Unsupported match data format")
  }
  const view = new DataView(rawBuffer)
  if (readAsciiMagic(view) !== BUNDLE_MAGIC) {
    throw new Error("Unsupported match data format")
  }
  const version = view.getUint16(4, true)
  if (version !== BUNDLE_VERSION) {
    throw new Error("Unsupported match data format")
  }
  const reader = new MatchBundleReader(rawBuffer)
  reader.offset = BUNDLE_HEADER_SIZE
  const gameCount = view.getUint32(8, true)
  const games = []
  for (let idx = 0; idx < gameCount; idx += 1) {
    games.push(decodeGameRecord(reader, idx + 1))
  }
  if (reader.offset !== rawBuffer.byteLength) {
    throw new Error("Match bundle size mismatch")
  }
  return games
}

function normalizeLoadedData(rawBuffer) {
  const records = decodeMatchBundle(rawBuffer)
  const games = []
  const gamesByIndex = new Map()
  let nextGameIndex = 1
  for (const record of records) {
    const recordGameIndex = Number(record.game_index)
    const gameIndex = Number.isInteger(recordGameIndex) && recordGameIndex > 0 ? recordGameIndex : nextGameIndex
    const game = buildGameFromRecord(record, gameIndex)
    games.push(game)
    gamesByIndex.set(game.game_index, game)
    nextGameIndex = Number(game.game_index) + 1
  }
  games.sort((a, b) => Number(a.game_index) - Number(b.game_index))
  return { games, gamesByIndex }
}

function gameIndexPosition(game) {
  const idx = state.games.findIndex((row) => row.game_index === game.game_index)
  return idx >= 0 ? idx + 1 : 0
}

function gameResultText(game) {
  const resignedSide = resignedSideForResult(game)
  if (resignedSide === "red") {
    return "0-1"
  }
  if (resignedSide === "blue") {
    return "1-0"
  }
  return "?"
}

function resignedSideForResult(game) {
  const result = String(game?.result || "").trim().toLowerCase()
  if (result === "red_resigned") {
    return "red"
  }
  if (result === "blue_resigned") {
    return "blue"
  }
  return null
}

function hexworldResultToken(game) {
  const result = String(game?.result || "").trim().toLowerCase()
  if (result === "red_resigned") {
    return ":rb"
  }
  if (result === "blue_resigned") {
    return ":rw"
  }
  return ""
}

function compactStepStreams(game, step = currentStep(game)) {
  const line = String(step?.line || "")
  const terminalToken = step?.kind === "terminal" ? hexworldResultToken(game) : ""
  const past = `${compactMoveStreamFromLine(line)}${terminalToken}`
  const currentMoves = parseMoves(line)
  const futureMoves = parseMoves(game.finalLine).slice(currentMoves.length)
  const future = `${compactMoveStreamFromLine(formatLine(futureMoves))}${step?.kind === "terminal" ? "" : hexworldResultToken(game)}`
  return { past, future }
}

function compactStepLineSpec(game, step = currentStep(game), { includeFuture = false } = {}) {
  const { past, future } = compactStepStreams(game, step)
  if (includeFuture && future) {
    return past ? `${game.board_size},${past},${future}` : `${game.board_size},,${future}`
  }
  return `${game.board_size}${past ? `,${past}` : ""}`
}

function renderMatchMoveList(game) {
  const step = currentStep(game)
  const parts = game.steps.slice(1).map((row) => ({
    text: row.text,
    isFuture: row.index > state.currentStepIndex,
    line: row.index,
  }))
  renderMoveList({
    container: elements.moveList,
    parts,
    currentMoveCount: step?.moveListCount || 0,
    activateLine: (stepIndex) => {
      goToStepIndex(stepIndex)
    },
  })
}

function renderMatchNav(game) {
  const nav = elements.matchNav
  const firstBtn = elements.moveFirstBtn
  const prevBtn = elements.movePrevBtn
  const nextBtn = elements.moveNextBtn
  const lastBtn = elements.moveLastBtn
  const gamePrevBtn = elements.gamePrevBtn
  const gameNextBtn = elements.gameNextBtn
  if (!(nav instanceof HTMLElement)
    || !(firstBtn instanceof HTMLButtonElement)
    || !(prevBtn instanceof HTMLButtonElement)
    || !(nextBtn instanceof HTMLButtonElement)
    || !(lastBtn instanceof HTMLButtonElement)
    || !(gamePrevBtn instanceof HTMLButtonElement)
    || !(gameNextBtn instanceof HTMLButtonElement)) {
    return
  }
  if (!game) {
    nav.hidden = true
    setNavButtonDisabled(firstBtn, true)
    setNavButtonDisabled(prevBtn, true)
    setNavButtonDisabled(nextBtn, true)
    setNavButtonDisabled(lastBtn, true)
    setNavButtonDisabled(gamePrevBtn, true)
    setNavButtonDisabled(gameNextBtn, true)
    return
  }
  const lastStepIndex = Math.max(0, game.steps.length - 1)
  const currentIndex = Math.max(0, Math.min(Number(state.currentStepIndex) || 0, lastStepIndex))
  const gamePosition = gameIndexPosition(game)
  nav.hidden = false
  setNavButtonDisabled(firstBtn, currentIndex <= 0)
  setNavButtonDisabled(prevBtn, currentIndex <= 0)
  setNavButtonDisabled(nextBtn, currentIndex >= lastStepIndex)
  setNavButtonDisabled(lastBtn, currentIndex >= lastStepIndex)
  setNavButtonDisabled(gamePrevBtn, gamePosition <= 1)
  setNavButtonDisabled(gameNextBtn, gamePosition >= state.games.length)
}

function renderPlayerNames(game) {
  const container = elements.playerNames
  if (!(container instanceof HTMLElement)) {
    return
  }
  container.replaceChildren()
  const red = String(game?.red || "").trim()
  const blue = String(game?.blue || "").trim()
  if (!red && !blue) {
    container.hidden = true
    return
  }
  container.hidden = false
  const players = [
    { label: "Red", labelClass: "match-player-label-red", name: red },
    { label: "Blue", labelClass: "match-player-label-blue", name: blue },
  ]
  players.forEach((player, index) => {
    if (index > 0) {
      const sep = document.createElement("span")
      sep.className = "match-player-sep"
      sep.textContent = " · "
      container.appendChild(sep)
    }
    const label = document.createElement("span")
    label.className = `match-player-label ${player.labelClass}`
    label.textContent = `${player.label} `
    container.appendChild(label)
    const name = document.createElement("span")
    name.className = "match-player-name"
    name.textContent = player.name || "?"
    container.appendChild(name)
  })
}

function renderExternalLinks(game) {
  const container = elements.externalLinks
  if (!(container instanceof HTMLElement)) {
    return
  }
  container.replaceChildren()
  renderHexWorldLink(container, currentHexWorldUrl(game))
  const gameUrl = String(game?.url || "").trim()
  if (gameUrl) {
    const separator = document.createElement("span")
    separator.textContent = " · "
    container.appendChild(separator)
    const link = document.createElement("a")
    link.href = gameUrl
    link.target = "_blank"
    link.rel = "noopener noreferrer"
    link.textContent = "Original game"
    container.appendChild(link)
  }
}

function renderGameSelect() {
  const current = currentGame()
  const select = elements.gameSelect
  if (!(select instanceof HTMLSelectElement)) {
    return
  }
  const currentValue = current ? String(current.game_index) : ""
  if (select.options.length !== state.games.length) {
    select.replaceChildren()
    for (const game of state.games) {
      const option = document.createElement("option")
      option.value = String(game.game_index)
      option.textContent = gameSelectText(game)
      select.appendChild(option)
    }
  }
  select.value = currentValue
}

function gameSelectText(game) {
  const opening = String(game.opening || "").trim()
  const red = String(game.red || "").trim()
  const blue = String(game.blue || "").trim()
  const result = gameResultText(game)
  if (red && blue && opening && result !== "?") {
    return `${game.game_index} · ${red} vs ${blue} · ${opening} · ${result}`
  }
  if (red && blue && opening) {
    return `${game.game_index} · ${red} vs ${blue} · ${opening}`
  }
  if (red && blue && result !== "?") {
    return `${game.game_index} · ${red} vs ${blue} · ${result}`
  }
  if (red && blue) {
    return `${game.game_index} · ${red} vs ${blue}`
  }
  return `Game ${game.game_index}`
}

function hideEvalGraphTooltip() {
  if (!(elements.evalGraphTooltip instanceof HTMLElement)) {
    return
  }
  elements.evalGraphTooltip.hidden = true
}

function showEvalGraphTooltip(event, text) {
  if (!(elements.evalGraphTooltip instanceof HTMLElement) || !(elements.evalGraphWrap instanceof HTMLElement)) {
    return
  }
  const wrapRect = elements.evalGraphWrap.getBoundingClientRect()
  const x = event.clientX - wrapRect.left
  const y = event.clientY - wrapRect.top
  elements.evalGraphTooltip.textContent = text
  elements.evalGraphTooltip.hidden = false
  const tooltip = elements.evalGraphTooltip
  const gap = 10
  const maxLeft = wrapRect.width - tooltip.offsetWidth - 8
  let left = x + gap
  if (left > maxLeft) {
    left = x - tooltip.offsetWidth - gap
  }
  left = Math.max(8, Math.min(left, maxLeft))
  tooltip.style.left = `${left}px`
  tooltip.style.top = `${Math.max(8, y - 10)}px`
}

function evalGraphPointLabel(point) {
  const moveText = String(point?.move || "").trim().toLowerCase()
  const prefix = moveText ? `${point.ply}. ${moveText}` : `${point.ply}.`
  return `${prefix} · ${percentText(point.winrate)}`
}

function renderEvalGraph(game) {
  if (!(elements.evalGraph instanceof SVGSVGElement)) {
    return
  }
  const svg = elements.evalGraph
  svg.replaceChildren()
  hideEvalGraphTooltip()

  const svgRect = svg.getBoundingClientRect()
  const width = Math.max(1, Math.round(svgRect.width || svg.clientWidth || 320))
  const height = Math.max(1, Math.round(svgRect.height || svg.clientHeight || 264))
  const marginLeft = 34
  const marginRight = 12
  const marginTop = 16
  const marginBottom = 28
  const plotWidth = Math.max(1, width - marginLeft - marginRight)
  const plotHeight = Math.max(1, height - marginTop - marginBottom)
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`)

  const points = (game?.graphPoints || []).filter((point) => typeof point.winrate === "number")
  const axisColor = "rgba(110, 89, 70, 0.72)"

  const background = createSvgNode("rect")
  background.setAttribute("x", "0")
  background.setAttribute("y", "0")
  background.setAttribute("width", String(width))
  background.setAttribute("height", String(height))
  background.setAttribute("fill", "transparent")
  background.setAttribute("stroke", "none")
  svg.appendChild(background)

  if (points.length === 0) {
    const emptyText = createSvgNode("text")
    emptyText.setAttribute("x", String(width / 2))
    emptyText.setAttribute("y", String(height / 2))
    emptyText.setAttribute("text-anchor", "middle")
    emptyText.setAttribute("dominant-baseline", "middle")
    emptyText.setAttribute("class", "eval-graph-label")
    emptyText.textContent = "No eval graph data"
    svg.appendChild(emptyText)
    return
  }

  const firstPly = 1
  const lastPly = points[points.length - 1].ply
  const xDenom = Math.max(1, lastPly - firstPly)

  function xForPly(ply) {
    return marginLeft + (((ply - firstPly) * plotWidth) / xDenom)
  }

  function yForWinrate(value) {
    return marginTop + ((1.0 - value) * plotHeight)
  }

  const yTicks = [
    { value: 1.0, label: "100" },
    { value: 0.5, label: "50" },
    { value: 0.0, label: "0" },
  ]
  for (const tick of yTicks) {
    const y = yForWinrate(tick.value)
    const grid = createSvgNode("line")
    grid.setAttribute("x1", String(marginLeft))
    grid.setAttribute("x2", String(width - marginRight))
    grid.setAttribute("y1", String(y))
    grid.setAttribute("y2", String(y))
    grid.setAttribute("stroke", axisColor)
    grid.setAttribute("stroke-width", "1")
    svg.appendChild(grid)

    const label = createSvgNode("text")
    label.setAttribute("x", String(marginLeft - 6))
    label.setAttribute("y", String(y))
    label.setAttribute("text-anchor", "end")
    label.setAttribute("dominant-baseline", "middle")
    label.setAttribute("class", "eval-graph-axis-label")
    label.textContent = tick.label
    svg.appendChild(label)
  }

  const xAxis = createSvgNode("line")
  xAxis.setAttribute("x1", String(marginLeft))
  xAxis.setAttribute("x2", String(width - marginRight))
  xAxis.setAttribute("y1", String(height - marginBottom))
  xAxis.setAttribute("y2", String(height - marginBottom))
  xAxis.setAttribute("stroke", axisColor)
  xAxis.setAttribute("stroke-width", "1")
  svg.appendChild(xAxis)

  const tickPlies = [...new Set([
    firstPly,
    Math.round((firstPly + lastPly) / 2),
    lastPly,
  ])]
  for (const ply of tickPlies) {
    const x = xForPly(ply)
    const tick = createSvgNode("line")
    tick.setAttribute("x1", String(x))
    tick.setAttribute("x2", String(x))
    tick.setAttribute("y1", String(height - marginBottom))
    tick.setAttribute("y2", String(height - marginBottom + 4))
    tick.setAttribute("stroke", axisColor)
    tick.setAttribute("stroke-width", "1")
    svg.appendChild(tick)

    const label = createSvgNode("text")
    label.setAttribute("x", String(x))
    label.setAttribute("y", String(height - marginBottom + 14))
    label.setAttribute("text-anchor", "middle")
    label.setAttribute("class", "eval-graph-axis-label")
    label.textContent = String(ply)
    svg.appendChild(label)
  }

  const series = [
    { key: "red", color: rgbText(RED_RGB), points: points.filter((point) => point.side === "red") },
    { key: "blue", color: rgbText(BLUE_RGB), points: points.filter((point) => point.side === "blue") },
  ]

  const activeLine = currentLine(game)
  const currentPoint = points.find((point) => point.line === activeLine) || null
  if (currentPoint) {
    const x = xForPly(currentPoint.ply)
    const cursor = createSvgNode("line")
    cursor.setAttribute("x1", String(x))
    cursor.setAttribute("x2", String(x))
    cursor.setAttribute("y1", String(marginTop))
    cursor.setAttribute("y2", String(height - marginBottom))
    cursor.setAttribute("stroke", axisColor)
    cursor.setAttribute("stroke-width", "1")
    cursor.setAttribute("stroke-dasharray", "3 4")
    svg.appendChild(cursor)
  }

  for (const seriesRow of series) {
    if (seriesRow.points.length === 0) {
      continue
    }
    const polyline = createSvgNode("polyline")
    polyline.setAttribute(
      "points",
      seriesRow.points.map((point) => `${xForPly(point.ply)},${yForWinrate(point.winrate)}`).join(" "),
    )
    polyline.setAttribute("fill", "none")
    polyline.setAttribute("stroke", seriesRow.color)
    polyline.setAttribute("stroke-width", "2.5")
    polyline.setAttribute("stroke-linecap", "round")
    polyline.setAttribute("stroke-linejoin", "round")
    svg.appendChild(polyline)
  }

  for (const point of points) {
    const x = xForPly(point.ply)
    const y = yForWinrate(point.winrate)
    const color = point.side === "blue" ? rgbText(BLUE_RGB) : rgbText(RED_RGB)

    const dot = createSvgNode("circle")
    dot.setAttribute("cx", String(x))
    dot.setAttribute("cy", String(y))
    const isActive = point.line === activeLine
    dot.setAttribute("r", isActive ? "4.5" : "3")
    dot.setAttribute("fill", color)
    dot.setAttribute("stroke", isActive ? "rgb(246, 241, 232)" : "none")
    dot.setAttribute("stroke-width", isActive ? "2" : "0")
    svg.appendChild(dot)

    const hit = createSvgNode("circle")
    hit.setAttribute("cx", String(x))
    hit.setAttribute("cy", String(y))
    hit.setAttribute("r", "9")
    hit.setAttribute("fill", "transparent")
    hit.setAttribute("class", "eval-graph-hit")
    hit.addEventListener("mouseenter", (event) => {
      showEvalGraphTooltip(event, evalGraphPointLabel(point))
    })
    hit.addEventListener("mousemove", (event) => {
      showEvalGraphTooltip(event, evalGraphPointLabel(point))
    })
    hit.addEventListener("mouseleave", () => {
      hideEvalGraphTooltip()
    })
    hit.addEventListener("click", () => {
      goToStepIndex(point.stepIndex)
    })
    svg.appendChild(hit)
  }
}

function renderBoard(game, node) {
  let topVisits = 0
  for (const candidate of node?.candidates || []) {
    if (typeof candidate?.visits === "number") {
      topVisits = Math.max(topVisits, candidate.visits)
    }
  }
  const board = renderMoveTreeBoard({
    boardSvg,
    boardSize: game.board_size,
    currentLine: currentLine(game),
    currentNode: node,
    extraViewboxPoints: [swapControlPoint(game.board_size)],
    childLineForCandidate,
    mirrorRootCandidates: false,
    buildOverlay: ({ candidate, displayMove, childLine, lookupChildLine, col, boardRow }) => ({
      ...candidate,
      move: displayMove,
      childLine,
      lookupChildLine,
      col,
      row: boardRow,
      className: [
        "board-hex",
        "candidate",
        "board-hex-face",
        ...(state.overlayTextMode === "prior" ? ["candidate-prior"] : []),
      ].join(" "),
      stroke: "none",
      strokeWidth: "0.85",
    }),
    candidateFill: (overlay) => {
      if (state.overlayTextMode === "prior") {
        return typeof overlay?.prior === "number" ? resultFill(overlay.prior) : rgbText(OFF_WHITE_RGB)
      }
      const wr = moverWinrate(overlay, node?.side)
      if (typeof wr !== "number") {
        return rgbText(OFF_WHITE_RGB)
      }
      return hexataCandidateFill(
        CANDIDATE_LOW,
        CANDIDATE_HIGH,
        typeof overlay?.visits === "number" ? overlay.visits : 0,
        topVisits,
        wr,
      )
    },
    overlayPrimaryText: (overlay) => {
      if (state.overlayTextMode === "prior") {
        return percentText(overlay?.prior)
      }
      return percentText(moverWinrate(overlay, node?.side))
    },
    overlaySecondaryText: (overlay) => (
      state.overlayTextMode === "winrate" ? formatVisits(overlay?.visits) : ""
    ),
    onGoToLine: goToLine,
    onGoPrevious: goPrevious,
  })
  renderPlayedMoveHighlight(node, board)
  renderSideActionCandidateHex(node, board, topVisits)
  return board
}

function renderPlayedMoveHighlight(node, board) {
  const played = String(node?.played || "").trim().toLowerCase()
  const childLine = String(node?.playedChildLine || "").trim()
  if (!played || played === "swap" || !childLine) {
    return
  }
  const point = tryParseCell(played)
  if (!point) {
    return
  }
  const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
  const renderedCell = board?.cellsByKey?.get(`${point.col},${point.row}`) || null
  if (renderedCell?.hex?.polygon instanceof SVGPolygonElement) {
    renderedCell.hex.polygon.style.stroke = hoverColor
    renderedCell.hex.polygon.style.strokeWidth = "1.6"
    renderedCell.hex.polygon.style.setProperty("--hover-outline", hoverColor)
  }
  if (renderedCell?.hoverHex?.polygon instanceof SVGPolygonElement && !renderedCell.onClick) {
    renderedCell.hoverHex.polygon.classList.add("clickable")
    renderedCell.hoverHex.polygon.addEventListener("click", () => {
      goToLine(childLine)
    })
  }
}

function renderSideActionCandidateHex(node, board, topVisits) {
  const sideActions = ["pass", "swap"]
  const played = String(node?.played || "").trim().toLowerCase()
  const playedSideAction = sideActions.find((value) => played === value)
  const candidateSideAction = sideActions.find((value) => (
    (node?.candidates || []).some((candidate) => candidate?.retained && candidate.move === value)
  ))
  const action = playedSideAction || candidateSideAction
  if (!action) {
    return
  }
  const candidate = (node?.candidates || []).find((row) => row?.retained && row.move === action) || null
  const playedAction = played === action
  const childLine = candidate?.childLine || (playedAction ? node?.playedChildLine : null)
  const wr = moverWinrate(candidate, node?.side)
  const fill = state.overlayTextMode === "prior"
    ? (typeof candidate?.prior === "number" ? resultFill(candidate.prior) : rgbText(OFF_WHITE_RGB))
    : (typeof wr === "number"
      ? hexataCandidateFill(
        CANDIDATE_LOW,
        CANDIDATE_HIGH,
        typeof candidate?.visits === "number" ? candidate.visits : 0,
        topVisits,
        wr,
      )
      : rgbText(OFF_WHITE_RGB))
  renderSideActionHex({
    boardSvg,
    point: swapControlPoint(currentBoardSize()),
    toPlay: board.toPlay,
    labelText: action === "pass" ? "Pass" : "Swap",
    title: action,
    onClick: typeof childLine === "string" && childLine
      ? () => {
          goToLine(childLine)
        }
      : null,
    fill,
    primaryText: state.overlayTextMode === "prior" ? percentText(candidate?.prior) : percentText(wr),
    secondaryText: state.overlayTextMode === "winrate" ? formatVisits(candidate?.visits) : "",
    highlight: playedAction,
  })
}

function currentHexWorldUrl(game) {
  const base = `https://hexworld.org/board/#${game.board_size}nc1`
  const { past, future } = compactStepStreams(game)
  return hexWorldUrlWithCursor(base, past, future)
}

function render() {
  const game = currentGame()
  if (!game) {
    setTurnStatus(elements.status, "red")
    if (elements.playerNames instanceof HTMLElement) {
      elements.playerNames.replaceChildren()
      elements.playerNames.hidden = true
    }
    elements.moveList.replaceChildren()
    renderMatchNav(null)
    elements.shortcutHint.hidden = true
    elements.externalLinks.replaceChildren()
    overlayModeControls.sync()
    boardSvg.clear()
    if (elements.evalGraph instanceof SVGSVGElement) {
      elements.evalGraph.replaceChildren()
    }
    hideEvalGraphTooltip()
    return
  }

  const node = currentNode(game)
  const board = renderBoard(game, node)
  setTurnStatus(elements.status, board.toPlay)
  renderPlayerNames(game)
  renderMatchMoveList(game)
  renderMatchNav(game)
  elements.shortcutHint.hidden = false
  renderExternalLinks(game)
  renderEvalGraph(game)
  renderGameSelect()
  overlayModeControls.sync()
}

async function loadData() {
  if (state.games.length > 0) {
    return { games: state.games, gamesByIndex: state.gamesByIndex }
  }
  if (state.loadingPromise) {
    return state.loadingPromise
  }
  render()
  state.loadingPromise = (async () => {
    const manifest = await fetchJson(MANIFEST_URL, { cache: "no-store", label: MANIFEST_URL })
    if (manifest?.version !== 1 || typeof manifest?.bundle !== "string") {
      throw new Error("Unsupported match data manifest")
    }
    const bundleUrl = new URL(manifest.bundle, new URL(MANIFEST_URL, window.location.href)).toString()
    const { games, gamesByIndex } = normalizeLoadedData(
      await fetchArrayBuffer(bundleUrl, { cache: "no-store", label: manifest.bundle }),
    )
    if (games.length === 0) {
      throw new Error("No games found in match data")
    }
    state.games = games
    state.gamesByIndex = gamesByIndex
    state.defaultBoardSize = Number(games[0].board_size || state.defaultBoardSize || 14)
    const parsed = parseHashState()
    setCurrentGame(parsed.gameIndex || games[0].game_index, null, {
      updateHash: parsed.hasHash,
      stepIndex: parsed.stepIndex,
    })
    return { games, gamesByIndex }
  })().catch(() => {
    render()
  }).finally(() => {
    state.loadingPromise = null
  })
  return state.loadingPromise
}

function stepGame(delta) {
  const game = currentGame()
  if (!game) {
    return
  }
  const idx = gameIndexPosition(game) - 1
  const next = state.games[idx + delta]
  if (!next) {
    return
  }
  setCurrentGame(next.game_index)
}

function currentSwapAction() {
  const game = currentGame()
  if (!game) {
    return null
  }
  const node = currentNode(game)
  if (!node) {
    return null
  }
  if (
    String(node.played || "").trim().toLowerCase() === "swap"
    && typeof node.playedChildLine === "string"
    && node.playedChildLine
  ) {
    return { childLine: node.playedChildLine }
  }
  return (node.candidates || []).find((candidate) => (
    candidate?.retained
    && candidate.move === "swap"
    && typeof candidate.childLine === "string"
    && candidate.childLine
  )) || null
}

function handleSwapShortcut(event) {
  if (shouldIgnoreGlobalKeydown(event)) {
    return false
  }
  if (!(event.key === "s" || event.key === "S")) {
    return false
  }
  const moves = parseMoves(currentLine())
  if (moves.length === 2 && moves[1] === "swap") {
    event.preventDefault()
    goPrevious()
    return true
  }
  const swapAction = currentSwapAction()
  if (swapAction) {
    event.preventDefault()
    goToLine(swapAction.childLine)
    return true
  }
  return false
}

function syncFromLocationHash() {
  if (state.games.length === 0) {
    return
  }
  const parsed = parseHashState()
  setCurrentGame(parsed.gameIndex || state.games[0].game_index, null, {
    updateHash: parsed.hasHash,
    stepIndex: parsed.stepIndex,
  })
}

function installEvalGraphResizeObserver() {
  if (!(elements.evalGraphWrap instanceof HTMLElement) || typeof ResizeObserver !== "function") {
    return
  }
  let lastSize = ""
  const observer = new ResizeObserver((entries) => {
    const rect = entries[0]?.contentRect
    if (!rect) {
      return
    }
    const nextSize = `${Math.round(rect.width)}x${Math.round(rect.height)}`
    if (nextSize === lastSize) {
      return
    }
    lastSize = nextSize
    const game = currentGame()
    if (game) {
      renderEvalGraph(game)
    }
  })
  observer.observe(elements.evalGraphWrap)
}

elements.matchNav.addEventListener("contextmenu", (event) => {
  event.preventDefault()
}, { capture: true })
elements.matchNav.addEventListener("selectstart", (event) => {
  event.preventDefault()
}, { capture: true })
elements.moveFirstBtn.addEventListener("click", () => {
  if (navButtonDisabled(elements.moveFirstBtn)) {
    return
  }
  goFirst()
})
elements.moveLastBtn.addEventListener("click", () => {
  if (navButtonDisabled(elements.moveLastBtn)) {
    return
  }
  goLast()
})
elements.gamePrevBtn.addEventListener("click", () => {
  if (navButtonDisabled(elements.gamePrevBtn)) {
    return
  }
  stepGame(-1)
})
elements.gameNextBtn.addEventListener("click", () => {
  if (navButtonDisabled(elements.gameNextBtn)) {
    return
  }
  stepGame(1)
})
elements.gameSelect.addEventListener("change", (event) => {
  const next = Number(event.target.value)
  if (Number.isInteger(next) && next > 0) {
    setCurrentGame(next)
  }
})
elements.gameSelect.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp") {
    event.preventDefault()
    stepGame(-1)
    return
  }
  if (event.key === "ArrowDown") {
    event.preventDefault()
    stepGame(1)
  }
})
elements.viewWinrateBtn.addEventListener("click", () => {
  overlayModeControls.set("winrate")
})
elements.viewPriorBtn.addEventListener("click", () => {
  overlayModeControls.set("prior")
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
    canDelete: () => false,
    deleteFromCursor: () => {},
  })
})

installHoldButton(elements.movePrevBtn, () => stepCursor(-1))
installHoldButton(elements.moveNextBtn, () => stepCursor(1))
installEvalGraphResizeObserver()
void loadData()
