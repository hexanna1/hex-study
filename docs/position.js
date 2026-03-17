((root) => {
  const IGNORED_HEXWORLD_TOKENS = Object.freeze([":S", ":rw", ":rb", ":fw", ":fb"])
  const ignoredHexWorldTokenSet = new Set(IGNORED_HEXWORLD_TOKENS)

  function tokenizeHexWorldMoveStream(stream) {
    const raw = String(stream ?? "").replace(/\s+/g, "")
    if (!raw) {
      return []
    }
    const tokens = [...raw.matchAll(/:p|:s|:S|:rw|:rb|:fw|:fb|[a-z]+[0-9]+/g)]
      .map((match) => match[0])
    if (tokens.join("") !== raw) {
      throw new Error("Bad HexWorld move stream")
    }
    return tokens
  }

  function parseHexWorldPrefix(prefix) {
    const match = /^([0-9]+)(?:x([0-9]+))?([a-z0-9]*)$/.exec(String(prefix ?? "").trim())
    if (!match) {
      throw new Error("Bad HexWorld prefix")
    }
    const cols = Number(match[1])
    const rows = match[2] === undefined ? cols : Number(match[2])
    if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 1 || rows < 1 || cols > 53 || rows > 53) {
      throw new Error("Bad HexWorld board size")
    }

    const configs = []
    const tail = match[3]
    let index = 0
    while (index < tail.length) {
      if (tail.startsWith("c1", index)) {
        configs.push("c1")
        index += 2
        continue
      }
      if (tail.startsWith("n", index)) {
        configs.push("n")
        index += 1
        continue
      }
      if (tail.startsWith("r", index)) {
        let end = index + 1
        while (end < tail.length && tail[end] >= "0" && tail[end] <= "9") {
          end += 1
        }
        if (end === index + 1) {
          throw new Error("Bad HexWorld rotation")
        }
        const rotation = Number(tail.slice(index + 1, end))
        if (!Number.isInteger(rotation) || rotation < 1 || rotation > 12) {
          throw new Error("Bad HexWorld rotation")
        }
        configs.push(`r${rotation}`)
        index = end
        continue
      }
      throw new Error("Unsupported HexWorld config")
    }
    return { cols, rows, configs: Object.freeze(configs) }
  }

  function normalizeMoveToken(token) {
    const raw = String(token || "").trim().toLowerCase()
    if (raw === ":s" || raw === "swap") {
      return "swap"
    }
    if (raw === ":p" || raw === "pass") {
      return "pass"
    }
    return raw
  }

  function parseMoves(line) {
    const raw = String(line || "").trim().toLowerCase()
    if (!raw) {
      return []
    }
    const moves = []
    const re = /(:s|:p|swap|pass|[a-z]+[1-9][0-9]*)/g
    let idx = 0
    while (idx < raw.length) {
      const match = re.exec(raw)
      if (!match || match.index !== idx) {
        return []
      }
      moves.push(normalizeMoveToken(match[0]))
      idx = re.lastIndex
    }
    return moves
  }

  function formatLine(moves) {
    return moves.map((move) => normalizeMoveToken(move)).join("")
  }

  function alphaLabel(index) {
    let n = Number(index)
    if (!Number.isInteger(n) || n <= 0) {
      return ""
    }
    const letters = []
    while (n > 0) {
      n -= 1
      letters.push(String.fromCharCode(97 + (n % 26)))
      n = Math.floor(n / 26)
    }
    return letters.reverse().join("")
  }

  function formatCell(col, row) {
    return `${alphaLabel(col)}${row}`
  }

  function tryParseCell(move) {
    const match = /^([a-z]+)([0-9]+)$/.exec(String(move || "").trim().toLowerCase())
    if (!match) {
      return null
    }
    const row = Number(match[2])
    if (!Number.isInteger(row) || row < 1) {
      return null
    }
    let col = 0
    for (const ch of match[1]) {
      col = (26 * col) + (ch.charCodeAt(0) - 96)
    }
    return { col, row }
  }

  function parseCell(move) {
    const point = tryParseCell(move)
    if (!point) {
      throw new Error(`Bad cell '${move}'`)
    }
    return point
  }

  function cellIdToMove(cellId, boardSize) {
    const id = Number(cellId)
    const size = Number(boardSize)
    if (!Number.isInteger(id) || !Number.isInteger(size) || size <= 0 || id < 0 || id >= (size * size)) {
      throw new Error(`Bad cell id '${cellId}' for board size ${boardSize}`)
    }
    const row = Math.floor(id / size) + 1
    const col = (id % size) + 1
    return formatCell(col, row)
  }

  function pointKey(col, row) {
    return `${col},${row}`
  }

  function lineParent(line) {
    const moves = parseMoves(line)
    if (moves.length === 0) {
      return ""
    }
    return formatLine(moves.slice(0, -1))
  }

  function linePrefixes(line) {
    const moves = parseMoves(line)
    const prefixes = []
    for (let i = 1; i <= moves.length; i += 1) {
      prefixes.push(formatLine(moves.slice(0, i)))
    }
    return prefixes
  }

  function appendMoveToLine(line, move) {
    const moveText = normalizeMoveToken(move)
    return moveText ? formatLine([...parseMoves(line), moveText]) : String(line || "")
  }

  function compactMoveStreamFromLine(line) {
    return parseMoves(line).map((move) => {
      if (move === "swap") {
        return ":s"
      }
      if (move === "pass") {
        return ":p"
      }
      return move
    }).join("")
  }

  function rotateCell180(move, boardSize) {
    const point = parseCell(move)
    const size = Number(boardSize)
    return formatCell((size + 1) - point.col, (size + 1) - point.row)
  }

  function transformMove(move, boardSize, rotation) {
    const normalized = normalizeMoveToken(move)
    if (normalized === "swap" || normalized === "pass") {
      return normalized
    }
    if (Number(rotation) === 180) {
      return rotateCell180(normalized, boardSize)
    }
    return normalized
  }

  function transformLine(line, boardSize, rotation) {
    return formatLine(parseMoves(line).map((move) => transformMove(move, boardSize, rotation)))
  }

  function lookupLineToDisplayLine(line, { boardSize, displayRotation }) {
    return transformLine(line, boardSize, displayRotation)
  }

  function materializeBoardState(moves, boardSize = null, options = {}) {
    const {
      isLegalCell = null,
      swapStone = null,
    } = options
    const stones = []
    const occupied = new Map()
    let lastStone = null
    for (let i = 0; i < moves.length; i += 1) {
      const move = moves[i]
      if (move === "pass") {
        lastStone = null
        continue
      }
      if (move === "swap") {
        if (i !== 1 || stones.length !== 1) {
          throw new Error("Illegal swap placement")
        }
        const firstStone = stones[0]
        occupied.delete(pointKey(firstStone.col, firstStone.row))
        const swappedStone = typeof swapStone === "function"
          ? swapStone(firstStone)
          : {
              ...firstStone,
              col: firstStone.row,
              row: firstStone.col,
              color: "blue",
              ply: "S",
            }
        if (
          boardSize !== null
          && (
            swappedStone.col < 1
            || swappedStone.col > Number(boardSize)
            || swappedStone.row < 1
            || swappedStone.row > Number(boardSize)
          )
        ) {
          throw new Error("Swap out of bounds")
        }
        occupied.set(pointKey(swappedStone.col, swappedStone.row), swappedStone)
        stones[0] = swappedStone
        lastStone = swappedStone
        continue
      }
      const point = parseCell(move)
      if (typeof isLegalCell === "function" && !isLegalCell(point.col, point.row, boardSize)) {
        throw new Error("Move out of bounds")
      }
      if (
        boardSize !== null
        && (point.col < 1 || point.col > Number(boardSize) || point.row < 1 || point.row > Number(boardSize))
      ) {
        throw new Error("Move out of bounds")
      }
      if (occupied.has(pointKey(point.col, point.row))) {
        throw new Error("Duplicate occupied cell")
      }
      const stone = {
        move,
        col: point.col,
        row: point.row,
        color: i % 2 === 0 ? "red" : "blue",
        ply: i + 1,
        isLast: false,
      }
      occupied.set(pointKey(point.col, point.row), stone)
      stones.push(stone)
      lastStone = stone
    }
    for (const stone of stones) {
      stone.isLast = stone === lastStone
    }
    return {
      moves: [...moves],
      stones,
      occupied,
      toPlay: moves.length % 2 === 0 ? "red" : "blue",
    }
  }

  function normalizeLine(line, boardSize = null) {
    const moves = parseMoves(line)
    try {
      materializeBoardState(moves, boardSize)
    } catch (_error) {
      return ""
    }
    return formatLine(moves)
  }

  function compactCursorText({
    boardSize,
    line = "",
    futureLines = [],
    defaultBoardSize = 11,
    keepEmptyLineComma = false,
  }) {
    const size = Number(boardSize)
    const current = formatLine(parseMoves(line))
    const past = compactMoveStreamFromLine(current)
    const futureMoves = []
    let previousLength = parseMoves(current).length
    for (const futureLine of futureLines || []) {
      const moves = parseMoves(futureLine)
      if (moves.length !== previousLength + 1) {
        break
      }
      futureMoves.push(moves[moves.length - 1])
      previousLength = moves.length
    }
    const future = compactMoveStreamFromLine(formatLine(futureMoves))
    if (future) {
      return `${size},${past},${future}`
    }
    if (past) {
      return `${size},${past}`
    }
    if (keepEmptyLineComma) {
      return `${size},`
    }
    return size !== Number(defaultBoardSize) ? String(size) : ""
  }

  function compactCursorHash(options) {
    const text = compactCursorText(options)
    return text ? `#${text}` : ""
  }

  function parseCompactCursorHash(text, {
    defaultBoardSize = 11,
    isBoardSizeSupported = () => true,
    normalizeLineFn = normalizeLine,
  } = {}) {
    const raw = String(text ?? "").trim()
    if (!raw) {
      return { boardSize: Number(defaultBoardSize), line: "", fullLine: "", cursor: 0, valid: true }
    }
    const parts = raw.split(",")
    if (parts.length > 3) {
      return { boardSize: null, line: "", fullLine: "", cursor: 0, valid: false }
    }
    let prefix
    let pastTokens
    let futureTokens
    try {
      prefix = parseHexWorldPrefix(parts[0])
      pastTokens = tokenizeHexWorldMoveStream(parts[1] || "")
      futureTokens = tokenizeHexWorldMoveStream(parts[2] || "")
    } catch (_error) {
      return { boardSize: null, line: "", fullLine: "", cursor: 0, valid: false }
    }
    const boardSize = prefix.cols
    if (prefix.cols !== prefix.rows || !isBoardSizeSupported(boardSize)) {
      return { boardSize: null, line: "", fullLine: "", cursor: 0, valid: false }
    }
    const canonicalMove = (token) => {
      if (token === ":p" || token === ":s") {
        return normalizeMoveToken(token)
      }
      const point = tryParseCell(token)
      return point ? formatCell(point.col, point.row) : token
    }
    const pastMoves = pastTokens
      .filter((token) => !ignoredHexWorldTokenSet.has(token))
      .map(canonicalMove)
    const futureMoves = futureTokens
      .filter((token) => !ignoredHexWorldTokenSet.has(token))
      .map(canonicalMove)
    const line = formatLine(pastMoves)
    const fullLine = formatLine([...pastMoves, ...futureMoves])
    const cursor = pastMoves.length
    const normalized = normalizeLineFn(line, boardSize)
    const normalizedFull = normalizeLineFn(fullLine, boardSize)
    if (normalized !== line || normalizedFull !== fullLine) {
      return { boardSize: null, line: "", fullLine: "", cursor: 0, valid: false }
    }
    return { boardSize, line: normalized, fullLine: normalizedFull, cursor, valid: true }
  }

  function positionKey(game, boardSize, moves) {
    return JSON.stringify([String(game || ""), Number(boardSize), moves.map(normalizeMoveToken)])
  }

  function createPositionSnapshot({ game, boardSize, moves, board }) {
    const frozenMoves = Object.freeze(moves.map(normalizeMoveToken))
    const stones = Object.freeze((board.stones || []).map((stone) => Object.freeze({
      move: stone.move,
      col: stone.col,
      row: stone.row,
      color: stone.color,
      ply: stone.ply,
      isLast: Boolean(stone.isLast),
    })))
    return Object.freeze({
      game: String(game || ""),
      boardSize: Number(boardSize),
      key: positionKey(game, boardSize, frozenMoves),
      line: formatLine(frozenMoves),
      moves: frozenMoves,
      stones,
      toPlay: board.toPlay,
    })
  }

  root.HexPosition = {
    IGNORED_HEXWORLD_TOKENS,
    alphaLabel,
    appendMoveToLine,
    cellIdToMove,
    compactCursorHash,
    compactCursorText,
    compactMoveStreamFromLine,
    createPositionSnapshot,
    formatCell,
    formatLine,
    lineParent,
    linePrefixes,
    lookupLineToDisplayLine,
    materializeBoardState,
    normalizeLine,
    normalizeMoveToken,
    parseCell,
    parseCompactCursorHash,
    parseHexWorldPrefix,
    parseMoves,
    pointKey,
    positionKey,
    tokenizeHexWorldMoveStream,
    transformLine,
    transformMove,
    tryParseCell,
  }
})(globalThis)
