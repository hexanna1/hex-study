(async () => {
  const HEX_SIZE = 24

  function decode(data) {
    let cursor = 0

    function varint() {
      let value = 0
      let shift = 0
      let byte
      do {
        byte = data[cursor++]
        value |= (byte & 127) << shift
        shift += 7
      } while (byte & 128)
      return value >>> 0
    }

    const levels = varint()
    const root = varint()
    const nodes = new Uint32Array(varint() * 4)
    nodes[0] = levels
    let bit = 0
    let next = 1

    function peek(count) {
      const p = bit >>> 3
      const offset = bit & 7
      const word = (data[p] << 24) | (data[p + 1] << 16) | (data[p + 2] << 8) | data[p + 3]
      return (word >>> (32 - offset - count)) & ((1 << count) - 1)
    }

    function read(count) {
      const result = peek(count)
      bit += count
      return result
    }

    function table() {
      const sizes = data.subarray(cursor, cursor + 88)
      cursor += 88
      const counts = new Uint32Array(32)
      const first = new Uint32Array(32)
      const base = new Uint32Array(32)
      for (const size of sizes) {
        if (size) {
          counts[size]++
        }
      }
      let code = 0
      let position = 0
      for (let size = 1; size < 32; size++) {
        code = (code + counts[size - 1]) << 1
        first[size] = code
        base[size] = position
        position += counts[size]
      }
      const symbols = new Uint8Array(position)
      const offsets = base.slice()
      const fast = new Uint32Array(1024)
      for (let symbol = 0; symbol < 88; symbol++) {
        const size = sizes[symbol]
        if (!size) {
          continue
        }
        const index = offsets[size]++
        symbols[index] = symbol
        if (size <= 10) {
          const start = (first[size] + index - base[size]) << (10 - size)
          fast.fill((symbol << 5) | size, start, start + (1 << (10 - size)))
        }
      }
      return { counts, first, base, symbols, fast }
    }

    function integer(book) {
      const entry = book.fast[peek(10)]
      let symbol
      if (entry) {
        bit += entry & 31
        symbol = entry >>> 5
      } else {
        let code = 0
        let size = 0
        do {
          code = (code << 1) | read(1)
          size++
        } while (code - book.first[size] >= book.counts[size])
        symbol = book.symbols[book.base[size] + code - book.first[size]]
      }
      if (symbol < 64) {
        return symbol
      }
      const remainder = symbol - 64 + 6
      return (1 << remainder) | read(remainder)
    }

    for (let level = levels - 1; level >= 0; level--) {
      const count = varint()
      const books = Array.from({ length: 5 }, table)
      bit = cursor * 8
      let previous = 0
      for (let i = 0; i < count; i++) {
        const p = 4 * (next + i)
        nodes[p] = level
        previous += integer(books[0])
        nodes[p + 3] = previous
      }
      previous = 0
      for (let i = 0; i < count; i++) {
        const p = 4 * (next + i)
        const same = nodes[p + 3] === (i ? nodes[p - 1] : 0)
        const x = integer(books[same ? 1 : 2])
        previous += same ? x : (x >>> 1) ^ -(x & 1)
        nodes[p + 1] = previous * 2
      }
      previous = 0
      for (let i = 0; i < count; i++) {
        const p = 4 * (next + i)
        const same =
          nodes[p + 3] === (i ? nodes[p - 1] : 0) && nodes[p + 1] === (i ? nodes[p - 3] : 0)
        const x = integer(books[same ? 3 : 4])
        previous += same ? x : (x >>> 1) ^ -(x & 1)
        nodes[p + 2] = previous
      }
      next += count
      cursor = (bit + 7) >>> 3
    }
    return { size: Math.sqrt(levels), root, nodes }
  }

  function orderFor(n) {
    const order = []
    for (let d = 0; d < 2 * n - 1; d++) {
      for (let r = 0; r < n; r++) {
        if (d - r >= 0 && d - r < n) {
          order.push(r * n + d - r)
        }
      }
    }
    return order
  }

  function makeOracle({ size: n, root, nodes }) {
    const order = orderFor(n)
    const neighbors = []
    let north = 0
    let south = 0
    let west = 0
    let east = 0
    for (let i = 0; i < n * n; i++) {
      if (i < n) {
        north |= 1 << i
      }
      if (i >= n * n - n) {
        south |= 1 << i
      }
      if (i % n === 0) {
        west |= 1 << i
      }
      if (i % n === n - 1) {
        east |= 1 << i
      }
      let mask = 0
      for (const [dr, dc] of [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [-1, 1],
        [1, -1],
      ]) {
        const r = Math.floor(i / n) + dr
        const c = (i % n) + dc
        if (r >= 0 && r < n && c >= 0 && c < n) {
          mask |= 1 << (r * n + c)
        }
      }
      neighbors.push(mask)
    }

    function connected(stones, color) {
      let work = stones & (color === 1 ? north : west)
      let seen = work
      let goal = color === 1 ? south : east
      while (work) {
        const bit = work & -work
        work ^= bit
        if (bit & goal) {
          return true
        }
        const add = neighbors[31 - Math.clz32(bit)] & stones & ~seen
        seen |= add
        work |= add
      }
      return false
    }

    function point(board, swap) {
      let id = root
      while (id >= 2) {
        const p = 4 * (id >>> 1)
        const cell = order[nodes[p]]
        const index = swap ? (cell % n) * n + Math.floor(cell / n) : cell
        let digit = board[index]
        if (swap && digit) {
          digit = 3 - digit
        }
        id = nodes[p + 1 + digit] ^ (id & 1)
      }
      return id
    }

    return (board) => {
      let black = 0
      let white = 0
      let answer = 0
      for (let i = 0; i < n * n; i++) {
        if (board[i] === 1) {
          black |= 1 << i
        }
        if (board[i] === 2) {
          white |= 1 << i
        }
      }
      for (let color = 1; color <= 2; color++) {
        for (let i = 0; i < n * n; i++) {
          if (!board[i]) {
            if (connected((color === 1 ? black : white) | (1 << i), color)) {
              continue
            }
            board[i] = color
            const winning = !point(board, color === 2)
            board[i] = 0
            if (winning) {
              answer |= 1 << (color - 1)
              break
            }
          }
        }
      }
      return answer
    }
  }

  const cellName = (i, size = 5) =>
    `${String.fromCharCode(97 + (i % size))}${1 + Math.floor(i / size)}`
  const cellIndex = (text, size = 5) => size * (Number(text[1]) - 1) + text.charCodeAt(0) - 97

  function connected(board, color) {
    const size = Math.sqrt(board.length)
    const pending = []
    const seen = new Set()
    for (let i = 0; i < board.length; i++) {
      if (board[i] === color && (color === 1 ? i < size : i % size === 0)) {
        pending.push(i)
        seen.add(i)
      }
    }
    while (pending.length) {
      const i = pending.pop()
      const r = Math.floor(i / size)
      const c = i % size
      if (color === 1 ? r === size - 1 : c === size - 1) {
        return true
      }
      for (const [dr, dc] of [
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
      ]) {
        const rr = r + dr
        const cc = c + dc
        const j = size * rr + cc
        if (rr >= 0 && rr < size && cc >= 0 && cc < size && board[j] === color && !seen.has(j)) {
          seen.add(j)
          pending.push(j)
        }
      }
    }
    return false
  }

  function replay(tokens, size = 5) {
    const state = {
      board: new Uint8Array(size * size),
      color: 1,
      batch: 0,
      turn: 1,
      winner: 0,
      stones: [],
    }
    for (const [actionIndex, token] of tokens.entries()) {
      if (state.winner) {
        throw Error("The game has ended. Go back to change the line.")
      }
      if (token === ";") {
        if (!state.batch) {
          throw Error("Place at least one stone before ending a turn.")
        }
        state.color = 3 - state.color
        state.batch = 0
        state.turn++
      } else {
        if (!/^[a-e][1-5]$/.test(token)) {
          throw Error(`Unknown cell: ${token}`)
        }
        const cell = cellIndex(token, size)
        if (token.charCodeAt(0) - 97 >= size || Number(token[1]) > size) {
          throw Error(`Cell outside board: ${token}`)
        }
        if (state.board[cell]) {
          throw Error(`${token} is already occupied.`)
        }
        state.board[cell] = state.color
        state.batch++
        state.stones.push({ actionIndex, cell, color: state.color, turn: state.turn })
        if (connected(state.board, state.color)) {
          state.winner = 3 - state.color
        }
      }
    }
    return state
  }

  function canForceWin(state, oracle) {
    if (state.winner) {
      return state.winner === state.color
    }
    const value = oracle(state.board)
    const own = 1 << (state.color - 1)
    const other = 1 << (2 - state.color)
    return Boolean(value & own || (state.batch && !(value & other)))
  }

  function endingWinner(state, oracle) {
    const other = 1 << (2 - state.color)
    return oracle(state.board) & other ? 3 - state.color : state.color
  }

  function winnerAfterPlacement(state, cell, oracle) {
    const board = state.board.slice()
    board[cell] = state.color
    if (connected(board, state.color)) {
      return 3 - state.color
    }
    return endingWinner({ ...state, board }, oracle)
  }

  function winningTurns(state, oracle, count = 8) {
    if (state.winner || !canForceWin(state, oracle)) {
      return []
    }
    const own = 1 << (state.color - 1)
    const other = 1 << (2 - state.color)
    const board = state.board.slice()
    const empty = Array.from(board.keys()).filter((i) => !board[i])
    const found = []
    if (state.batch && !(oracle(state.board) & other)) {
      found.push([])
    }
    const path = []
    let budget = 2048

    function search(start, left) {
      for (let j = start; j <= empty.length - left && budget > 0 && found.length < count; j++) {
        const cell = empty[j]
        board[cell] = state.color
        path.push(cell)
        budget--
        if (!connected(board, state.color)) {
          const value = oracle(board)
          if (left === 1) {
            if (!(value & other)) {
              found.push(path.slice())
            }
          } else if (value & own) {
            search(j + 1, left - 1)
          }
        }
        path.pop()
        board[cell] = 0
      }
    }

    for (let length = 1; length <= empty.length && budget > 0 && found.length < count; length++) {
      search(0, length)
    }
    if (found.length) {
      return found
    }
    while (path.length < empty.length) {
      let advanced = false
      for (const cell of empty) {
        if (!board[cell]) {
          board[cell] = state.color
          if (!connected(board, state.color)) {
            const value = oracle(board)
            if (!(value & other)) {
              return [[...path, cell]]
            }
            if (value & own) {
              path.push(cell)
              advanced = true
              break
            }
          }
          board[cell] = 0
        }
      }
      if (!advanced) {
        return []
      }
    }
    return []
  }

  const $ = (id) => document.getElementById(id)
  const ui = window.HexStudyUI
  const svg = window.HexMoveTree.createBoardSvg($("board"))
  const {
    BLUE_RGB,
    GRID_EDGE,
    OFF_WHITE_RGB,
    RED_RGB,
  } = ui.THEME
  const colors = ["", "Red", "Blue"]
  const fills = ["", ui.rgbText(RED_RGB), ui.rgbText(BLUE_RGB)]
  let lineActions = []
  let cursor = 0
  let undo = []
  let redo = []
  let oracle = null
  let numbers = false
  let coordsHeld = false
  let rotation = 0
  let size = 5
  const oracles = new Map()
  const loadErrors = new Map()
  let renderedLine = null
  let suggestionKey = ""
  let suggestions = []
  let shortcutHelpOpen = false
  let drag = null
  let boardPointerController = null
  const snapshot = () => structuredClone({ lineActions, cursor, size })

  function cancelBoardDrag() {
    boardPointerController?.cancel({ notify: false })
    drag = null
  }

  function edit(action) {
    cancelBoardDrag()
    $("line-status").textContent = ""
    undo.push(snapshot())
    redo = []
    action()
    render()
  }

  function restore(from, to) {
    if (!from.length) {
      return
    }
    cancelBoardDrag()
    $("line-status").textContent = ""
    to.push(snapshot())
    ;({ lineActions, cursor, size } = from.pop())
    render()
  }

  function go(nextCursor) {
    cancelBoardDrag()
    $("line-status").textContent = ""
    cursor = nextCursor
    render()
  }

  function add(token) {
    try {
      replay([...lineActions.slice(0, cursor), token], size)
    } catch (error) {
      $("line-status").textContent = error.message
      return
    }
    if (lineActions[cursor] === token) {
      go(cursor + 1)
      return
    }
    edit(() => {
      lineActions = [...lineActions.slice(0, cursor), token]
      cursor++
    })
  }

  function formatLine() {
    const cursorPrefix = cursor === 0 && lineActions.length ? "," : ""
    const stream = lineActions
      .map((token, index) => token + (cursor === index + 1 && cursor < lineActions.length ? "," : ""))
      .join("")
    return `${size},${cursorPrefix}${stream}`
  }

  function formatHash(line) {
    const flags = (rotation ? "r9" : "") + (numbers ? "n" : "")
    if (!lineActions.length) {
      return size === 5 && !flags ? "" : `${size}${flags}`
    }
    return line.replace(/^([2-5]),/, `$1${flags},`)
  }

  function decodePositionFragment(fragment) {
    const text = String(fragment ?? "").trim()
    try {
      return decodeURIComponent(text)
    } catch (_error) {
      return text
    }
  }

  function lineInputPositionText(text) {
    const raw = String(text ?? "").trim()
    const hashIndex = raw.indexOf("#")
    if (hashIndex < 0) {
      return raw
    }
    const path = raw.slice(0, hashIndex)
    if (path === "" || /(?:^|\/)rexplus\.html(?:\?.*)?$/i.test(path)) {
      return decodePositionFragment(raw.slice(hashIndex + 1))
    }
    return raw
  }

  function parseLine(text) {
    const raw = text.trim().replace(/^#/, "")
    const header = /^([2-5])(r9)?(n)?,/.exec(raw)
    if (!header) {
      throw Error("Invalid board size")
    }
    const boardSize = Number(header[1])
    const actions = []
    let position = header[0].length
    let markedCursor = null
    function markCursor(index) {
      if (markedCursor !== null) {
        throw Error("Only one cursor marker is allowed")
      }
      markedCursor = index
      position += 1
    }
    if (raw[position] === ",") {
      markCursor(0)
    }
    while (position < raw.length) {
      const match = /^(?:[a-e][1-5]|;)/.exec(raw.slice(position))
      if (!match) {
        throw Error("Expected action")
      }
      actions.push(match[0])
      position += match[0].length
      replay(actions, boardSize)
      if (raw[position] === ",") {
        markCursor(actions.length)
      }
    }
    return {
      actions,
      boardSize,
      cursor: markedCursor ?? actions.length,
      rotation: header[2] ? 1 : 0,
      numbers: Boolean(header[3]),
    }
  }

  function load(text, { message = "Loaded position.", restoreDisplay = false } = {}) {
    try {
      const parsed = parseLine(text)
      cancelBoardDrag()
      renderedLine = null
      size = parsed.boardSize
      lineActions = parsed.actions
      cursor = parsed.cursor
      undo = []
      redo = []
      if (restoreDisplay) {
        rotation = parsed.rotation
        numbers = parsed.numbers
      }
      render()
      $("line-status").textContent = message
      return true
    } catch (error) {
      $("line-status").textContent = error.message
      return false
    }
  }

  function loadLineInput({ message = "Loaded position." } = {}) {
    const text = lineInputPositionText($("current-line").value)
    $("current-line").value = text
    return load(text, { message })
  }

  function reset(boardSize = size) {
    cancelBoardDrag()
    $("line-status").textContent = ""
    renderedLine = null
    size = boardSize
    lineActions = []
    cursor = 0
    undo = []
    redo = []
    render()
  }

  function readHash() {
    const hash = ui.decodeLocationHash()
    if (hash !== null) {
      const line = /^[2-5](?:r9)?n?$/.test(hash) ? `${hash},` : hash
      load(line || "5,", { message: "", restoreDisplay: true })
    }
  }

  function renderMoveList() {
    const container = $("move-list")
    container.replaceChildren()
    let color = 1
    let turn = 1
    let track = null
    let currentMove = null
    for (const [index, token] of lineActions.entries()) {
      if (!track) {
        const row = document.createElement("div")
        row.className = "move-list-row rex-move-list-row"
        const ply = document.createElement("span")
        ply.className = "move-list-ply"
        ply.textContent = `${turn}.`
        track = document.createElement("span")
        track.className = "rex-turn-moves"
        row.append(ply, track)
        container.append(row)
      }
      const move = document.createElement("span")
      move.className = "move-list-move move-list-link"
      move.classList.add(index >= cursor ? "move-list-future" : (color === 1 ? "move-list-red" : "move-list-blue"))
      move.classList.toggle("move-list-current", index + 1 === cursor)
      if (index + 1 === cursor) {
        currentMove = move
      }
      move.textContent = token === ";" ? "↵" : token
      move.title = token === ";" ? "End turn" : token
      move.setAttribute("role", "button")
      move.tabIndex = 0
      if (index + 1 === cursor) {
        move.setAttribute("aria-current", "step")
      }
      const activate = () => {
        go(index + 1)
      }
      move.addEventListener("click", activate)
      move.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          activate()
        }
      })
      track.append(move)
      if (token === ";") {
        color = 3 - color
        turn++
        track = null
      }
    }
    const lastTurn = Math.max(1, turn - (track ? 0 : 1))
    const plyWidth = `${lastTurn}.`.length
    container.style.setProperty("--move-list-ply-width", `${plyWidth}ch`)
    ui.scrollChildIntoView(container, currentMove)
  }

  function render() {
    oracle = oracles.get(size)
    ui.syncPressedButtonGroup(
      [2, 3, 4, 5].map((boardSize) => [boardSize, $("size-" + boardSize)]),
      size,
    )
    $("board").setAttribute("aria-label", `${size} by ${size} Rex+ board`)
    const state = replay(lineActions.slice(0, cursor), size)
    const line = formatLine()
    if (line !== renderedLine) {
      $("current-line").value = line
      renderedLine = line
    }
    ui.replaceHash(formatHash(line))
    ui.setTurnStatus($("rex-status"), state.color === 1 ? "red" : "blue")
    let winner = state.winner
    if (!winner && oracle) {
      winner = state.batch
        ? endingWinner(state, oracle)
        : (canForceWin(state, oracle) ? state.color : 3 - state.color)
    }
    $("result").textContent = winner
      ? `${colors[winner]} wins`
      : (loadErrors.has(size) ? "Unavailable" : "Loading…")
    $("result").classList.toggle("rex-result-red", winner === 1)
    $("result").classList.toggle("rex-result-blue", winner === 2)
    if (loadErrors.has(size)) {
      $("line-status").textContent = loadErrors.get(size)
    }
    ui.setButtonDisabled($("move-end-turn-btn"), !state.batch || Boolean(state.winner))
    $("move-end-turn-btn").title =
      oracle && state.batch && !state.winner
        ? `End turn: ${colors[endingWinner(state, oracle)]} wins`
        : "End turn"
    for (const [id, disabled] of [
      ["move-first-btn", cursor === 0],
      ["move-prev-btn", cursor === 0],
      ["move-next-btn", cursor === lineActions.length],
      ["move-last-btn", cursor === lineActions.length],
      ["move-undo-btn", !undo.length],
      ["move-redo-btn", !redo.length],
      ["move-delete-btn", cursor === 0 && !lineActions.length],
    ]) {
      ui.setNavButtonDisabled($(id), disabled)
    }
    const deleteLabel = cursor < lineActions.length
      ? "Delete tail"
      : (cursor ? "Delete current move" : "Delete move")
    $("move-delete-btn").title = deleteLabel
    $("move-delete-btn").setAttribute("aria-label", deleteLabel)
    svg.clear()
    svg.setBoardOrientation(rotation ? "diamond" : "flat")
    svg.setupViewBox(size)
    $("board").classList.toggle("dragging", Boolean(drag))
    const hoverColor = ui.rgbText(state.color === 1 ? RED_RGB : BLUE_RGB)
    const hoverFill = ui.turnRgbaText(state.color === 1 ? "red" : "blue", 0.12)
    const futureBatch = new Set()
    for (const token of lineActions.slice(cursor)) {
      if (token === ";") {
        break
      }
      futureBatch.add(cellIndex(token, size))
    }
    for (let i = 0; i < size * size; i++) {
      const col = (i % size) + 1
      const row = Math.floor(i / size) + 1
      const stone = state.stones.find((s) => s.cell === i)
      const cellWinner = !stone && oracle && !state.winner
        ? winnerAfterPlacement(state, i, oracle)
        : 0
      const fill = stone
        ? fills[stone.color]
        : cellWinner === 1
          ? "rgb(249, 210, 207)"
          : cellWinner === 2
            ? "rgb(210, 225, 249)"
            : ui.rgbText(OFF_WHITE_RGB)
      const legal = !stone && !state.winner
      const title = cellName(i, size)
      const hitClasses = ["board-hover-hit", "rex-cell-hit"]
      if (legal) {
        hitClasses.push("clickable", "hoverable")
      }
      if (stone) {
        hitClasses.push("board-drag-hit")
      }
      const hoverHex = svg.appendHex(col, row, {
        fill: "transparent",
        stroke: "none",
        className: hitClasses.join(" "),
        title,
        boardPoint: true,
      })
      hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
      const hex = svg.appendHex(col, row, {
        fill,
        stroke: stone ? "none" : GRID_EDGE,
        className: "board-hex board-hex-face",
      })
      hex.polygon.style.setProperty("--hover-outline", hoverColor)
      if (!stone && futureBatch.has(i)) {
        hex.polygon.classList.add("branch-mainline")
        hex.polygon.style.stroke = hoverColor
        hex.polygon.style.strokeWidth = "2.2"
        hex.polygon.style.strokeLinejoin = "round"
      }
      if (drag && stone?.actionIndex === drag.sourceIndex) {
        hex.polygon.classList.add("drag-source")
      }
      if (legal) {
        hoverHex.polygon.setAttribute("tabindex", "0")
        hoverHex.polygon.setAttribute("role", "button")
        hoverHex.polygon.setAttribute(
          "aria-label",
          cellName(i, size) + (cellWinner ? `, ${colors[cellWinner]} wins after this placement` : ""),
        )
        hoverHex.polygon.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault()
            event.stopPropagation()
            add(cellName(i, size))
          }
        })
      }
      const text = coordsHeld ? cellName(i, size) : stone && numbers ? String(stone.turn) : ""
      if (text) {
        const isLast = stone === state.stones.at(-1)
        const textFill = stone
          ? ui.stoneTextColor({ color: stone.color === 1 ? "red" : "blue", isLast })
          : null
        svg.appendText(hex.cx, hex.cy, text, "cell-text", textFill)
      } else if (stone && stone === state.stones.at(-1)) {
        svg.appendCircle(hex.cx, hex.cy, Math.max(2, HEX_SIZE * 0.14), {
          className: "last-move-dot",
          fill: ui.rgbText(OFF_WHITE_RGB),
        })
      }
    }
    if (drag?.targetPoint) {
      svg.appendHex(drag.targetPoint.col, drag.targetPoint.row, {
        fill: fills[drag.sourceColor],
        stroke: "none",
        className: "board-ghost",
      })
    }
    svg.renderFrame(size)
    renderMoveList()
    ui.syncPressedButtonGroup([
      [false, $("move-numbers-off-btn")],
      [true, $("move-numbers-on-btn")],
    ], numbers)
    ui.syncPressedButtonGroup([
      [0, $("orientation-flat-btn")],
      [1, $("orientation-diamond-btn")],
    ], rotation)
    const key =
      size + ":" + Array.from(state.board).join("") + ":" + state.color + ":" + Boolean(state.batch)
    if (oracle && key !== suggestionKey) {
      suggestions = winningTurns(state, oracle)
      suggestionKey = key
    }
    $("suggestions").replaceChildren()
    $("suggestions-panel").hidden = !oracle || state.winner || !suggestions.length
    if (oracle && !state.winner) {
      for (const batch of suggestions) {
        const button = document.createElement("button")
        button.type = "button"
        button.textContent = batch.length
          ? batch.map((i) => cellName(i, size)).join(" ") + " ↵"
          : "↵"
        if (!batch.length) {
          button.setAttribute("aria-label", "End turn")
        }
        button.title = "Play this turn"
        button.addEventListener("click", () => playTurn(batch))
        $("suggestions").append(button)
      }
    }
  }

  function playTurn(batch) {
    const actions = [...batch.map((i) => cellName(i, size)), ";"]
    if (actions.every((token, index) => lineActions[cursor + index] === token)) {
      go(cursor + actions.length)
      return
    }
    edit(() => {
      lineActions = [
        ...lineActions.slice(0, cursor),
        ...actions,
      ]
      cursor = lineActions.length
    })
  }

  function removeAction() {
    if (!cursor) {
      return
    }
    edit(() => {
      lineActions = lineActions.slice(0, cursor - 1)
      cursor--
    })
  }

  function pointsEqual(a, b) {
    return Boolean(a && b && a.col === b.col && a.row === b.row)
  }

  function pointFromHexElement(element) {
    if (!(element instanceof Element)) {
      return null
    }
    const hex = element.closest("[data-board-point='1']")
    if (!(hex instanceof Element)) {
      return null
    }
    const col = Number(hex.getAttribute("data-q"))
    const row = Number(hex.getAttribute("data-r"))
    if (
      !Number.isInteger(col)
      || !Number.isInteger(row)
      || col < 1
      || col > size
      || row < 1
      || row > size
    ) {
      return null
    }
    return { col, row }
  }

  function pointFromClientPosition(clientX, clientY) {
    return pointFromHexElement(document.elementFromPoint(clientX, clientY))
  }

  function currentBoardState() {
    return replay(lineActions.slice(0, cursor), size)
  }

  function stoneAtPoint(state, point) {
    const cell = (point.row - 1) * size + point.col - 1
    return state.stones.find((stone) => stone.cell === cell) || null
  }

  function boardDragData(point) {
    const stone = stoneAtPoint(currentBoardState(), point)
    if (!stone) {
      return null
    }
    return {
      sourceColor: stone.color,
      sourceIndex: stone.actionIndex,
    }
  }

  function takeBackLastStone(state) {
    const stone = state.stones.at(-1)
    if (!stone) {
      return
    }
    if (cursor < lineActions.length) {
      go(stone.actionIndex)
      return
    }
    edit(() => {
      lineActions = lineActions.slice(0, stone.actionIndex)
      cursor = lineActions.length
    })
  }

  function tapBoardPoint(point) {
    const state = currentBoardState()
    const stone = stoneAtPoint(state, point)
    if (stone === state.stones.at(-1)) {
      takeBackLastStone(state)
    } else if (!stone && !state.winner) {
      add(cellName((point.row - 1) * size + point.col - 1, size))
    }
  }

  function dragTargetFromPoint(sourcePoint, point, state) {
    if (!point || pointsEqual(sourcePoint, point) || stoneAtPoint(state, point)) {
      return null
    }
    return { col: point.col, row: point.row }
  }

  function beginBoardDrag(interaction) {
    drag = {
      sourceColor: interaction.dragData.sourceColor,
      sourceIndex: interaction.dragData.sourceIndex,
      startPoint: interaction.startPoint,
      targetPoint: null,
    }
    render()
  }

  function moveBoardDrag(_interaction, point) {
    if (!drag) {
      return
    }
    const targetPoint = dragTargetFromPoint(drag.startPoint, point, currentBoardState())
    if (pointsEqual(drag.targetPoint, targetPoint) || (!drag.targetPoint && !targetPoint)) {
      return
    }
    drag.targetPoint = targetPoint
    render()
  }

  function rewriteDraggedStone(sourceIndex, targetPoint) {
    const nextActions = [...lineActions]
    nextActions[sourceIndex] = cellName((targetPoint.row - 1) * size + targetPoint.col - 1, size)
    let end = sourceIndex + 1
    while (end < nextActions.length) {
      try {
        replay(nextActions.slice(0, end + 1), size)
        end++
      } catch (_error) {
        break
      }
    }
    edit(() => {
      lineActions = nextActions.slice(0, end)
      cursor = Math.min(cursor, lineActions.length)
    })
  }

  function finishBoardDrag(interaction, releasePoint) {
    drag = null
    const state = currentBoardState()
    const targetPoint = dragTargetFromPoint(interaction.startPoint, releasePoint, state)
    if (targetPoint) {
      rewriteDraggedStone(interaction.dragData.sourceIndex, targetPoint)
    } else {
      render()
    }
  }

  function cancelBoardPointerInteraction(interaction) {
    if (interaction.dragging) {
      drag = null
      render()
    }
  }

  function showCoords(on) {
    if (coordsHeld === on) {
      return
    }
    coordsHeld = on
    render()
  }

  function stepCursor(delta) {
    const nextCursor = Math.max(0, Math.min(lineActions.length, cursor + Number(delta)))
    if (nextCursor === cursor) {
      return false
    }
    go(nextCursor)
    return true
  }

  const actions = {
    goPrevious: () => stepCursor(-1),
    goNext: () => stepCursor(1),
    goFirst: () => go(0),
    goLast: () => go(lineActions.length),
    canDelete: () => cursor !== 0 || lineActions.length !== 0,
    deleteFromCursor: () => {
      if (cursor === lineActions.length) {
        removeAction()
        return
      }
      edit(() => {
        lineActions = lineActions.slice(0, cursor)
      })
    },
  }
  for (const [id, action] of Object.entries({
    "move-first-btn": actions.goFirst,
    "move-last-btn": actions.goLast,
    "move-delete-btn": actions.deleteFromCursor,
    "move-undo-btn": () => restore(undo, redo),
    "move-redo-btn": () => restore(redo, undo),
    "move-end-turn-btn": () => add(";"),
    "orientation-flat-btn": () => {
      $("line-status").textContent = ""
      rotation = 0
      render()
    },
    "orientation-diamond-btn": () => {
      $("line-status").textContent = ""
      rotation = 1
      render()
    },
    "move-numbers-off-btn": () => {
      $("line-status").textContent = ""
      numbers = false
      render()
    },
    "move-numbers-on-btn": () => {
      $("line-status").textContent = ""
      numbers = true
      render()
    },
    "reset-btn": () => reset(),
  })) {
    $(id).addEventListener("click", () => {
      if (!ui.navButtonDisabled($(id))) {
        action()
      }
    })
  }
  ui.installHoldButton($("move-prev-btn"), actions.goPrevious)
  ui.installHoldButton($("move-next-btn"), actions.goNext)
  $("move-nav").addEventListener("contextmenu", (event) => {
    event.preventDefault()
  }, { capture: true })
  $("move-nav").addEventListener("selectstart", (event) => {
    event.preventDefault()
  }, { capture: true })
  for (const n of [2, 3, 4, 5]) {
    $("size-" + n).addEventListener("click", () => {
      if (n !== size) {
        reset(n)
      }
    })
  }
  boardPointerController = ui.createBoardPointerController({
    board: $("board"),
    pointFromTarget: pointFromHexElement,
    pointFromClientPosition,
    pointsEqual,
    dragDataForPoint: boardDragData,
    onTap: tapBoardPoint,
    onDragStart: beginBoardDrag,
    onDragMove: moveBoardDrag,
    onDrop: finishBoardDrag,
    onCancel: cancelBoardPointerInteraction,
  })
  window.addEventListener("blur", () => showCoords(false))
  window.addEventListener("keyup", (event) => {
    if (event.key.toLowerCase() === "c") {
      showCoords(false)
    }
  })
  $("current-line").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault()
      if (loadLineInput()) {
        $("current-line").blur()
      }
    } else if (event.key === "Escape") {
      $("current-line").value = formatLine()
      $("line-status").textContent = ""
      $("current-line").blur()
    }
  })
  $("line-load-btn").addEventListener("click", () => {
    loadLineInput()
  })
  window.addEventListener("paste", (event) => {
    const target = event.target
    if (target instanceof HTMLElement) {
      const tag = target.tagName.toLowerCase()
      if (tag === "input" || tag === "textarea" || target.isContentEditable) {
        return
      }
    }
    const text = event.clipboardData?.getData("text/plain") || ""
    if (!String(text).trim()) {
      return
    }
    const pastedLine = lineInputPositionText(text)
    event.preventDefault()
    $("current-line").value = pastedLine
    load(pastedLine, { message: "Loaded pasted position." })
  })
  window.addEventListener("keydown", (event) => {
    if (
      event.defaultPrevented ||
      event.target.closest?.("input,textarea,select,[contenteditable=true]")
    ) {
      return
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
      event.preventDefault()
      restore(event.shiftKey ? redo : undo, event.shiftKey ? undo : redo)
      return
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
      event.preventDefault()
      restore(redo, undo)
      return
    }
    if (ui.shouldIgnoreGlobalKeydown(event)) {
      return
    }
    if (event.key.toLowerCase() === "m") {
      event.preventDefault()
      $("line-status").textContent = ""
      numbers = !numbers
      render()
      return
    }
    if (event.key === "Backspace") {
      event.preventDefault()
      removeAction()
      return
    }
    if (event.key === "Enter" || event.key === ";") {
      if (event.target.closest?.("button,a")) {
        return
      }
      event.preventDefault()
      if (!ui.navButtonDisabled($("move-end-turn-btn"))) {
        add(";")
      }
      return
    }
    if (event.key === "?") {
      event.preventDefault()
      toggleShortcutHelp()
      return
    }
    if (event.shiftKey && event.key.toLowerCase() === "o") {
      $("line-status").textContent = ""
      rotation = 1 - rotation
      render()
      return
    }
    if (!event.shiftKey && event.key.toLowerCase() === "c") {
      event.preventDefault()
      showCoords(true)
      return
    }
    ui.handleStandardKeydown(event, actions)
  })
  function syncShortcutHelpExpanded() {
    $("shortcut-help-link").setAttribute("aria-expanded", shortcutHelpOpen ? "true" : "false")
  }

  function placeShortcutHelpPopover() {
    const popover = $("shortcut-help-popover")
    const link = $("shortcut-help-link")
    if (!shortcutHelpOpen) {
      return
    }
    const margin = 12
    const linkRect = link.getBoundingClientRect()
    const popoverRect = popover.getBoundingClientRect()
    const maxLeft = Math.max(margin, window.innerWidth - popoverRect.width - margin)
    const left = Math.min(maxLeft, Math.max(margin, linkRect.right - popoverRect.width))
    const maxTop = Math.max(margin, window.innerHeight - popoverRect.height - margin)
    const linkIsInView = linkRect.bottom >= margin && linkRect.top <= window.innerHeight - margin
    const below = linkRect.bottom + 8
    const above = linkRect.top - popoverRect.height - 8
    const preferredTop = linkIsInView
      ? (below + popoverRect.height <= window.innerHeight - margin ? below : above)
      : maxTop
    const top = Math.min(maxTop, Math.max(margin, preferredTop))
    popover.style.left = `${left}px`
    popover.style.top = `${top}px`
  }

  function showShortcutHelp() {
    const popover = $("shortcut-help-popover")
    if (shortcutHelpOpen || typeof popover.showPopover !== "function") {
      return false
    }
    popover.showPopover()
    shortcutHelpOpen = true
    placeShortcutHelpPopover()
    syncShortcutHelpExpanded()
    return true
  }

  function hideShortcutHelp() {
    const popover = $("shortcut-help-popover")
    if (!shortcutHelpOpen || typeof popover.hidePopover !== "function") {
      return false
    }
    popover.hidePopover()
    shortcutHelpOpen = false
    syncShortcutHelpExpanded()
    return true
  }

  function toggleShortcutHelp() {
    return shortcutHelpOpen ? hideShortcutHelp() : showShortcutHelp()
  }

  $("shortcut-help-popover").addEventListener("toggle", (event) => {
    if (event.newState === "open") {
      shortcutHelpOpen = true
    } else if (event.newState === "closed") {
      shortcutHelpOpen = false
    }
    syncShortcutHelpExpanded()
    placeShortcutHelpPopover()
  })
  window.addEventListener("resize", placeShortcutHelpPopover)
  window.addEventListener("scroll", placeShortcutHelpPopover, true)
  function svgExportFileName() {
    const orientationFlag = rotation ? "r9" : ""
    const moveNumberFlag = numbers ? "n" : ""
    const sizeText = `${size}${orientationFlag}${moveNumberFlag}`
    const pathText = lineActions
      .slice(0, cursor)
      .map((token) => token === ";" ? "e" : token)
      .join("")
    return `rexplus-${sizeText}-${ui.safeFileStem(pathText, "root")}.svg`
  }

  $("export-svg-link").querySelector("a").addEventListener("click", (event) => {
    event.preventDefault()
    ui.downloadTextFile({
      text: ui.serializeBoardSvg($("board")),
      filename: svgExportFileName(),
      type: "image/svg+xml;charset=utf-8",
    })
  })
  window.addEventListener("hashchange", readHash)
  readHash()
  await Promise.all(
    [2, 3, 4, 5].map(async (boardSize) => {
      try {
        const response = await fetch(`./data/rexplus/${boardSize}x${boardSize}.bin`)
        if (!response.ok) {
          throw Error("Could not load analysis. Reload to retry.")
        }
        oracles.set(boardSize, makeOracle(decode(new Uint8Array(await response.arrayBuffer()))))
        if (size === boardSize) {
          render()
        }
      } catch (error) {
        loadErrors.set(boardSize, error.message)
        if (size === boardSize) {
          render()
        }
      }
    }),
  )
})()
