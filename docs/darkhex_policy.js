(() => {
  const inspector = document.getElementById("policy-inspector")
  const status = document.getElementById("policy-status")
  const warning = document.getElementById("policy-warning")
  const boardElement = document.getElementById("policy-board")
  const historyText = document.getElementById("policy-history")
  const outcomes = document.getElementById("policy-outcomes")
  const placedButton = document.getElementById("policy-placed")
  const occupiedButton = document.getElementById("policy-occupied")
  const resetButton = document.getElementById("policy-reset")
  const redButton = document.getElementById("policy-red")
  const blueButton = document.getElementById("policy-blue")
  const board = window.HexMoveTree.createBoardSvg(boardElement)
  const { handleStandardKeydown, installMoveNavigation, lerpRgb, rgbText, shouldIgnoreGlobalKeydown, THEME } =
    window.HexStudyUI
  let root = 0
  let reference = 0
  let profiles = []
  let opponentStrategy = null
  const columns = 4
  const rows = 4
  let history = []
  let future = []
  let worlds = new Set([0])
  let selected = null
  let loaded = false
  let player = "red"
  let loadGeneration = 0
  const cache = new Map()
  const sessions = new Map()
  let connectionCache = []
  board.setupViewBox(4)
  inspector.hidden = false

  function parseStrategy(text) {
    const lines = text.trim().split(/\r?\n/)
    const root = Number(lines[0].split(/\s+/)[4])
    const profiles = lines.slice(3).map((line) => {
      const fields = line.trim().split(/\s+/)
      const actions = []
      for (let i = 0; i < Number(fields[0]); i++) {
        const offset = 1 + 4 * i
        actions.push({
          cell: Number(fields[offset]),
          failure: Number(fields[offset + 1]),
          success: Number(fields[offset + 2]),
          weight: BigInt(fields[offset + 3]),
        })
      }
      return actions
    })
    return { root, profiles }
  }

  function actionsAt(profiles, reference) {
    const orientation = reference & 1
    return profiles[Math.floor(reference / 2)].map((edge) => ({
      ...edge,
      cell: orientation ? columns * rows - 1 - edge.cell : edge.cell,
      failure: (edge.failure & ~1) | ((edge.failure & 1) ^ orientation),
      success: (edge.success & ~1) | ((edge.success & 1) ^ orientation),
    }))
  }

  function strategyFor(color) {
    if (!cache.has(color)) {
      const strategy = fetch(`./data/darkhex/${color}.strategy.gz`)
        .then(async (response) => {
          if (!response.ok) throw new Error("Strategy could not be loaded.")
          const stream = response.body.pipeThrough(new DecompressionStream("gzip"))
          return parseStrategy(await new Response(stream).text())
        })
        .catch((error) => {
          cache.delete(color)
          throw error
        })
      cache.set(color, strategy)
    }
    return cache.get(color)
  }

  function name(cell) { return `${String.fromCharCode(97 + (cell % columns))}${Math.floor(cell / columns) + 1}` }

  function currentActions() {
    const edges = actionsAt(profiles, reference)
    const total = edges.reduce((sum, edge) => sum + edge.weight, 0n)
    return edges
      .map((edge) => ({ ...edge, probability: Number(edge.weight) / Number(total) }))
      .sort((a, b) => b.probability - a.probability || a.cell - b.cell)
  }

  function occursAgainstPublishedOpponent() {
    if (!history.length) return true
    const seen = new Set()
    function opponentTurn(index, own, opponent, ref) {
      const key = `${index}:${own}:${opponent}:${ref}`
      if (seen.has(key)) return false
      seen.add(key)
      for (const action of actionsAt(opponentStrategy.profiles, ref)) {
        const bit = 1 << action.cell
        if (own & bit) {
          if (opponentTurn(index, own, opponent, action.failure)) return true
        } else {
          const next = opponent | bit
          if (!connected(next, player === "blue") && play(index, own, next, action.success)) return true
        }
      }
      return false
    }
    function play(index, own, opponent, ref) {
      const probe = history[index]
      const bit = 1 << probe.cell
      if (Boolean(opponent & bit) === probe.placed) return false
      if (index === history.length - 1) return true
      if (!probe.placed) return play(index + 1, own, opponent, ref)
      return opponentTurn(index + 1, own | bit, opponent, ref)
    }
    return player === "red" ? play(0, 0, 0, opponentStrategy.root) : opponentTurn(0, 0, 0, opponentStrategy.root)
  }

  function percentage(probability) {
    const value = probability * 100
    return value > 0 && value < 0.05 ? "<0.1" : value.toFixed(1)
  }

  function connected(mask, red) {
    const cache = connectionCache[Number(red)]
    if (cache[mask]) return cache[mask] === 2
    const seen = new Set(
      Array.from({ length: columns * rows }, (_, cell) => cell).filter(
        (cell) => mask & (1 << cell) && (red ? cell < columns : cell % columns === 0),
      ),
    )
    const pending = [...seen]
    while (pending.length) {
      const cell = pending.pop()
      if (red ? cell >= columns * (rows - 1) : cell % columns === columns - 1) {
        cache[mask] = 2
        return true
      }
      const col = cell % columns
      const row = Math.floor(cell / columns)
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]]) {
        const x = col + dx
        const y = row + dy
        const neighbor = y * columns + x
        if (x >= 0 && x < columns && y >= 0 && y < rows && mask & (1 << neighbor) && !seen.has(neighbor)) {
          seen.add(neighbor)
          pending.push(neighbor)
        }
      }
    }
    cache[mask] = 1
    return false
  }

  function ownStones() { return history.reduce((mask, probe) => mask | (probe.placed ? 1 << probe.cell : 0), 0) }

  function initialWorlds() {
    return player === "red" ? new Set([0]) : new Set(Array.from({ length: columns * rows }, (_, cell) => 1 << cell))
  }

  function possibleOutcomes(cell) {
    const bit = 1 << cell
    return { empty: [...worlds].some((mask) => !(mask & bit)), occupied: [...worlds].some((mask) => mask & bit) }
  }

  function choose(action) {
    selected = action
    const possible = possibleOutcomes(action.cell)
    if (!possible.empty || !possible.occupied) {
      let explanation = ""
      if (history.length || player === "blue") {
        const cell = name(action.cell)
        const known = history.reduce((mask, probe) => mask | (!probe.placed ? 1 << probe.cell : 0), 0)
        const successes = history.filter((probe) => probe.placed).length
        const placements = successes + Number(player === "blue")
        if (possible.empty && connected(known | (1 << action.cell), player === "blue")) {
          const opponent = player === "red" ? "Blue" : "Red"
          explanation = `${cell} must be empty: a ${opponent} stone there would already connect ${opponent}'s sides.`
        } else if (possible.empty && history.length - successes === placements) {
          explanation = `${cell} must be empty: all opposing stones placed so far have already been found.`
        } else {
          explanation = `${cell} must be ${possible.empty ? "empty" : "occupied"} given this history and an ongoing game.`
        }
      }
      record(possible.empty, explanation)
    } else render()
  }

  function undoLast() {
    if (!history.length) return false
    const previous = history.pop()
    future.unshift(previous)
    reference = previous.previous
    worlds = previous.previousWorlds
    selected = null
    render()
    return true
  }

  function goFirst() {
    while (history.length) undoLast()
    if (selected) {
      selected = null
      render()
    }
  }

  function goNext() {
    if (!future.length) return false
    const probe = future[0]
    selected = currentActions().find((action) => action.cell === probe.cell)
    record(probe.placed, probe.explanation)
    return true
  }

  function goLast() {
    while (future.length) goNext()
  }

  function navigateTo(position) {
    while (history.length > position) undoLast()
    while (history.length < position) goNext()
    if (selected) {
      selected = null
      render()
    }
  }

  function render() {
    if (!loaded) return
    const won = connected(ownStones(), player === "red")
    const opponentWon = !won && worlds.size === 0
    const actions = won || opponentWon ? [] : currentActions()
    const byCell = new Map(actions.map((action) => [action.cell, action]))
    const probed = new Map()
    let turn = player === "red" ? 1 : 2
    for (const probe of history) {
      probed.set(probe.cell, { ...probe, turn })
      if (probe.placed) turn += 2
    }
    const lastCell = history.at(-1)?.cell
    const highest = actions[0]?.probability || 1
    board.clear()
    for (let cell = 0; cell < columns * rows; cell++) {
      const probe = probed.get(cell)
      const action = byCell.get(cell)
      const isRed = probe && (probe.placed ? player === "red" : player === "blue")
      const stoneColor = isRed ? [194, 47, 47] : THEME.BLUE_RGB
      const fill = probe
        ? rgbText(probe.placed ? stoneColor : lerpRgb(stoneColor, THEME.OFF_WHITE_RGB, 0.5))
        : action
          ? rgbText(lerpRgb(THEME.OFF_WHITE_RGB, THEME.CANDIDATE_HIGH, (0.65 * action.probability) / highest))
          : rgbText(THEME.OFF_WHITE_RGB)
      const isLast = probe && cell === lastCell
      const hex = board.appendHex((cell % columns) + 1, Math.floor(cell / columns) + 1, {
        fill,
        stroke: probe?.placed ? "none" : THEME.GRID_EDGE,
        className: action ? "board-hex policy-move" : isLast ? "board-hex policy-last-probe" : "board-hex",
        title: name(cell),
        onClick: action ? () => choose(action) : isLast ? undoLast : null,
      })
      if (selected?.cell === cell) {
        hex.polygon.style.stroke = "#1d1c1a"
        hex.polygon.style.strokeWidth = "2"
      }
      if (probe)
        board.appendText(
          hex.cx,
          hex.cy,
          `${probe.turn}${probe.placed ? "" : "×"}`,
          "cell-text",
          probe.placed ? "white" : "#111",
        )
      else if (action) board.appendText(hex.cx, hex.cy, percentage(action.probability))
    }
    board.renderFrame(columns, rows)

    outcomes.hidden = !selected
    const possible = selected ? possibleOutcomes(selected.cell) : null
    const canPlace = possible?.empty
    const canCollide = possible?.occupied
    placedButton.disabled = !canPlace
    occupiedButton.disabled = !canCollide
    resetButton.disabled = !selected && history.length === 0 && future.length === 0
    navigation.sync({ atStart: history.length === 0, atEnd: future.length === 0 })
    historyText.replaceChildren()
    for (const [index, probe] of [...history, ...future].entries()) {
      const button = document.createElement("button")
      button.type = "button"
      const red = probe.placed ? player === "red" : player === "blue"
      button.className = red ? "policy-history-red" : "policy-history-blue"
      if (index >= history.length) button.classList.add("policy-history-future")
      button.textContent = `${name(probe.cell)}${probe.placed ? "" : "×"}`
      button.setAttribute(
        "aria-label",
        `Probe ${index + 1}: ${name(probe.cell)} ${probe.placed ? "placed" : "occupied"}. Go to this point.`,
      )
      if (index === history.length - 1) button.setAttribute("aria-current", "step")
      button.addEventListener("click", () => {
        navigateTo(index + 1)
        historyText.querySelectorAll("button")[index].focus()
      })
      historyText.append(button)
    }
    status.textContent = won
      ? `${player === "red" ? "Red" : "Blue"} has connected their sides.`
      : opponentWon
        ? `${player === "red" ? "Blue" : "Red"} must have won before the next turn.`
        : !actions.length
          ? "No continuation is recorded for this history."
          : selected
            ? `What did you find at ${name(selected.cell)}?`
            : "Choose a probe."
    const offStrategy = !occursAgainstPublishedOpponent()
    warning.hidden = !offStrategy
    warning.textContent = offStrategy
      ? `This history cannot occur against the published ${player === "red" ? "Blue" : "Red"} strategy. The displayed strategy may choose weaker moves here.`
      : ""
    const explanation = !selected && history.at(-1)?.explanation
    if (explanation) status.textContent = `${explanation} ${status.textContent}`
  }

  function record(placed, explanation = "") {
    const child = placed ? selected.success : selected.failure
    if (future[0]?.cell === selected.cell && future[0].placed === placed) future.shift()
    else future = []
    const bit = 1 << selected.cell
    const compatible = [...worlds].filter((mask) => Boolean(mask & bit) !== placed)
    const own = ownStones() | (placed ? bit : 0)
    const nextWorlds = new Set()
    if (!placed || connected(own, player === "red")) {
      for (const mask of compatible) nextWorlds.add(mask)
    } else {
      for (const mask of compatible) {
        let free = ((1 << (columns * rows)) - 1) & ~(own | mask)
        while (free) {
          const move = free & -free
          free ^= move
          const next = mask | move
          if (!connected(next, player === "blue")) nextWorlds.add(next)
        }
      }
    }
    history.push({ cell: selected.cell, placed, previous: reference, previousWorlds: worlds, explanation })
    worlds = nextWorlds
    reference = child
    selected = null
    render()
  }

  placedButton.addEventListener("click", () => record(true))
  occupiedButton.addEventListener("click", () => record(false))
  function reset() {
    reference = root
    history = []
    future = []
    worlds = initialWorlds()
    selected = null
    render()
  }

  resetButton.addEventListener("click", reset)

  async function loadStrategy(nextPlayer) {
    if (loaded) {
      sessions.set(player, { reference, history: history.slice(), future: future.slice(), worlds, selected })
    }
    player = nextPlayer
    connectionCache = [new Uint8Array(1 << (columns * rows)), new Uint8Array(1 << (columns * rows))]
    const generation = ++loadGeneration
    loaded = false
    history = []
    future = []
    worlds = initialWorlds()
    selected = null
    redButton.setAttribute("aria-pressed", String(player === "red"))
    blueButton.setAttribute("aria-pressed", String(player === "blue"))
    placedButton.disabled = occupiedButton.disabled = resetButton.disabled = true
    navigation.sync({ atStart: true, atEnd: true })
    outcomes.hidden = true
    board.clear()
    historyText.textContent = ""
    status.textContent = "Loading strategy…"
    warning.hidden = true
    try {
      const key = player
      const [strategy, opponent] = await Promise.all([
        strategyFor(nextPlayer),
        strategyFor(nextPlayer === "red" ? "blue" : "red"),
      ])
      if (generation !== loadGeneration) return
      root = reference = strategy.root
      profiles = strategy.profiles
      opponentStrategy = opponent
      const saved = sessions.get(key)
      if (saved) {
        reference = saved.reference
        history = saved.history.slice()
        future = saved.future.slice()
        worlds = saved.worlds
        selected = saved.selected
      }
      loaded = true
      render()
    } catch {
      if (generation === loadGeneration) status.textContent = "Could not load the strategy."
    }
  }

  const navigation = installMoveNavigation({
    container: document.getElementById("policy-nav"),
    goFirst,
    goPrevious: undoLast,
    goNext,
    goLast,
  })
  document.addEventListener("keydown", (event) => {
    if (!loaded || shouldIgnoreGlobalKeydown(event)) return
    const key = event.key.toLowerCase()
    if (event.shiftKey && key === "n") {
      handleStandardKeydown(event, { newGame: reset })
    } else if (!event.shiftKey && (key === "e" || key === "o")) {
      const button = key === "e" ? placedButton : occupiedButton
      if (!selected || button.disabled) return
      event.preventDefault()
      if (!event.repeat) record(key === "e")
    } else if (!event.shiftKey && key === "backspace") {
      event.preventDefault()
      undoLast()
    } else if (!event.shiftKey && ["f", "l", "p", "n", "arrowleft", "arrowright"].includes(key)) {
      handleStandardKeydown(event, { goFirst, goPrevious: undoLast, goNext, goLast })
    }
  })
  redButton.addEventListener("click", () => {
    if (player !== "red" || !loaded) void loadStrategy("red")
  })
  blueButton.addEventListener("click", () => {
    if (player !== "blue" || !loaded) void loadStrategy("blue")
  })
  void loadStrategy("red")
})()
