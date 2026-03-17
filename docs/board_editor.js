(() => {
  const {
    copyTextToClipboard,
    decodeLocationHash,
    handleStandardKeydown,
    installHoldButton,
    navButtonDisabled,
    replaceHash,
    setNavButtonDisabled,
    setTurnStatus,
    shouldIgnoreGlobalKeydown,
  } = window.HexStudyUI
  const {
    formatCell,
    formatLine,
    parseMoves,
    pointKey,
    renderSideActionHex,
    swapControlPoint,
    tryParseCell,
  } = window.HexMoveTree

  const ROOT_ID = 0

  function createMoveTree() {
    const root = {
      id: ROOT_ID,
      move: null,
      parent: null,
      children: [],
    }
    return {
      root,
      cursor: root,
      nextId: ROOT_ID + 1,
    }
  }

  function preferredChild(node) {
    return node?.children?.[0] || null
  }

  function findChild(parent, move) {
    return parent.children.find((child) => child.move === move) || null
  }

  function appendChild(tree, parent, move) {
    const child = {
      id: tree.nextId,
      move,
      parent,
      children: [],
    }
    tree.nextId += 1
    parent.children.push(child)
    return child
  }

  function removeChild(parent, child) {
    const index = parent.children.indexOf(child)
    if (index < 0) {
      return false
    }
    parent.children.splice(index, 1)
    child.parent = null
    return true
  }

  function currentPathNodes(tree) {
    const path = []
    let node = tree.cursor
    while (node && node.parent !== null) {
      path.push(node)
      node = node.parent
    }
    path.reverse()
    return path
  }

  function currentPathMoves(tree) {
    return currentPathNodes(tree).map((node) => node.move)
  }

  function mainlineTailNodes(node) {
    const tail = []
    let cursor = preferredChild(node)
    while (cursor) {
      tail.push(cursor)
      cursor = preferredChild(cursor)
    }
    return tail
  }

  function mainlineTailMoves(tree) {
    return mainlineTailNodes(tree.cursor).map((node) => node.move)
  }

  function mainlineTailNode(tree) {
    let node = tree.root
    while (preferredChild(node)) {
      node = preferredChild(node)
    }
    return node
  }

  function nodeDepth(node) {
    let depth = 0
    let cursor = node
    while (cursor && cursor.parent !== null) {
      depth += 1
      cursor = cursor.parent
    }
    return depth
  }

  function frontierAtPly(tree, targetPly) {
    const frontier = []
    if (targetPly <= 0) {
      return frontier
    }

    const stack = [...tree.root.children].reverse().map((node) => ({
      node,
      ply: 1,
    }))
    while (stack.length > 0) {
      const item = stack.pop()
      if (item.ply === targetPly) {
        frontier.push(item.node)
        continue
      }
      for (const child of [...item.node.children].reverse()) {
        stack.push({
          node: child,
          ply: item.ply + 1,
        })
      }
    }
    return frontier
  }

  function samePlyNeighborCursor(tree, direction) {
    const frontier = frontierAtPly(tree, currentPathNodes(tree).length)
    const cursorIndex = frontier.findIndex((node) => node === tree.cursor)
    if (cursorIndex < 0) {
      return null
    }
    const neighborIndex = cursorIndex + direction
    return neighborIndex >= 0 && neighborIndex < frontier.length ? frontier[neighborIndex] : null
  }

  function siblingCursor(tree, direction) {
    if (direction !== -1 && direction !== 1) {
      return null
    }
    const samePly = samePlyNeighborCursor(tree, direction)
    if (samePly || direction === 1) {
      return samePly
    }

    const path = currentPathNodes(tree)
    for (let depth = path.length - 1; depth >= 0; depth -= 1) {
      const parent = depth === 0 ? tree.root : path[depth - 1]
      const index = parent.children.indexOf(path[depth])
      if (index > 0) {
        let node = parent.children[index - 1]
        while (preferredChild(node)) {
          node = preferredChild(node)
        }
        return node
      }
    }
    return null
  }

  function normalizeMove(move) {
    return formatLine([move])
  }

  function compactMoveToken(move) {
    if (move === "pass") {
      return ":p"
    }
    if (move === "swap") {
      return ":s"
    }
    return move
  }

  function serializeTreeLine(firstNode, { includePeerVariations, cursorNode }) {
    const parts = []
    const stack = [
      {
        action: "visit",
        node: firstNode,
        includePeerVariations,
      },
    ]
    while (stack.length > 0) {
      const item = stack.pop()
      if (item.action === "open") {
        parts.push("(")
        continue
      }
      if (item.action === "close") {
        parts.push(")")
        continue
      }

      const node = item.node
      if (!node || node.move === null) {
        throw new Error("Cannot serialize a tree node without a move")
      }
      parts.push(compactMoveToken(node.move))
      if (node === cursorNode) {
        parts.push(",")
      }

      const child = preferredChild(node)
      if (child) {
        stack.push({
          action: "visit",
          node: child,
          includePeerVariations: true,
        })
      }

      if (item.includePeerVariations && node.parent) {
        for (const peer of [...node.parent.children].reverse()) {
          if (peer === node) {
            continue
          }
          stack.push({ action: "close" })
          stack.push({
            action: "visit",
            node: peer,
            includePeerVariations: false,
          })
          stack.push({ action: "open" })
        }
      }
    }
    return parts.join("")
  }

  function buildHexataText(boardSize, tree) {
    const size = Number(boardSize)
    const first = preferredChild(tree.root)
    if (!first) {
      return `${size},`
    }
    const cursorNode = tree.cursor === mainlineTailNode(tree) ? null : tree.cursor
    const cursorPrefix = tree.cursor === tree.root ? "," : ""
    return `${size},${cursorPrefix}${serializeTreeLine(first, {
      includePeerVariations: true,
      cursorNode,
    })}`
  }

  function isAsciiLowerLetter(ch) {
    return ch >= "a" && ch <= "z"
  }

  function isAsciiUpperLetter(ch) {
    return ch >= "A" && ch <= "Z"
  }

  function isAsciiDigit(ch) {
    return ch >= "0" && ch <= "9"
  }

  function parseHexataTreeText(text, {
    defaultBoardSize,
    isBoardSizeSupported,
    materializeLine,
  }) {
    const raw = String(text ?? "")
    if (!raw) {
      return {
        valid: true,
        boardSize: Number(defaultBoardSize),
        tree: createMoveTree(),
      }
    }
    if (/\s/.test(raw)) {
      return { valid: false }
    }

    let pos = 0
    const tree = createMoveTree()
    let cursorNode = null

    function peek() {
      return pos < raw.length ? raw[pos] : null
    }

    function consume(token) {
      if (!raw.startsWith(token, pos)) {
        throw new Error(`Expected '${token}'`)
      }
      pos += token.length
    }

    function markCursor(node) {
      if (cursorNode !== null) {
        throw new Error("Only one cursor marker is allowed")
      }
      cursorNode = node
    }

    function parseMoveToken() {
      if (raw.startsWith(":p", pos)) {
        pos += 2
        return "pass"
      }
      if (raw.startsWith(":s", pos)) {
        pos += 2
        return "swap"
      }

      const start = pos
      while (pos < raw.length && (isAsciiLowerLetter(raw[pos]) || isAsciiUpperLetter(raw[pos]))) {
        if (isAsciiUpperLetter(raw[pos])) {
          throw new Error("Move coordinates must be lowercase")
        }
        pos += 1
      }
      if (pos === start) {
        throw new Error("Expected move token")
      }

      const letters = raw.slice(start, pos)
      const digitsStart = pos
      while (pos < raw.length && isAsciiDigit(raw[pos])) {
        pos += 1
      }
      if (pos === digitsStart || raw[digitsStart] === "0") {
        throw new Error("Expected row number")
      }
      return `${letters}${raw.slice(digitsStart, pos)}`
    }

    function validateMoves(moves, boardSize) {
      materializeLine(moves, boardSize)
    }

    function parseTreeLine(parent, parentMoves, boardSize) {
      let sawMove = false
      let currentParent = parent
      let currentMoves = [...parentMoves]

      while (true) {
        const ch = peek()
        if (ch === null || ch === ")") {
          break
        }
        if (ch === ",") {
          throw new Error("Invalid cursor marker")
        }

        const beforeParent = currentParent
        const beforeMoves = [...currentMoves]
        const move = normalizeMove(parseMoveToken())
        if (!move) {
          throw new Error("Invalid move token")
        }
        if (findChild(currentParent, move)) {
          throw new Error("Duplicate sibling move")
        }

        const child = appendChild(tree, currentParent, move)
        currentMoves = [...currentMoves, move]
        validateMoves(currentMoves, boardSize)
        sawMove = true

        if (peek() === ",") {
          markCursor(child)
          consume(",")
        }

        while (peek() === "(") {
          consume("(")
          parseTreeLine(beforeParent, beforeMoves, boardSize)
          consume(")")
        }

        currentParent = child
      }

      if (!sawMove) {
        throw new Error("Empty variation is not allowed")
      }
    }

    try {
      const sizeStart = pos
      while (pos < raw.length && isAsciiDigit(raw[pos])) {
        pos += 1
      }
      if (pos === sizeStart || raw[sizeStart] === "0") {
        return { valid: false }
      }
      const boardSize = Number(raw.slice(sizeStart, pos))
      if (!isBoardSizeSupported(boardSize)) {
        return { valid: false }
      }

      consume(",")
      if (peek() === ",") {
        markCursor(tree.root)
        consume(",")
      }
      if (pos < raw.length) {
        parseTreeLine(tree.root, [], boardSize)
      }
      if (pos !== raw.length) {
        return { valid: false }
      }

      tree.cursor = cursorNode || mainlineTailNode(tree)
      return {
        valid: true,
        boardSize,
        tree,
      }
    } catch (_error) {
      return { valid: false }
    }
  }

  function absorbMatchingSubtree(preferred, absorbed) {
    for (const child of [...absorbed.children]) {
      if (!removeChild(absorbed, child)) {
        throw new Error("Failed to detach absorbed child")
      }
      const match = findChild(preferred, child.move)
      if (match) {
        absorbMatchingSubtree(match, child)
      } else {
        child.parent = preferred
        preferred.children.push(child)
      }
    }
    if (absorbed.parent) {
      removeChild(absorbed.parent, absorbed)
    }
    return preferred
  }

  function mergeEquivalentSiblings(first, second) {
    if (first === second) {
      return first
    }
    const parent = first.parent
    if (!parent || second.parent !== parent || first.move !== second.move) {
      throw new Error("Can only merge sibling nodes with the same move")
    }
    return parent.children.indexOf(first) <= parent.children.indexOf(second)
      ? absorbMatchingSubtree(first, second)
      : absorbMatchingSubtree(second, first)
  }

  function buildMoveListCell(node, currentPathIds) {
    return {
      column: 0,
      current: false,
      label: node.move,
      node,
      played: currentPathIds.has(node.id),
      side: nodeDepth(node) % 2 === 1 ? "red" : "blue",
    }
  }

  function shiftMoveListSubtree(subtree, { rowDelta = 0, colDelta = 0 } = {}) {
    return subtree.map((placement) => ({
      row: placement.row + rowDelta,
      cell: {
        ...placement.cell,
        column: placement.cell.column + colDelta,
      },
    }))
  }

  function moveListRowRightEdge(subtree, row) {
    let right = -1
    for (const placement of subtree) {
      if (placement.row === row) {
        right = Math.max(right, placement.cell.column + 1)
      }
    }
    return right
  }

  function moveListRequiredShift(existing, incoming, minColumn) {
    let required = minColumn
    const existingByRow = new Map()
    for (const placement of existing) {
      if (!existingByRow.has(placement.row)) {
        existingByRow.set(placement.row, [])
      }
      existingByRow.get(placement.row).push(placement)
    }
    for (const incomingPlacement of incoming) {
      for (const existingPlacement of existingByRow.get(incomingPlacement.row) || []) {
        required = Math.max(
          required,
          (existingPlacement.cell.column + 1) - incomingPlacement.cell.column,
        )
      }
    }
    return required
  }

  function packMoveListSubtrees(subtrees) {
    let packed = []
    subtrees.forEach((subtree, index) => {
      if (index === 0) {
        packed = [...packed, ...subtree]
        return
      }
      const shift = moveListRequiredShift(
        packed,
        subtree,
        moveListRowRightEdge(packed, 0),
      )
      packed = [...packed, ...shiftMoveListSubtree(subtree, { colDelta: shift })]
    })
    return packed
  }

  function buildMoveListSubtree(node, currentPathIds) {
    const root = [{
      row: 0,
      cell: buildMoveListCell(node, currentPathIds),
    }]
    if (node.children.length === 0) {
      return root
    }
    const children = packMoveListSubtrees(
      node.children.map((child) => buildMoveListSubtree(child, currentPathIds)),
    )
    return [
      ...root,
      ...shiftMoveListSubtree(children, { rowDelta: 1 }),
    ]
  }

  function buildMoveListView(tree) {
    const path = currentPathNodes(tree)
    const currentPathIds = new Set(path.map((node) => node.id))
    if (tree.root.children.length === 0) {
      return { rows: [], widthColumns: 0 }
    }

    const packed = packMoveListSubtrees(
      tree.root.children.map((child) => buildMoveListSubtree(child, currentPathIds)),
    )
    for (const placement of packed) {
      placement.cell.current = placement.cell.node === tree.cursor
    }

    const laneWidths = new Map()
    for (const placement of packed) {
      laneWidths.set(
        placement.cell.column,
        Math.max(laneWidths.get(placement.cell.column) || 0, placement.cell.label.length),
      )
    }

    const laneStarts = new Map()
    let nextStart = 0
    const maxLane = Math.max(...laneWidths.keys())
    for (let lane = 0; lane <= maxLane; lane += 1) {
      laneStarts.set(lane, nextStart)
      nextStart += (laneWidths.get(lane) || 0) + 1
    }

    const rowsByIndex = new Map()
    for (const placement of packed) {
      const cell = {
        ...placement.cell,
        column: laneStarts.get(placement.cell.column) || 0,
      }
      if (!rowsByIndex.has(placement.row)) {
        rowsByIndex.set(placement.row, [])
      }
      rowsByIndex.get(placement.row).push(cell)
    }

    const maxRow = Math.max(...rowsByIndex.keys())
    const rows = []
    for (let row = 0; row <= maxRow; row += 1) {
      rows.push({
        ply: row + 1,
        cells: [...(rowsByIndex.get(row) || [])].sort((a, b) => a.column - b.column),
      })
    }

    return {
      rows,
      widthColumns: Math.max(0, nextStart - 1),
    }
  }

  function renderBranchMoveList({ container, view, selectNode }) {
    container.replaceChildren()
    for (const row of view.rows) {
      const rowElement = document.createElement("div")
      rowElement.className = "move-list-row move-list-branch-row"

      const ply = document.createElement("span")
      ply.className = "move-list-ply"
      ply.textContent = `${row.ply}.`
      rowElement.appendChild(ply)

      const track = document.createElement("span")
      track.className = "move-list-branch-track"
      track.style.minWidth = `${Math.max(view.widthColumns, 1)}ch`

      for (const cell of row.cells) {
        const move = document.createElement("span")
        const classes = ["move-list-move", "move-list-link", "move-list-branch-cell"]
        if (cell.played) {
          classes.push(cell.side === "red" ? "move-list-red" : "move-list-blue")
        } else {
          classes.push("move-list-future")
        }
        if (cell.current) {
          classes.push("move-list-current")
          move.dataset.currentMove = "true"
        }
        move.className = classes.join(" ")
        move.style.left = `${cell.column}ch`
        move.setAttribute("role", "button")
        move.tabIndex = 0
        move.textContent = cell.label
        const activate = () => {
          selectNode(cell.node)
        }
        move.addEventListener("click", activate)
        move.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault()
            activate()
          }
        })
        track.appendChild(move)
      }

      rowElement.appendChild(track)
      container.appendChild(rowElement)
    }

    const currentMove = container.querySelector("[data-current-move='true']")
    if (currentMove && typeof currentMove.scrollIntoView === "function") {
      currentMove.scrollIntoView({ block: "nearest", inline: "nearest" })
    }
  }

  function renderSwapSideAction({ boardSvg, state, board, canSwap, swapMove }) {
    if (!canSwap()) {
      return
    }
    renderSideActionHex({
      boardSvg,
      point: swapControlPoint(state.boardSize),
      toPlay: board.toPlay,
      labelText: "Swap",
      title: "Swap",
      onClick: swapMove,
      primaryText: "S",
    })
  }

  function createBranchingBoardEditor({
    elements,
    defaultBoardSize,
    minBoardSize = 1,
    maxBoardSize = 42,
    isLegalCell,
    materializeBoardState,
    boardCells,
    renderBoard,
    targetMoveForDrag = null,
  }) {
    const state = {
      boardSize: defaultBoardSize,
      tree: createMoveTree(),
      drag: null,
      showCoords: false,
      suppressNextBoardClick: false,
    }

    function legalCell(col, row, boardSize = state.boardSize) {
      return typeof isLegalCell === "function"
        ? Boolean(isLegalCell(col, row, boardSize))
        : (
            Number.isInteger(col)
            && Number.isInteger(row)
            && col >= 1
            && row >= 1
            && col <= Number(boardSize)
            && row <= Number(boardSize)
          )
    }

    function isBoardSizeSupported(boardSize) {
      const size = Number(boardSize)
      return Number.isInteger(size) && size >= minBoardSize && size <= maxBoardSize
    }

    function parseBoardSize(value) {
      if (String(value || "").trim() === "") {
        return null
      }
      const size = Number(value)
      if (!Number.isFinite(size)) {
        return null
      }
      return Math.max(minBoardSize, Math.min(maxBoardSize, Math.trunc(size)))
    }

    function currentMoves() {
      return currentPathMoves(state.tree)
    }

    function futureTailLines() {
      const current = currentMoves()
      const tail = mainlineTailMoves(state.tree)
      return tail.map((_, index) => (
        formatLine([...current, ...tail.slice(0, index + 1)])
      ))
    }

    function currentPly() {
      return currentPathNodes(state.tree).length
    }

    function materializeCurrentBoardState(moves, boardSize = state.boardSize) {
      return materializeBoardState(moves, boardSize)
    }

    function boardCellsForSize(boardSize = state.boardSize) {
      return boardCells(boardSize)
    }

    function currentLineText() {
      return buildHexataText(state.boardSize, state.tree)
    }

    function parseLine(text) {
      const parsed = parseHexataTreeText(text, {
        defaultBoardSize: state.boardSize,
        isBoardSizeSupported,
        materializeLine: materializeCurrentBoardState,
      })
      if (!parsed.valid) {
        return null
      }
      return parsed
    }

    function setHash() {
      replaceHash(`#${currentLineText()}`)
    }

    function syncSizeInput() {
      elements.sizeInput.value = String(state.boardSize)
      setNavButtonDisabled(elements.sizePrevBtn, state.boardSize <= minBoardSize)
      setNavButtonDisabled(elements.sizeNextBtn, state.boardSize >= maxBoardSize)
    }

    function syncNavigationButtons() {
      setNavButtonDisabled(elements.moveFirstBtn, state.tree.cursor === state.tree.root)
      setNavButtonDisabled(elements.movePrevBtn, state.tree.cursor === state.tree.root)
      setNavButtonDisabled(elements.moveNextBtn, preferredChild(state.tree.cursor) === null)
      setNavButtonDisabled(elements.moveLastBtn, preferredChild(state.tree.cursor) === null)
    }

    function renderMoveLine() {
      renderBranchMoveList({
        container: elements.moveList,
        view: buildMoveListView(state.tree),
        selectNode: (node) => {
          state.tree.cursor = node
          sync()
        },
      })
    }

    function renderBoardFromState() {
      renderBoard({
        state,
        currentMoves,
        materializeBoardState: materializeCurrentBoardState,
        boardCells: boardCellsForSize,
        playMove,
        goPrevious,
        goToLine,
        canSwap,
        swapMove,
      })
    }

    function sync({ rewriteHash = true, message = "" } = {}) {
      const board = materializeCurrentBoardState(currentMoves())
      setTurnStatus(elements.status, board.toPlay)
      elements.currentLine.textContent = currentLineText()
      elements.lineStatus.textContent = message
      syncSizeInput()
      syncNavigationButtons()
      renderMoveLine()
      renderBoardFromState()
      if (rewriteHash) {
        setHash()
      }
    }

    function movesStartWith(moves, prefix) {
      if (prefix.length > moves.length) {
        return false
      }
      for (let index = 0; index < prefix.length; index += 1) {
        if (moves[index] !== prefix[index]) {
          return false
        }
      }
      return true
    }

    function nodeForMoves(moves) {
      let node = state.tree.root
      for (const move of moves) {
        node = findChild(node, move)
        if (!node) {
          return null
        }
      }
      return node
    }

    function playMoveAtCursor(move) {
      const moveText = normalizeMove(move)
      if (!moveText) {
        return false
      }
      const nextMoves = [...currentMoves(), moveText]
      try {
        materializeCurrentBoardState(nextMoves)
      } catch (_error) {
        return false
      }

      const preferred = preferredChild(state.tree.cursor)
      if (preferred?.move === moveText) {
        state.tree.cursor = preferred
        return true
      }

      const existing = findChild(state.tree.cursor, moveText)
      if (existing) {
        state.tree.cursor = existing
        return true
      }

      state.tree.cursor = appendChild(state.tree, state.tree.cursor, moveText)
      return true
    }

    function playMove(move) {
      if (playMoveAtCursor(move)) {
        sync()
      }
    }

    function dragTargetMove(moveIndex, targetPoint) {
      if (typeof targetMoveForDrag === "function") {
        return targetMoveForDrag({
          moveIndex,
          targetPoint,
          currentMoves: currentMoves(),
        })
      }
      return formatCell(targetPoint.col, targetPoint.row)
    }

    function pruneInvalidDescendants(node, movesToNode) {
      const stack = [{ node, moves: movesToNode }]
      while (stack.length > 0) {
        const current = stack.pop()
        for (const child of [...current.node.children]) {
          const childMoves = [...current.moves, child.move]
          try {
            materializeCurrentBoardState(childMoves)
          } catch (_error) {
            removeChild(current.node, child)
            continue
          }
          stack.push({ node: child, moves: childMoves })
        }
      }
    }

    function rewriteMovePreservingLegalTail(moveIndex, targetPoint) {
      const pathNodes = currentPathNodes(state.tree)
      const index = Number(moveIndex)
      if (!Number.isInteger(index) || index < 0 || index >= pathNodes.length) {
        return false
      }

      const editNode = pathNodes[index]
      if (!tryParseCell(editNode.move)) {
        return false
      }

      const targetMove = normalizeMove(dragTargetMove(index, targetPoint))
      if (!targetMove || targetMove === editNode.move) {
        return false
      }

      const prefixBefore = pathNodes.slice(0, index).map((node) => node.move)
      const prefixMoves = [...prefixBefore, targetMove]
      try {
        materializeCurrentBoardState(prefixMoves)
      } catch (_error) {
        return false
      }

      const parent = editNode.parent
      const existing = parent ? findChild(parent, targetMove) : null
      const editedTailMoves = pathNodes.slice(index + 1).map((node) => node.move)
      editNode.move = targetMove

      let mergeRoot = editNode
      if (existing && existing !== editNode) {
        mergeRoot = mergeEquivalentSiblings(editNode, existing)
      }
      pruneInvalidDescendants(mergeRoot, prefixMoves)

      let cursor = mergeRoot
      for (const move of editedTailMoves) {
        const child = findChild(cursor, move)
        if (!child) {
          break
        }
        cursor = child
      }
      state.tree.cursor = cursor
      sync()
      return true
    }

    function goToLine(line) {
      const raw = String(line || "").trim()
      const moves = parseMoves(raw)
      if (raw && moves.length === 0) {
        return
      }
      try {
        materializeCurrentBoardState(moves)
      } catch (_error) {
        return
      }

      const existing = nodeForMoves(moves)
      if (existing) {
        state.tree.cursor = existing
        sync()
        return
      }

      const current = currentMoves()
      if (!movesStartWith(moves, current)) {
        return
      }
      for (const move of moves.slice(current.length)) {
        if (!playMoveAtCursor(move)) {
          return
        }
      }
      sync()
    }

    function canSwap() {
      const moves = currentMoves()
      if (moves.length !== 1) {
        return false
      }
      const first = formatLine([moves[0]])
      return first !== "pass" && first !== "swap" && tryParseCell(first) !== null
    }

    function passMove() {
      playMove("pass")
    }

    function swapMove() {
      if (canSwap()) {
        playMove("swap")
      }
    }

    function handleSwapShortcut(event) {
      if (shouldIgnoreGlobalKeydown(event)) {
        return false
      }
      if (!(event.key === "s" || event.key === "S")) {
        return false
      }
      const moves = currentMoves()
      if (canSwap()) {
        event.preventDefault()
        playMove("swap")
        return true
      }
      if (moves.length === 2 && moves[1] === "swap") {
        event.preventDefault()
        goPrevious()
        return true
      }
      return false
    }

    function handlePassShortcut(event) {
      if (shouldIgnoreGlobalKeydown(event)) {
        return false
      }
      if (!(event.shiftKey && event.key === "P")) {
        return false
      }
      event.preventDefault()
      passMove()
      return true
    }

    function isCoordKey(event) {
      return (
        !shouldIgnoreGlobalKeydown(event)
        && String(event.key || "").toLowerCase() === "c"
        && !event.altKey
        && !event.ctrlKey
        && !event.metaKey
        && !event.shiftKey
      )
    }

    function handleCoordKeydown(event) {
      if (!isCoordKey(event)) {
        return false
      }
      if (!state.showCoords) {
        state.showCoords = true
        renderBoardFromState()
      }
      return true
    }

    function handleCoordKeyup(event) {
      if (String(event.key || "").toLowerCase() !== "c" || !state.showCoords) {
        return
      }
      state.showCoords = false
      renderBoardFromState()
    }

    function clearCoordOverlay() {
      if (!state.showCoords) {
        return
      }
      state.showCoords = false
      renderBoardFromState()
    }

    function resetTree() {
      state.tree = createMoveTree()
    }

    function resetBoard(boardSize = state.boardSize) {
      const size = parseBoardSize(boardSize)
      if (size === null) {
        syncSizeInput()
        return
      }
      state.boardSize = size
      resetTree()
      sync()
    }

    function resetToDefault() {
      state.boardSize = defaultBoardSize
      resetTree()
      state.drag = null
      state.showCoords = false
      state.suppressNextBoardClick = false
    }

    function applySizeInput() {
      const size = parseBoardSize(elements.sizeInput.value)
      if (size === null) {
        syncSizeInput()
        return
      }
      if (size === state.boardSize) {
        syncSizeInput()
        return
      }
      resetBoard(size)
    }

    function stepSize(delta) {
      const nextSize = Math.max(
        minBoardSize,
        Math.min(maxBoardSize, state.boardSize + Number(delta)),
      )
      if (nextSize === state.boardSize) {
        syncSizeInput()
        return false
      }
      resetBoard(nextSize)
      return true
    }

    function loadHash() {
      const hashText = decodeLocationHash()
      if (hashText === null || !hashText) {
        resetToDefault()
        sync()
        return
      }
      const parsed = parseLine(hashText)
      if (parsed) {
        state.boardSize = parsed.boardSize
        state.tree = parsed.tree
      } else {
        replaceHash("")
        resetToDefault()
      }
      sync()
    }

    function goPrevious() {
      const parent = state.tree.cursor.parent
      if (parent) {
        state.tree.cursor = parent
        sync()
      }
    }

    function goNext() {
      const child = preferredChild(state.tree.cursor)
      if (child) {
        state.tree.cursor = child
        sync()
      }
    }

    function goFirst() {
      if (state.tree.cursor !== state.tree.root) {
        state.tree.cursor = state.tree.root
        sync()
      }
    }

    function goLast() {
      if (!preferredChild(state.tree.cursor)) {
        return
      }
      while (preferredChild(state.tree.cursor)) {
        state.tree.cursor = preferredChild(state.tree.cursor)
      }
      sync()
    }

    function goSibling(direction) {
      const target = siblingCursor(state.tree, Number(direction))
      if (!target) {
        return false
      }
      state.tree.cursor = target
      sync()
      return true
    }

    function stepCursor(delta) {
      const direction = Number(delta)
      if (direction < 0 && state.tree.cursor.parent) {
        state.tree.cursor = state.tree.cursor.parent
        sync()
        return true
      }
      if (direction > 0 && preferredChild(state.tree.cursor)) {
        state.tree.cursor = preferredChild(state.tree.cursor)
        sync()
        return true
      }
      syncNavigationButtons()
      return false
    }

    function handleBranchingNavigationKeydown(event) {
      if (shouldIgnoreGlobalKeydown(event)) {
        return false
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault()
        goSibling(event.key === "ArrowLeft" ? -1 : 1)
        return true
      }
      if (event.key === "ArrowUp") {
        event.preventDefault()
        goPrevious()
        return true
      }
      if (event.key === "ArrowDown") {
        event.preventDefault()
        goNext()
        return true
      }
      return false
    }

    function deleteFromCursor() {
      const tail = preferredChild(state.tree.cursor)
      if (tail) {
        removeChild(state.tree.cursor, tail)
        sync()
        return
      }
      if (state.tree.cursor.parent) {
        const parent = state.tree.cursor.parent
        removeChild(parent, state.tree.cursor)
        state.tree.cursor = parent
        sync()
      }
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
      if (!legalCell(col, row)) {
        return null
      }
      return { col, row }
    }

    function pointFromClientPosition(clientX, clientY) {
      return pointFromHexElement(document.elementFromPoint(clientX, clientY))
    }

    function stoneMoveIndex(stone) {
      if (stone?.ply === "S") {
        return 0
      }
      const ply = Number(stone?.ply)
      return Number.isInteger(ply) && ply > 0 ? ply - 1 : null
    }

    function dragTargetFromPoint(sourcePoint, point, occupied) {
      if (!point || pointsEqual(sourcePoint, point)) {
        return null
      }
      if (occupied.has(pointKey(point.col, point.row))) {
        return null
      }
      return { col: point.col, row: point.row }
    }

    function suppressNextBoardClick() {
      state.suppressNextBoardClick = true
      window.setTimeout(() => {
        state.suppressNextBoardClick = false
      }, 0)
    }

    function beginBoardDrag(event, point, stone) {
      const moveIndex = stoneMoveIndex(stone)
      if (!Number.isInteger(moveIndex) || moveIndex >= currentPly()) {
        return
      }
      state.drag = {
        pointerId: event.pointerId,
        sourceIndex: moveIndex,
        startPoint: { col: point.col, row: point.row },
        sourceColor: stone.color,
        sourceIsLast: Boolean(stone.isLast),
        targetPoint: null,
      }
      elements.board.setPointerCapture(event.pointerId)
      renderBoardFromState()
    }

    function releaseBoardPointer(pointerId) {
      if (elements.board.hasPointerCapture(pointerId)) {
        elements.board.releasePointerCapture(pointerId)
      }
    }

    function updateBoardDragTarget(event) {
      if (!state.drag || event.pointerId !== state.drag.pointerId) {
        return
      }
      const board = materializeCurrentBoardState(currentMoves())
      const nextTarget = dragTargetFromPoint(
        state.drag.startPoint,
        pointFromClientPosition(event.clientX, event.clientY),
        board.occupied,
      )
      const targetUnchanged =
        (state.drag.targetPoint === null && nextTarget === null)
        || (state.drag.targetPoint !== null && nextTarget !== null && pointsEqual(state.drag.targetPoint, nextTarget))
      if (targetUnchanged) {
        return
      }
      state.drag.targetPoint = nextTarget
      renderBoardFromState()
    }

    function handleBoardPointerDown(event) {
      if (state.drag || event.pointerType === "touch" || event.button !== 0) {
        return
      }
      const point = pointFromHexElement(event.target)
      if (!point) {
        return
      }
      const board = materializeCurrentBoardState(currentMoves())
      const stone = board.occupied.get(pointKey(point.col, point.row)) || null
      if (!stone) {
        return
      }
      beginBoardDrag(event, point, stone)
    }

    function handleBoardPointerMove(event) {
      updateBoardDragTarget(event)
    }

    function handleBoardPointerUp(event) {
      if (!state.drag || event.pointerId !== state.drag.pointerId) {
        return
      }
      const interaction = state.drag
      const releasePoint = pointFromClientPosition(event.clientX, event.clientY)
      const releasedOnSource = Boolean(releasePoint && pointsEqual(interaction.startPoint, releasePoint))
      releaseBoardPointer(event.pointerId)
      state.drag = null
      let changed = false
      if (releasedOnSource) {
        if (interaction.sourceIsLast) {
          goPrevious()
          changed = true
        }
        if (!changed) {
          renderBoardFromState()
        }
        suppressNextBoardClick()
        return
      }
      if (!releasePoint) {
        renderBoardFromState()
        suppressNextBoardClick()
        return
      }
      const board = materializeCurrentBoardState(currentMoves())
      const targetPoint = dragTargetFromPoint(
        interaction.startPoint,
        releasePoint,
        board.occupied,
      )
      if (targetPoint) {
        changed = rewriteMovePreservingLegalTail(interaction.sourceIndex, targetPoint)
      }
      if (!changed) {
        renderBoardFromState()
      }
      suppressNextBoardClick()
    }

    function handleBoardPointerCancel(event) {
      if (!state.drag || event.pointerId !== state.drag.pointerId) {
        return
      }
      releaseBoardPointer(event.pointerId)
      state.drag = null
      renderBoardFromState()
      suppressNextBoardClick()
    }

    function handleBoardClickCapture(event) {
      if (!state.suppressNextBoardClick) {
        return
      }
      state.suppressNextBoardClick = false
      event.preventDefault()
      event.stopPropagation()
    }

    elements.currentLine.addEventListener("click", () => {
      void copyTextToClipboard(currentLineText())
    })
    elements.sizeStepper.addEventListener("contextmenu", (event) => {
      event.preventDefault()
    }, { capture: true })
    elements.sizeStepper.addEventListener("selectstart", (event) => {
      event.preventDefault()
    }, { capture: true })
    elements.moveNav.addEventListener("contextmenu", (event) => {
      event.preventDefault()
    }, { capture: true })
    elements.moveNav.addEventListener("selectstart", (event) => {
      event.preventDefault()
    }, { capture: true })
    elements.board.addEventListener("click", handleBoardClickCapture, { capture: true })
    elements.board.addEventListener("pointerdown", handleBoardPointerDown)
    elements.board.addEventListener("pointermove", handleBoardPointerMove)
    elements.board.addEventListener("pointerup", handleBoardPointerUp)
    elements.board.addEventListener("pointercancel", handleBoardPointerCancel)
    elements.resetBtn.addEventListener("click", () => resetBoard())
    installHoldButton(elements.sizePrevBtn, () => stepSize(-1))
    installHoldButton(elements.sizeNextBtn, () => stepSize(1))
    installHoldButton(elements.movePrevBtn, () => stepCursor(-1))
    installHoldButton(elements.moveNextBtn, () => stepCursor(1))
    elements.moveFirstBtn.addEventListener("click", () => {
      if (!navButtonDisabled(elements.moveFirstBtn)) {
        goFirst()
      }
    })
    elements.moveLastBtn.addEventListener("click", () => {
      if (!navButtonDisabled(elements.moveLastBtn)) {
        goLast()
      }
    })
    elements.sizeInput.addEventListener("blur", syncSizeInput)
    elements.sizeInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        applySizeInput()
        elements.sizeInput.blur()
      } else if (event.key === "Escape") {
        syncSizeInput()
        elements.sizeInput.blur()
      }
    })
    window.addEventListener("hashchange", loadHash)
    window.addEventListener("keydown", (event) => {
      if (
        handleCoordKeydown(event)
        || handleSwapShortcut(event)
        || handlePassShortcut(event)
        || handleBranchingNavigationKeydown(event)
      ) {
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
    window.addEventListener("keyup", handleCoordKeyup)
    window.addEventListener("blur", clearCoordOverlay)

    loadHash()

    return {
      currentMoves,
      futureTailLines,
      goToLine,
      resetBoard,
      state,
      sync,
    }
  }

  window.HexBoardEditor = {
    createBranchingBoardEditor,
    renderSwapSideAction,
  }
})()
