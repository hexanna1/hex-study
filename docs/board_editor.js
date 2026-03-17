(() => {
  const {
    createBoardPointerController,
    decodeLocationHash,
    handleStandardKeydown,
    installHoldButton,
    lerpRgb,
    navButtonDisabled,
    replaceHash,
    rgbText,
    scrollChildIntoView,
    setButtonDisabled,
    setNavButtonDisabled,
    setTurnStatus,
    shouldIgnoreGlobalKeydown,
    syncPressedButtonGroup,
    THEME: {
      BLUE_RGB,
      OFF_WHITE_RGB,
      RED_RGB,
    },
  } = window.HexStudyUI
  const {
    createPositionSnapshot,
    formatCell,
    formatLine,
    IGNORED_HEXWORLD_TOKENS,
    parseMoves,
    parseHexWorldPrefix,
    pointKey,
    positionKey,
    tokenizeHexWorldMoveStream,
    tryParseCell,
  } = globalThis.HexPosition
  const ROOT_ID = 0
  const SVG_NS = "http://www.w3.org/2000/svg"
  const DEFAULT_BOARD_ORIENTATION = "flat"
  const DIAMOND_HASH_FLAG = "r9"

  function requiredElement(root, id) {
    const element = root.getElementById(id)
    if (!element) {
      throw new Error(`Missing board editor element '#${id}'`)
    }
    return element
  }

  function collectBoardEditorElements(statusId, root = document) {
    const ids = [
      "board",
      "current-line",
      "export-svg-link",
      "line-load-btn",
      "line-status",
      "move-nav",
      "move-first-btn",
      "move-last-btn",
      "move-list",
      "move-undo-btn",
      "move-redo-btn",
      "move-delete-btn",
      "move-numbers-off-btn",
      "move-numbers-on-btn",
      "move-next-btn",
      "move-prev-btn",
      "orientation-diamond-btn",
      "orientation-flat-btn",
      "reset-btn",
      "shortcut-help-link",
      "shortcut-help-popover",
      "size-input",
      "size-next-btn",
      "size-prev-btn",
      "size-stepper",
    ]
    const elements = Object.fromEntries(ids.map((id) => [id, requiredElement(root, id)]))
    return {
      board: elements.board,
      currentLine: elements["current-line"],
      exportSvgLink: elements["export-svg-link"],
      lineLoadBtn: elements["line-load-btn"],
      lineStatus: elements["line-status"],
      moveNav: elements["move-nav"],
      moveFirstBtn: elements["move-first-btn"],
      moveLastBtn: elements["move-last-btn"],
      moveList: elements["move-list"],
      moveUndoBtn: elements["move-undo-btn"],
      moveRedoBtn: elements["move-redo-btn"],
      moveDeleteBtn: elements["move-delete-btn"],
      moveNumbersOffBtn: elements["move-numbers-off-btn"],
      moveNumbersOnBtn: elements["move-numbers-on-btn"],
      moveNextBtn: elements["move-next-btn"],
      movePrevBtn: elements["move-prev-btn"],
      orientationDiamondBtn: elements["orientation-diamond-btn"],
      orientationFlatBtn: elements["orientation-flat-btn"],
      resetBtn: elements["reset-btn"],
      shortcutHelpLink: elements["shortcut-help-link"],
      shortcutHelpPopover: elements["shortcut-help-popover"],
      sizeInput: elements["size-input"],
      sizeNextBtn: elements["size-next-btn"],
      sizePrevBtn: elements["size-prev-btn"],
      sizeStepper: elements["size-stepper"],
      status: requiredElement(root, statusId),
    }
  }

  function normalizeBoardOrientation(orientation) {
    return orientation === "diamond" ? "diamond" : DEFAULT_BOARD_ORIENTATION
  }

  function toggleBoardOrientation(orientation) {
    return normalizeBoardOrientation(orientation) === "diamond" ? "flat" : "diamond"
  }

  const SHORTCUT_HELP_GROUPS = [
    {
      title: "Navigation",
      lines: [
        [
          { keys: ["p"], text: "prev" },
          { keys: ["n"], text: "next" },
          { keys: ["f"], text: "first" },
          { keys: ["l"], text: "last" },
        ],
        [
          { keys: ["←", "→"], text: "branch" },
        ],
      ],
    },
    {
      title: "Moves / edit",
      lines: [
        [
          { keys: ["shift+p"], text: "pass" },
          { keys: ["s"], text: "swap" },
        ],
        [
          { action: "left-drag", text: "move stone" },
          { keys: ["del"], text: "delete tail" },
        ],
        [
          { keys: ["ctrl+z"], text: "undo" },
          { keys: ["ctrl+y"], text: "redo" },
        ],
      ],
    },
    {
      title: "Display",
      lines: [
        [
          { keys: ["c"], text: "coords" },
          { keys: ["m"], text: "moves" },
          { keys: ["shift+o"], text: "orient" },
        ],
      ],
    },
  ]

  function appendShortcutKeys(container, shortcut) {
    if (shortcut.prefix) {
      container.append(document.createTextNode(shortcut.prefix))
    }
    if (Array.isArray(shortcut.keys) && shortcut.keys.length > 0) {
      shortcut.keys.forEach((key, index) => {
        if (index > 0) {
          container.append(document.createTextNode("/"))
        }
        const kbd = document.createElement("kbd")
        kbd.textContent = key
        container.append(kbd)
      })
      return
    }
    container.append(document.createTextNode(shortcut.action || ""))
  }

  function appendShortcutPair(container, shortcut) {
    const pair = document.createElement("span")
    pair.className = "shortcut-help-pair"
    appendShortcutKeys(pair, shortcut)
    pair.append(document.createTextNode(`:${shortcut.text}`))
    container.append(pair)
  }

  function renderShortcutHelpPopover(popover) {
    if (!(popover instanceof HTMLElement)) {
      return
    }
    popover.replaceChildren()
    popover.setAttribute("aria-label", "Shortcuts")

    const title = document.createElement("h2")
    title.textContent = "Help (? to hide)"
    popover.append(title)

    for (const group of SHORTCUT_HELP_GROUPS) {
      const section = document.createElement("section")
      section.className = "shortcut-help-section"

      const heading = document.createElement("h3")
      heading.textContent = group.title
      section.append(heading)

      for (const line of group.lines) {
        const row = document.createElement("p")
        row.className = "shortcut-help-line"
        for (const shortcut of line) {
          appendShortcutPair(row, shortcut)
        }
        section.append(row)
      }

      popover.append(section)
    }
  }

  function branchChildrenByKey(branchChildren, occupied) {
    const byKey = new Map()
    for (const child of branchChildren || []) {
      const point = child?.point
      if (!point) {
        continue
      }
      const key = pointKey(point.col, point.row)
      if (occupied?.has?.(key)) {
        continue
      }
      byKey.set(key, child)
    }
    return byKey
  }

  function branchOutlineColor(toPlay, isMainline) {
    const color = toPlay === "red" ? RED_RGB : BLUE_RGB
    return rgbText(isMainline ? color : lerpRgb(color, OFF_WHITE_RGB, 0.5))
  }

  function applyBranchOutline(polygon, child, toPlay) {
    const color = branchOutlineColor(toPlay, child?.isMainline)
    polygon.classList.add(child?.isMainline ? "branch-mainline" : "branch-variation")
    polygon.style.stroke = color
    polygon.style.strokeWidth = child?.isMainline ? "2.2" : "1.6"
    polygon.style.strokeLinejoin = "round"
    polygon.style.setProperty("--hover-outline", color)
  }

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

  function cloneMoveTree(tree) {
    const root = {
      id: tree.root.id,
      move: tree.root.move,
      parent: null,
      children: [],
    }
    const nodesById = new Map([[root.id, root]])
    const stack = [{ source: tree.root, target: root }]
    while (stack.length > 0) {
      const { source, target } = stack.pop()
      for (const child of source.children) {
        const next = {
          id: child.id,
          move: child.move,
          parent: target,
          children: [],
        }
        target.children.push(next)
        nodesById.set(next.id, next)
        stack.push({ source: child, target: next })
      }
    }
    return {
      root,
      cursor: nodesById.get(tree.cursor?.id) || root,
      nextId: tree.nextId,
    }
  }

  function moveTreeSignature(tree) {
    const tokens = []
    const stack = [tree.root]
    while (stack.length > 0) {
      const node = stack.pop()
      tokens.push([node.move, node.children.length])
      for (const child of [...node.children].reverse()) {
        stack.push(child)
      }
    }
    return JSON.stringify([tree.cursor?.id ?? ROOT_ID, tokens])
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

  function pathNodesTo(tree, cursor) {
    const path = []
    let node = cursor
    while (node && node.parent !== null) {
      path.push(node)
      node = node.parent
    }
    if (node !== tree.root) {
      throw new Error("Cursor does not belong to move tree")
    }
    path.reverse()
    return path
  }

  function currentPathNodes(tree) {
    return pathNodesTo(tree, tree.cursor)
  }

  function currentPathMoves(tree) {
    return currentPathNodes(tree).map((node) => node.move)
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

  function buildHashText(boardSize, tree, {
    boardOrientation = DEFAULT_BOARD_ORIENTATION,
    showMoveNumbers = false,
  } = {}) {
    // Keep the copyable line as Hexata branch text; only the URL hash carries UI flags.
    const text = buildHexataText(boardSize, tree)
    const orientationFlag = normalizeBoardOrientation(boardOrientation) === "diamond" ? DIAMOND_HASH_FLAG : ""
    const moveNumberFlag = showMoveNumbers ? "n" : ""
    return text.replace(/^([1-9][0-9]*),/, `$1${orientationFlag}${moveNumberFlag},`)
  }

  function displayOptionsFromHexWorldConfigs(configs) {
    return {
      boardOrientation: configs.includes(DIAMOND_HASH_FLAG) ? "diamond" : DEFAULT_BOARD_ORIENTATION,
      showMoveNumbers: configs.includes("n"),
    }
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
        boardOrientation: DEFAULT_BOARD_ORIENTATION,
        showMoveNumbers: false,
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

    function consumeIgnoredHexWorldTokens() {
      while (true) {
        const token = IGNORED_HEXWORLD_TOKENS.find((candidate) => raw.startsWith(candidate, pos))
        if (token === undefined) {
          return
        }
        pos += token.length
      }
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
      if (pos === digitsStart) {
        throw new Error("Expected row number")
      }
      return `${letters}${Number(raw.slice(digitsStart, pos))}`
    }

    function validateMoves(moves, boardSize) {
      materializeLine(moves, boardSize)
    }

    function parseTreeLine(parent, parentMoves, boardSize) {
      let sawMove = false
      let currentParent = parent
      let currentMoves = [...parentMoves]

      while (true) {
        consumeIgnoredHexWorldTokens()
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

        consumeIgnoredHexWorldTokens()
        if (peek() === ",") {
          markCursor(child)
          consume(",")
        }

        while (true) {
          consumeIgnoredHexWorldTokens()
          if (peek() !== "(") {
            break
          }
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
      const prefixEnd = raw.indexOf(",")
      if (prefixEnd < 0) {
        return { valid: false }
      }
      const { cols, rows, configs } = parseHexWorldPrefix(raw.slice(0, prefixEnd))
      if (cols !== rows || !isBoardSizeSupported(cols)) {
        return { valid: false }
      }
      const boardSize = cols
      const { boardOrientation, showMoveNumbers } = displayOptionsFromHexWorldConfigs(configs)
      pos = prefixEnd + 1
      consumeIgnoredHexWorldTokens()
      if (peek() === ",") {
        markCursor(tree.root)
        consume(",")
        consumeIgnoredHexWorldTokens()
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
        boardOrientation,
        showMoveNumbers,
        tree,
      }
    } catch (_error) {
      return { valid: false }
    }
  }

  function trimPositionText(text) {
    return String(text ?? "").trim()
  }

  function decodePositionFragment(fragment) {
    const text = String(fragment ?? "").trim()
    try {
      return decodeURIComponent(text)
    } catch (_error) {
      return text
    }
  }

  function hexWorldPositionText(text) {
    const raw = trimPositionText(text)
    const hashIndex = raw.indexOf("#")
    if (hashIndex < 0) {
      return raw
    }
    return decodePositionFragment(raw.slice(hashIndex + 1))
  }

  function lineInputPositionText(text) {
    const raw = trimPositionText(text)
    const hashIndex = raw.indexOf("#")
    if (hashIndex < 0) {
      return raw
    }
    const path = raw.slice(0, hashIndex)
    if (path === "" || /(?:^|\/)(?:hex|y)\.html(?:\?.*)?$/i.test(path)) {
      return decodePositionFragment(raw.slice(hashIndex + 1))
    }
    return raw
  }

  function errorMessage(error) {
    return error instanceof Error && error.message ? error.message : "Unknown error"
  }

  function safeFileStem(text, fallback = "board") {
    const stem = String(text || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
    return stem || fallback
  }

  function moveFileToken(move) {
    const token = compactMoveToken(move)
    return token.startsWith(":") ? token.slice(1) : token
  }

  function linearPathFileStem(moves) {
    return safeFileStem((moves || []).map(moveFileToken).join(""), "root")
  }

  function xmlCommentText(text) {
    return String(text ?? "").replace(/--+/g, "-")
  }

  function applyExportSvgSize(svg) {
    if (svg.getAttribute("width") && svg.getAttribute("height")) {
      return
    }
    const parts = String(svg.getAttribute("viewBox") || "")
      .trim()
      .split(/\s+/)
      .map(Number)
    if (parts.length !== 4 || parts.some((part) => !Number.isFinite(part))) {
      return
    }
    svg.setAttribute("width", String(Math.ceil(parts[2])))
    svg.setAttribute("height", String(Math.ceil(parts[3])))
  }

  function exportSvgStyle(svg) {
    const fontFamily = (window.getComputedStyle(svg).fontFamily || window.getComputedStyle(document.body).fontFamily || "serif")
      .replace(/;/g, "")
    const coordSample = svg.querySelector(".coord-text")
    const coordFill = coordSample ? window.getComputedStyle(coordSample).fill : "currentColor"
    return `
      .board-hover-hit { display: none; }
      .board-hex {
        stroke-width: 0.75;
      }
      .cell-text,
      .cell-stack-text {
        fill: #111111;
        font-family: ${fontFamily};
        font-size: 12px;
        text-anchor: middle;
        dominant-baseline: middle;
      }
      .coord-text {
        fill: ${coordFill};
        font-family: ${fontFamily};
        font-size: 13px;
        font-weight: 700;
        text-anchor: middle;
        dominant-baseline: middle;
      }
      .last-move-dot,
      .board-ghost {
        pointer-events: none;
      }
    `
  }

  function serializeBoardSvg(svg) {
    if (!svg || String(svg.tagName || "").toLowerCase() !== "svg") {
      return ""
    }
    const clone = svg.cloneNode(true)
    clone.setAttribute("xmlns", SVG_NS)
    clone.setAttribute("version", "1.1")
    applyExportSvgSize(clone)

    const sourceComment = document.createComment(` ${xmlCommentText(window.location.href)} `)
    clone.insertBefore(sourceComment, clone.firstChild)

    const style = document.createElementNS(SVG_NS, "style")
    style.textContent = exportSvgStyle(svg)
    clone.insertBefore(style, sourceComment.nextSibling)

    return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}\n`
  }

  function downloadTextFile({ text, filename, type }) {
    if (!text) {
      return false
    }
    const blob = new Blob([text], { type })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
    return true
  }

  function tokenizeFlexibleMoveText(text) {
    const raw = String(text || "")
    const tokens = []
    let needMove = false
    let index = 0
    while (index < raw.length) {
      if (/\s/.test(raw[index])) {
        index += 1
        continue
      }

      if (isAsciiDigit(raw[index])) {
        const numberRe = /[0-9]+(?:\.\s*|\s+|$)/y
        numberRe.lastIndex = index
        const match = numberRe.exec(raw)
        if (!match) {
          throw new Error("Move number must be followed by '.' or whitespace")
        }
        if (needMove) {
          throw new Error("Move number must be followed by a move")
        }
        needMove = true
        index = numberRe.lastIndex
        continue
      }

      const moveRe = /resign(?![0-9])|swap(?![0-9])|pass(?![0-9])|[a-z]+[0-9]+/iy
      moveRe.lastIndex = index
      const match = moveRe.exec(raw)
      if (!match) {
        throw new Error("Unexpected move text")
      }
      tokens.push(match[0].toLowerCase())
      needMove = false
      index = moveRe.lastIndex
    }

    if (needMove) {
      throw new Error("Move number must be followed by a move")
    }
    if (tokens.length === 0) {
      throw new Error("Move text is empty")
    }
    return tokens
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
    const lastPly = view.rows.reduce((maxPly, row) => Math.max(maxPly, Number(row.ply) || 0), 1)
    const plyWidth = `${lastPly}.`.length
    container.style.setProperty("--move-list-ply-width", `${plyWidth}ch`)
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
    scrollChildIntoView(container, currentMove, { inline: true })
  }

  function createBranchingBoardEditor({
    game,
    elements,
    defaultBoardSize,
    minBoardSize = 1,
    maxBoardSize = 42,
    isLegalCell,
    materializeBoardState,
    getBoardCells,
    renderBoard,
    supportsHexWorldImport = false,
    targetMoveForDrag = null,
    getBoardDecorations = null,
  }) {
    const session = {
      boardSize: defaultBoardSize,
      tree: createMoveTree(),
      editUndo: [],
      editRedo: [],
    }
    const ui = {
      boardOrientation: DEFAULT_BOARD_ORIENTATION,
      drag: null,
      showCoords: false,
      showMoveNumbers: false,
    }
    let boardPointerController = null
    let shortcutHelpOpen = false
    let currentPositionContext = null
    const positionListeners = new Set()

    function shortcutHelpPopoverElement() {
      return elements.shortcutHelpPopover instanceof HTMLElement ? elements.shortcutHelpPopover : null
    }

    function shortcutHelpLinkElement() {
      return elements.shortcutHelpLink instanceof HTMLElement ? elements.shortcutHelpLink : null
    }

    function shortcutHelpIsOpen() {
      return shortcutHelpOpen
    }

    function syncShortcutHelpExpanded() {
      const link = shortcutHelpLinkElement()
      if (link) {
        link.setAttribute("aria-expanded", shortcutHelpIsOpen() ? "true" : "false")
      }
    }

    function placeShortcutHelpPopover() {
      const popover = shortcutHelpPopoverElement()
      const link = shortcutHelpLinkElement()
      if (!popover || !link || !shortcutHelpIsOpen()) {
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
      const popover = shortcutHelpPopoverElement()
      if (!popover || shortcutHelpIsOpen() || typeof popover.showPopover !== "function") {
        return false
      }
      popover.showPopover()
      shortcutHelpOpen = true
      placeShortcutHelpPopover()
      syncShortcutHelpExpanded()
      return true
    }

    function hideShortcutHelp() {
      const popover = shortcutHelpPopoverElement()
      if (!popover || !shortcutHelpIsOpen() || typeof popover.hidePopover !== "function") {
        return false
      }
      popover.hidePopover()
      shortcutHelpOpen = false
      syncShortcutHelpExpanded()
      return true
    }

    function toggleShortcutHelp() {
      return shortcutHelpIsOpen() ? hideShortcutHelp() : showShortcutHelp()
    }

    function legalCell(col, row, boardSize = session.boardSize) {
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

    function currentMoves(tree = session.tree) {
      return currentPathMoves(tree)
    }

    function branchChildren(current) {
      return session.tree.cursor.children.map((child, index) => {
        const point = tryParseCell(child.move)
        return {
          move: child.move,
          line: formatLine([...current, child.move]),
          isMainline: index === 0,
          point: point ? { col: point.col, row: point.row } : null,
        }
      })
    }

    function rebuildPositionContext(prepared = null) {
      const moves = prepared?.moves ?? currentMoves()
      const key = prepared?.key ?? positionKey(game, session.boardSize, moves)
      const positionChanged = currentPositionContext?.position.key !== key
      const boardSizeChanged = currentPositionContext?.position.boardSize !== session.boardSize
      const board = positionChanged
        ? (prepared?.board ?? materializeCurrentBoardState(moves))
        : currentPositionContext.board
      const position = positionChanged
        ? createPositionSnapshot({ game, boardSize: session.boardSize, moves, board })
        : currentPositionContext.position
      currentPositionContext = {
        board,
        cells: boardSizeChanged ? cellsForBoardSize() : currentPositionContext.cells,
        branchChildren: branchChildren(moves),
        position,
      }
      return positionChanged
    }

    function getPosition() {
      return currentPositionContext?.position || null
    }

    function currentBoardState() {
      if (!currentPositionContext) {
        rebuildPositionContext()
      }
      return currentPositionContext.board
    }

    function subscribePosition(listener) {
      if (typeof listener !== "function") {
        throw new TypeError("Position listener must be a function")
      }
      positionListeners.add(listener)
      if (currentPositionContext) {
        listener(currentPositionContext.position)
      }
      return () => positionListeners.delete(listener)
    }

    function notifyPositionChanged() {
      for (const listener of positionListeners) {
        listener(currentPositionContext.position)
      }
    }

    function currentPly() {
      return currentPathNodes(session.tree).length
    }

    function canUndoEdit() {
      return session.editUndo.length > 0
    }

    function canRedoEdit() {
      return session.editRedo.length > 0
    }

    function prepareTransition(tree, cursor, boardSize) {
      if (!isBoardSizeSupported(boardSize)) {
        throw new Error("Unsupported board size")
      }
      const moves = pathNodesTo(tree, cursor).map((node) => node.move)
      const key = positionKey(game, boardSize, moves)
      const positionChanged = currentPositionContext?.position.key !== key
      return {
        key,
        moves,
        board: positionChanged
          ? materializeCurrentBoardState(moves, boardSize)
          : currentPositionContext.board,
      }
    }

    function commitTransition({
      tree = session.tree,
      cursor = tree.cursor,
      boardSize = session.boardSize,
      history = "preserve",
      message = "",
    } = {}) {
      if (history === "track" && tree === session.tree) {
        throw new Error("Tracked edits must install a detached tree")
      }
      if (history === "undo" && session.editUndo.at(-1)?.tree !== tree) {
        throw new Error("Undo transition must install the latest snapshot")
      }
      if (history === "redo" && session.editRedo.at(-1)?.tree !== tree) {
        throw new Error("Redo transition must install the latest snapshot")
      }
      if (!["preserve", "track", "clear", "undo", "redo"].includes(history)) {
        throw new Error("Unknown edit history transition")
      }
      const prepared = prepareTransition(tree, cursor, boardSize)
      const previousTree = session.tree

      tree.cursor = cursor
      session.boardSize = boardSize
      session.tree = tree
      if (history === "track") {
        session.editUndo.push({ tree: previousTree })
        session.editRedo.length = 0
      } else if (history === "clear") {
        session.editUndo.length = 0
        session.editRedo.length = 0
      } else if (history === "undo") {
        session.editUndo.pop()
        session.editRedo.push({ tree: previousTree })
      } else if (history === "redo") {
        session.editRedo.pop()
        session.editUndo.push({ tree: previousTree })
      }
      sync({ message, prepared })
      return true
    }

    function runTrackedEdit(mutate) {
      const tree = cloneMoveTree(session.tree)
      const beforeSignature = moveTreeSignature(tree)
      if (mutate(tree) === false) {
        return false
      }
      if (moveTreeSignature(tree) === beforeSignature) {
        return false
      }
      return commitTransition({ tree, history: "track" })
    }

    function undoEdit() {
      if (session.editUndo.length === 0) {
        return false
      }
      const target = session.editUndo[session.editUndo.length - 1]
      return commitTransition({
        tree: target.tree,
        history: "undo",
      })
    }

    function redoEdit() {
      if (session.editRedo.length === 0) {
        return false
      }
      const target = session.editRedo[session.editRedo.length - 1]
      return commitTransition({
        tree: target.tree,
        history: "redo",
      })
    }

    function materializeCurrentBoardState(moves, boardSize = session.boardSize) {
      return materializeBoardState(moves, boardSize)
    }

    function isLegalMoveSequence(moves) {
      try {
        materializeCurrentBoardState(moves)
        return true
      } catch (_error) {
        return false
      }
    }

    function cellsForBoardSize(boardSize = session.boardSize) {
      const cells = getBoardCells(boardSize)
      if (!Array.isArray(cells)) {
        throw new TypeError("Board cells must be an array")
      }
      return cells
    }

    function currentLineText() {
      return buildHexataText(session.boardSize, session.tree)
    }

    function currentHashText() {
      return buildHashText(session.boardSize, session.tree, {
        boardOrientation: ui.boardOrientation,
        showMoveNumbers: ui.showMoveNumbers,
      })
    }

    function svgExportFileName() {
      const title = String(document.title || "Board")
        .replace(/\s+board\s+editor$/i, "")
      const orientationFlag = normalizeBoardOrientation(ui.boardOrientation) === "diamond" ? DIAMOND_HASH_FLAG : ""
      const moveNumberFlag = ui.showMoveNumbers ? "n" : ""
      const sizeText = `${session.boardSize}${orientationFlag}${moveNumberFlag}`
      return `${safeFileStem(title)}-${sizeText}-${linearPathFileStem(currentMoves())}.svg`
    }

    function downloadBoardSvg() {
      return downloadTextFile({
        text: serializeBoardSvg(elements.board),
        filename: svgExportFileName(),
        type: "image/svg+xml;charset=utf-8",
      })
    }

    function parseLine(text) {
      const parsed = parseHexataTreeText(text, {
        defaultBoardSize: session.boardSize,
        isBoardSizeSupported,
        materializeLine: materializeCurrentBoardState,
      })
      if (!parsed.valid) {
        return null
      }
      return parsed
    }

    function treeFromLineMoves(pastMoves, futureMoves = []) {
      const tree = createMoveTree()
      let node = tree.root
      for (const move of pastMoves) {
        node = appendChild(tree, node, move)
      }
      tree.cursor = node
      for (const move of futureMoves) {
        node = appendChild(tree, node, move)
      }
      return tree
    }

    function moveFromImportedCell(token) {
      const point = tryParseCell(token)
      if (!point) {
        throw new Error("Bad cell")
      }
      return formatCell(point.col, point.row)
    }

    function parseHexWorldMoveStream(stream, boardSize, previousMoves = []) {
      let allMoves = [...previousMoves]
      const moves = []
      for (const token of tokenizeHexWorldMoveStream(stream)) {
        let move = null
        if (IGNORED_HEXWORLD_TOKENS.includes(token)) {
          continue
        }
        if (token === ":p") {
          move = "pass"
        } else if (token === ":s") {
          if (allMoves.includes("swap") || allMoves.length !== 1 || !tryParseCell(allMoves[0])) {
            throw new Error("Bad HexWorld swap")
          }
          move = "swap"
        } else {
          move = moveFromImportedCell(token)
        }

        const nextMoves = [...allMoves, move]
        materializeCurrentBoardState(nextMoves, boardSize)
        moves.push(move)
        allMoves = nextMoves
      }
      return moves
    }

    function parseHexWorldImport(text) {
      if (!supportsHexWorldImport) {
        return null
      }
      const raw = hexWorldPositionText(text)
      if (!raw) {
        throw new Error("Position text is empty")
      }
      const parts = raw.split(",")
      if (parts.length > 3) {
        throw new Error("Too many HexWorld sections")
      }
      const { cols, rows, configs } = parseHexWorldPrefix(parts[0])
      if (cols !== rows || !isBoardSizeSupported(cols)) {
        throw new Error("Unsupported HexWorld board size")
      }
      const pastMoves = parseHexWorldMoveStream(parts[1] || "", cols)
      const futureMoves = parseHexWorldMoveStream(parts[2] || "", cols, pastMoves)
      return {
        boardSize: cols,
        ...displayOptionsFromHexWorldConfigs(configs),
        tree: treeFromLineMoves(pastMoves, futureMoves),
      }
    }

    function parseHexataImport(text) {
      const raw = trimPositionText(text)
      if (!raw) {
        throw new Error("Position text is empty")
      }
      const parsed = parseLine(raw)
      if (!parsed) {
        throw new Error("Invalid syntax")
      }
      return parsed
    }

    function parseFlexibleImport(text) {
      const raw = trimPositionText(text)
      if (!raw) {
        throw new Error("Position text is empty")
      }
      const moves = []
      for (const token of tokenizeFlexibleMoveText(raw)) {
        if (token === "resign") {
          continue
        }
        let move = null
        if (token === "pass") {
          move = "pass"
        } else if (token === "swap") {
          if (moves.length !== 1 || !tryParseCell(moves[0])) {
            throw new Error("Bad swap")
          }
          move = "swap"
        } else {
          move = moveFromImportedCell(token)
        }
        const nextMoves = [...moves, move]
        materializeCurrentBoardState(nextMoves, session.boardSize)
        moves.push(move)
      }
      return {
        boardSize: session.boardSize,
        tree: treeFromLineMoves(moves),
      }
    }

    function parseImportAttempt(source, parse) {
      try {
        return { parsed: parse(), error: "" }
      } catch (error) {
        return { parsed: null, error: `${source} parse failed: ${errorMessage(error)}` }
      }
    }

    function parseImportedPosition(text) {
      const raw = trimPositionText(text)
      const errors = []
      const hexWorld = supportsHexWorldImport
        ? parseImportAttempt("HexWorld", () => parseHexWorldImport(raw))
        : null
      if (hexWorld?.parsed) {
        return { parsed: hexWorld.parsed, errors: [] }
      }
      if (hexWorld?.error) {
        errors.push(hexWorld.error)
      }

      const hexata = parseImportAttempt("Hexata format", () => parseHexataImport(raw))
      if (hexata.parsed) {
        return { parsed: hexata.parsed, errors: [] }
      }
      if (hexata.error) {
        errors.push(hexata.error)
      }

      const flexible = parseImportAttempt("Flexible move format", () => parseFlexibleImport(raw))
      if (flexible.parsed) {
        return { parsed: flexible.parsed, errors: [] }
      }
      if (flexible.error) {
        errors.push(flexible.error)
      }
      return { parsed: null, errors }
    }

    function logImportFailures(errors) {
      for (const error of errors) {
        console.info(error)
      }
    }

    function installParsedPosition(parsed, { message = "", applyUiFlags = true } = {}) {
      if (applyUiFlags && parsed.boardOrientation !== undefined) {
        ui.boardOrientation = normalizeBoardOrientation(parsed.boardOrientation)
      }
      if (applyUiFlags && parsed.showMoveNumbers !== undefined) {
        ui.showMoveNumbers = Boolean(parsed.showMoveNumbers)
      }
      commitTransition({
        tree: parsed.tree,
        boardSize: parsed.boardSize,
        history: "clear",
        message,
      })
    }

    function loadPositionText(text, { message = "Loaded position." } = {}) {
      const result = parseImportedPosition(text)
      if (!result.parsed) {
        logImportFailures(result.errors)
        elements.lineStatus.textContent = "Could not parse position."
        return false
      }
      installParsedPosition(result.parsed, { message, applyUiFlags: false })
      return true
    }

    function loadLineInput({ message = "Loaded position." } = {}) {
      const text = lineInputPositionText(elements.currentLine.value)
      elements.currentLine.value = text
      return loadPositionText(text, { message })
    }

    function setHash() {
      replaceHash(`#${currentHashText()}`)
    }

    function syncSizeInput() {
      elements.sizeInput.value = String(session.boardSize)
      setButtonDisabled(elements.sizePrevBtn, session.boardSize <= minBoardSize)
      setButtonDisabled(elements.sizeNextBtn, session.boardSize >= maxBoardSize)
    }

    function syncNavigationButtons() {
      const deleteLabel = deleteLabelFromCursor()
      elements.moveDeleteBtn.setAttribute("aria-label", deleteLabel)
      elements.moveDeleteBtn.setAttribute("title", deleteLabel)
      setNavButtonDisabled(elements.moveFirstBtn, session.tree.cursor === session.tree.root)
      setNavButtonDisabled(elements.movePrevBtn, session.tree.cursor === session.tree.root)
      setNavButtonDisabled(elements.moveNextBtn, preferredChild(session.tree.cursor) === null)
      setNavButtonDisabled(elements.moveLastBtn, preferredChild(session.tree.cursor) === null)
      setNavButtonDisabled(elements.moveUndoBtn, !canUndoEdit())
      setNavButtonDisabled(elements.moveRedoBtn, !canRedoEdit())
      setNavButtonDisabled(elements.moveDeleteBtn, !canDeleteFromCursor())
    }

    function syncMoveNumberButton() {
      syncPressedButtonGroup([
        [false, elements.moveNumbersOffBtn],
        [true, elements.moveNumbersOnBtn],
      ], Boolean(ui.showMoveNumbers))
    }

    function syncBoardOrientationButton() {
      syncPressedButtonGroup([
        ["flat", elements.orientationFlatBtn],
        ["diamond", elements.orientationDiamondBtn],
      ], normalizeBoardOrientation(ui.boardOrientation))
    }

    function renderMoveLine() {
      renderBranchMoveList({
        container: elements.moveList,
        view: buildMoveListView(session.tree),
        selectNode,
      })
    }

    function refreshBoard() {
      if (!currentPositionContext) {
        rebuildPositionContext()
      }
      const decorations = typeof getBoardDecorations === "function"
        ? (getBoardDecorations(currentPositionContext.position) || {})
        : {}
      const analysisByKey = decorations.analysisByKey ?? new Map()
      if (!(analysisByKey instanceof Map)) {
        throw new TypeError("Board analysis overlays must be a Map")
      }
      renderBoard({
        analysisByKey,
        board: currentPositionContext.board,
        cells: currentPositionContext.cells,
        branchChildren: currentPositionContext.branchChildren,
        display: {
          boardOrientation: ui.boardOrientation,
          drag: ui.drag,
          showCoords: ui.showCoords,
          showMoveNumbers: ui.showMoveNumbers,
        },
        position: currentPositionContext.position,
      })
    }

    function sync({
      rewriteHash = true,
      message = "",
      cancelDrag = true,
      prepared = null,
    } = {}) {
      if (cancelDrag) {
        cancelBoardDrag()
      }
      const positionChanged = rebuildPositionContext(prepared)
      setTurnStatus(elements.status, currentPositionContext.position.toPlay)
      elements.currentLine.value = currentLineText()
      elements.lineStatus.textContent = message
      syncSizeInput()
      syncNavigationButtons()
      syncMoveNumberButton()
      syncBoardOrientationButton()
      renderMoveLine()
      refreshBoard()
      if (rewriteHash) {
        setHash()
      }
      if (positionChanged) {
        notifyPositionChanged()
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

    function movesFromLine(line) {
      const raw = String(line || "").trim()
      const moves = parseMoves(raw)
      if (raw && moves.length === 0) {
        return null
      }
      if (!isLegalMoveSequence(moves)) {
        return null
      }
      return moves
    }

    function selectNode(node) {
      if (!node || node === session.tree.cursor) {
        return false
      }
      return commitTransition({ cursor: node })
    }

    function playMoveAtCursor(tree, moveText) {
      const preferred = preferredChild(tree.cursor)
      if (preferred?.move === moveText) {
        tree.cursor = preferred
        return true
      }

      const existing = findChild(tree.cursor, moveText)
      if (existing) {
        tree.cursor = existing
        return true
      }

      tree.cursor = appendChild(tree, tree.cursor, moveText)
      return true
    }

    function playMove(move) {
      const moveText = normalizeMove(move)
      if (!moveText || !isLegalMoveSequence([...currentMoves(), moveText])) {
        return false
      }
      return runTrackedEdit((tree) => playMoveAtCursor(tree, moveText))
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
          if (!isLegalMoveSequence(childMoves)) {
            removeChild(current.node, child)
            continue
          }
          stack.push({ node: child, moves: childMoves })
        }
      }
    }

    function rewriteMovePreservingLegalTail(moveIndex, targetPoint) {
      const pathNodes = currentPathNodes(session.tree)
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
      if (!isLegalMoveSequence(prefixMoves)) {
        return false
      }

      return runTrackedEdit((tree) => {
        const pathNodes = currentPathNodes(tree)
        const editNode = pathNodes[index]
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
        tree.cursor = cursor
      })
    }

    function playLineAtCursor(tree, line) {
      const moves = movesFromLine(line)
      if (!moves) {
        return false
      }
      const current = currentMoves(tree)
      if (!movesStartWith(moves, current)) {
        return false
      }
      if (moves.length === current.length) {
        return false
      }
      for (const move of moves.slice(current.length)) {
        playMoveAtCursor(tree, move)
      }
      return true
    }

    function playLineFromCursor(line) {
      return runTrackedEdit((tree) => playLineAtCursor(tree, line))
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

    function activateLastStone() {
      if (canSwap()) {
        swapMove()
        return
      }
      if (preferredChild(session.tree.cursor)) {
        goPrevious()
        return
      }
      runTrackedEdit((tree) => {
        if (tree.cursor.parent) {
          deleteFromCursorAtCursor(tree)
        }
      })
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
      if (!ui.showCoords) {
        ui.showCoords = true
        refreshBoard()
      }
      return true
    }

    function handleCoordKeyup(event) {
      if (String(event.key || "").toLowerCase() !== "c" || !ui.showCoords) {
        return
      }
      ui.showCoords = false
      refreshBoard()
    }

    function handleMoveNumberShortcut(event) {
      if (
        shouldIgnoreGlobalKeydown(event)
        || String(event.key || "").toLowerCase() !== "m"
      ) {
        return false
      }
      event.preventDefault()
      ui.showMoveNumbers = !ui.showMoveNumbers
      sync({ cancelDrag: false })
      return true
    }

    function handleBoardOrientationShortcut(event) {
      if (
        shouldIgnoreGlobalKeydown(event)
        || !event.shiftKey
        || String(event.key || "").toLowerCase() !== "o"
      ) {
        return false
      }
      event.preventDefault()
      ui.boardOrientation = toggleBoardOrientation(ui.boardOrientation)
      sync({ cancelDrag: false })
      return true
    }

    function handleShortcutHelpShortcut(event) {
      if (!event || event.defaultPrevented || event.metaKey || event.key !== "?") {
        return false
      }
      const target = event.target
      if (target instanceof HTMLElement) {
        const tag = target.tagName.toLowerCase()
        if (tag === "input" || tag === "textarea" || target.isContentEditable) {
          return false
        }
      }
      if (!shortcutHelpPopoverElement()) {
        return false
      }
      event.preventDefault()
      toggleShortcutHelp()
      return true
    }

    function hasEditHistoryShortcutModifier(event) {
      return (event.ctrlKey || event.metaKey) && !event.altKey
    }

    function shouldIgnoreEditHistoryKeydown(event) {
      if (!event || event.defaultPrevented || event.altKey) {
        return true
      }
      const target = event.target
      if (target instanceof HTMLElement) {
        const tag = target.tagName.toLowerCase()
        if (tag === "input" || tag === "textarea" || target.isContentEditable) {
          return true
        }
      }
      return false
    }

    function handleEditHistoryShortcut(event) {
      if (shouldIgnoreEditHistoryKeydown(event) || !hasEditHistoryShortcutModifier(event)) {
        return false
      }
      const key = String(event.key || "").toLowerCase()
      const isUndo = key === "z" && !event.shiftKey
      const isRedo = key === "y" || (key === "z" && event.shiftKey)
      if (!isUndo && !isRedo) {
        return false
      }
      event.preventDefault()
      if (isRedo) {
        redoEdit()
      } else {
        undoEdit()
      }
      return true
    }

    function clearCoordOverlay() {
      if (!ui.showCoords) {
        return
      }
      ui.showCoords = false
      refreshBoard()
    }

    function cancelBoardDrag() {
      boardPointerController?.cancel({ notify: false })
      ui.drag = null
    }

    function resetBoard(boardSize = session.boardSize) {
      const size = parseBoardSize(boardSize)
      if (size === null) {
        syncSizeInput()
        return
      }
      commitTransition({
        tree: createMoveTree(),
        boardSize: size,
        history: "clear",
      })
    }

    function resetToDefault() {
      ui.boardOrientation = DEFAULT_BOARD_ORIENTATION
      ui.showCoords = false
      ui.showMoveNumbers = false
      commitTransition({
        tree: createMoveTree(),
        boardSize: defaultBoardSize,
        history: "clear",
      })
    }

    function applySizeInput() {
      const size = parseBoardSize(elements.sizeInput.value)
      if (size === null) {
        syncSizeInput()
        return
      }
      if (size === session.boardSize) {
        syncSizeInput()
        return
      }
      resetBoard(size)
    }

    function stepSize(delta) {
      const nextSize = Math.max(
        minBoardSize,
        Math.min(maxBoardSize, session.boardSize + Number(delta)),
      )
      if (nextSize === session.boardSize) {
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
        return
      }
      let parsed = parseLine(hashText)
      if (parsed === null && supportsHexWorldImport) {
        try {
          parsed = parseHexWorldImport(hashText)
        } catch (_error) {
          parsed = null
        }
      }
      if (parsed) {
        installParsedPosition(parsed)
      } else {
        replaceHash("")
        resetToDefault()
      }
    }

    function goPrevious() {
      const parent = session.tree.cursor.parent
      if (parent) {
        commitTransition({ cursor: parent })
      }
    }

    function goNext() {
      const child = preferredChild(session.tree.cursor)
      if (child) {
        commitTransition({ cursor: child })
      }
    }

    function goFirst() {
      if (session.tree.cursor !== session.tree.root) {
        commitTransition({ cursor: session.tree.root })
      }
    }

    function goLast() {
      if (!preferredChild(session.tree.cursor)) {
        return
      }
      let cursor = session.tree.cursor
      while (preferredChild(cursor)) {
        cursor = preferredChild(cursor)
      }
      commitTransition({ cursor })
    }

    function goSibling(direction) {
      const target = siblingCursor(session.tree, Number(direction))
      if (!target) {
        return false
      }
      return commitTransition({ cursor: target })
    }

    function stepCursor(delta) {
      const direction = Number(delta)
      if (direction < 0 && session.tree.cursor.parent) {
        return commitTransition({ cursor: session.tree.cursor.parent })
      }
      if (direction > 0 && preferredChild(session.tree.cursor)) {
        return commitTransition({ cursor: preferredChild(session.tree.cursor) })
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

    function canDeleteFromCursor() {
      return Boolean(preferredChild(session.tree.cursor) || session.tree.cursor.parent)
    }

    function deleteLabelFromCursor() {
      if (preferredChild(session.tree.cursor)) {
        return "Delete tail"
      }
      if (session.tree.cursor.parent) {
        return "Delete current move"
      }
      return "Delete move"
    }

    function deleteFromCursorAtCursor(tree) {
      const tail = preferredChild(tree.cursor)
      if (tail) {
        removeChild(tree.cursor, tail)
        return true
      }
      if (tree.cursor.parent) {
        const parent = tree.cursor.parent
        removeChild(parent, tree.cursor)
        tree.cursor = parent
        return true
      }
      return false
    }

    function deleteFromCursor() {
      return runTrackedEdit((tree) => deleteFromCursorAtCursor(tree))
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

    function boardDragData(point) {
      const board = currentBoardState()
      const stone = board.occupied.get(pointKey(point.col, point.row)) || null
      const moveIndex = stoneMoveIndex(stone)
      if (!Number.isInteger(moveIndex) || moveIndex >= currentPly()) {
        return null
      }
      return {
        sourceIndex: moveIndex,
        sourceColor: stone.color,
      }
    }

    function tapBoardPoint(point) {
      const board = currentBoardState()
      const stone = board.occupied.get(pointKey(point.col, point.row)) || null
      if (stone?.isLast) {
        activateLastStone()
      } else if (!stone) {
        playMove(formatCell(point.col, point.row))
      }
    }

    function beginBoardDrag(interaction) {
      ui.drag = {
        pointerId: interaction.pointerId,
        sourceIndex: interaction.dragData.sourceIndex,
        startPoint: interaction.startPoint,
        sourceColor: interaction.dragData.sourceColor,
        targetPoint: null,
      }
      refreshBoard()
    }

    function moveBoardDrag(_interaction, point) {
      if (!ui.drag) {
        return
      }
      const board = currentBoardState()
      const nextTarget = dragTargetFromPoint(ui.drag.startPoint, point, board.occupied)
      const targetUnchanged =
        (ui.drag.targetPoint === null && nextTarget === null)
        || (ui.drag.targetPoint !== null && nextTarget !== null && pointsEqual(ui.drag.targetPoint, nextTarget))
      if (targetUnchanged) {
        return
      }
      ui.drag.targetPoint = nextTarget
      refreshBoard()
    }

    function finishBoardDrag(interaction, releasePoint) {
      ui.drag = null
      const board = currentBoardState()
      const targetPoint = dragTargetFromPoint(interaction.startPoint, releasePoint, board.occupied)
      const changed = targetPoint
        ? rewriteMovePreservingLegalTail(interaction.dragData.sourceIndex, targetPoint)
        : false
      if (!changed) {
        refreshBoard()
      }
    }

    function cancelBoardPointerInteraction(interaction) {
      if (interaction.dragging) {
        ui.drag = null
        refreshBoard()
      }
    }

    renderShortcutHelpPopover(elements.shortcutHelpPopover)
    boardPointerController = createBoardPointerController({
      board: elements.board,
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
    elements.shortcutHelpPopover?.addEventListener("toggle", (event) => {
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

    elements.exportSvgLink?.querySelector("a")?.addEventListener("click", (event) => {
      event.preventDefault()
      downloadBoardSvg()
    })
    elements.currentLine.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault()
        if (loadLineInput()) {
          elements.currentLine.blur()
        }
      } else if (event.key === "Escape") {
        elements.currentLine.value = currentLineText()
        elements.lineStatus.textContent = ""
        elements.currentLine.blur()
      }
    })
    elements.lineLoadBtn.addEventListener("click", () => {
      loadLineInput()
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
    elements.resetBtn.addEventListener("click", () => resetBoard())
    elements.moveNumbersOffBtn?.addEventListener("click", () => {
      ui.showMoveNumbers = false
      sync({ cancelDrag: false })
    })
    elements.moveNumbersOnBtn?.addEventListener("click", () => {
      ui.showMoveNumbers = true
      sync({ cancelDrag: false })
    })
    elements.orientationFlatBtn?.addEventListener("click", () => {
      ui.boardOrientation = "flat"
      sync({ cancelDrag: false })
    })
    elements.orientationDiamondBtn?.addEventListener("click", () => {
      ui.boardOrientation = "diamond"
      sync({ cancelDrag: false })
    })
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
    elements.moveUndoBtn.addEventListener("click", () => {
      if (!navButtonDisabled(elements.moveUndoBtn)) {
        undoEdit()
      }
    })
    elements.moveRedoBtn.addEventListener("click", () => {
      if (!navButtonDisabled(elements.moveRedoBtn)) {
        redoEdit()
      }
    })
    elements.moveDeleteBtn.addEventListener("click", () => {
      if (!navButtonDisabled(elements.moveDeleteBtn)) {
        deleteFromCursor()
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
      elements.currentLine.value = pastedLine
      loadPositionText(pastedLine, { message: "Loaded pasted position." })
    })
    window.addEventListener("keydown", (event) => {
      if (
        handleShortcutHelpShortcut(event)
        || handleCoordKeydown(event)
        || handleMoveNumberShortcut(event)
        || handleBoardOrientationShortcut(event)
        || handleSwapShortcut(event)
        || handlePassShortcut(event)
        || handleEditHistoryShortcut(event)
        || handleBranchingNavigationKeydown(event)
      ) {
        return
      }
      handleStandardKeydown(event, {
        goPrevious,
        goNext,
        goFirst,
        goLast,
        canDelete: canDeleteFromCursor,
        deleteFromCursor,
      })
    })
    window.addEventListener("keyup", handleCoordKeyup)
    window.addEventListener("blur", clearCoordOverlay)

    loadHash()

    return {
      getPosition,
      playLineFromCursor,
      refreshBoard,
      resetBoard,
      subscribePosition,
    }
  }

  window.HexBoardEditor = {
    applyBranchOutline,
    branchChildrenByKey,
    collectBoardEditorElements,
    createBranchingBoardEditor,
  }
})()
