(() => {
  const HEX_SIZE = 24
  const VIEW_PADDING = 34
  const COORD_VIEW_PADDING = 8

  const {
    createSvgTools,
    decodeLocationHash,
    replaceHash,
    renderExternalLink,
    renderMoveList: renderSharedMoveList,
    rgbText,
    stoneTextColor,
    turnRgbaText,
    THEME,
  } = window.HexStudyUI
  const {
    BLUE_RGB,
    GRID_EDGE,
    OFF_WHITE_RGB,
    RED_RGB,
  } = THEME
  const {
    alphaLabel,
    compactCursorHash,
    compactCursorText,
    formatCell,
    formatLine,
    lookupLineToDisplayLine,
    materializeBoardState,
    normalizeLine,
    normalizeMoveToken,
    parseCell,
    parseCompactCursorHash,
    parseMoves,
    pointKey,
    transformMove,
    tryParseCell,
  } = globalThis.HexPosition

  function numberText(value) {
    return Number(value).toFixed(1)
  }

  function swapControlPoint(boardSize) {
    const size = Number(boardSize || 11)
    return {
      col: -2,
      row: Math.round(1 + ((2 * (size - 1)) / 3)),
    }
  }

  function lineDisplay(line, boardSize, options = {}) {
    const size = Number(boardSize || 11)
    const text = compactCursorText({
      boardSize: size,
      line,
      defaultBoardSize: size,
      ...options,
    })
    return text || String(size)
  }

  function setHashFromLine(line, {
    boardSize,
    defaultBoardSize = 11,
    futureLines = [],
    keepEmptyLineComma = false,
  }) {
    const hash = compactCursorHash({
      boardSize,
      line,
      futureLines,
      defaultBoardSize,
      keepEmptyLineComma,
    })
    replaceHash(hash)
  }

  function clearHash() {
    replaceHash("")
  }

  function parseHashState({ availableBoardSizes, defaultBoardSize = 11 }) {
    const hashText = decodeLocationHash()
    if (hashText === null) {
      return { boardSize: null, line: "", fullLine: "", cursor: 0, valid: false }
    }
    return parseCompactCursorHash(hashText, {
      defaultBoardSize,
      isBoardSizeSupported: (boardSize) => availableBoardSizes.includes(boardSize),
      normalizeLineFn: normalizeLine,
    })
  }

  function renderHexWorldLink(container, url, text = "View in HexWorld") {
    container.replaceChildren()
    renderExternalLink(container, url, text)
  }

  function renderLineMoveList({
    container,
    currentLine,
    futureTailLines,
    setCursorLine,
  }) {
    const currentMoves = parseMoves(currentLine)
    const currentMoveCount = currentMoves.length
    const parts = [
      ...currentMoves.map((move, index) => ({
        text: move,
        isFuture: false,
        line: formatLine(currentMoves.slice(0, index + 1)),
      })),
      ...futureTailLines().map((line) => {
        const moves = parseMoves(line)
        return {
          text: moves[moves.length - 1] || "",
          isFuture: true,
          line,
        }
      }),
    ]
    renderSharedMoveList({
      container,
      parts,
      currentMoveCount,
      activateLine: (line) => {
        setCursorLine(line)
      },
    })
  }

  function createBoardSvg(board) {
    const tools = createSvgTools({
      board,
      hexSize: HEX_SIZE,
      defaultFill: rgbText(OFF_WHITE_RGB),
      defaultStroke: GRID_EDGE,
      defaultStrokeWidth: "0.85",
    })

    function setupViewBox(boardSize, extraPoints = null) {
      const boardPoints = []
      for (let row = 1; row <= boardSize; row += 1) {
        for (let col = 1; col <= boardSize; col += 1) {
          boardPoints.push({ col, row })
        }
      }
      const coordPoints = []
      for (let row = 1; row <= boardSize; row += 1) {
        coordPoints.push({ col: 0, row })
      }
      for (let col = 1; col <= boardSize; col += 1) {
        coordPoints.push({ col, row: 0 })
      }
      tools.setViewBoxFromPoints({
        boardPoints,
        coordPoints,
        extraPoints,
        viewPadding: VIEW_PADDING,
        coordViewPadding: COORD_VIEW_PADDING,
      })
    }

    function renderFrame(boardSize) {
      const borderRed = rgbText(RED_RGB)
      const borderBlue = rgbText(BLUE_RGB)
      const borderWidth = 4

      for (let col = 1; col <= boardSize; col += 1) {
        let cx
        let cy
        ;[cx, cy] = tools.pointToPixel(col, 1)
        let a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
        let b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
        let c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
        tools.appendLine(a[0], a[1], b[0], b[1], borderRed, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderRed, borderWidth)

        ;[cx, cy] = tools.pointToPixel(col, boardSize)
        a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
        b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
        c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
        tools.appendLine(a[0], a[1], b[0], b[1], borderRed, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderRed, borderWidth)
      }

      for (let row = 1; row <= boardSize; row += 1) {
        let cx
        let cy
        ;[cx, cy] = tools.pointToPixel(1, row)
        let a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 2)
        let b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 3)
        let c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 4)
        tools.appendLine(a[0], a[1], b[0], b[1], borderBlue, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderBlue, borderWidth)

        ;[cx, cy] = tools.pointToPixel(boardSize, row)
        a = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 5)
        b = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 0)
        c = tools.hexCorner(cx, cy, HEX_SIZE - 1.5, 1)
        tools.appendLine(a[0], a[1], b[0], b[1], borderBlue, borderWidth)
        tools.appendLine(b[0], b[1], c[0], c[1], borderBlue, borderWidth)
      }

      for (let col = 1; col <= boardSize; col += 1) {
        const [cx, cy] = tools.pointToPixel(col, 0)
        tools.appendText(cx, cy, alphaLabel(col), "coord-text")
      }
      for (let row = 1; row <= boardSize; row += 1) {
        const [cx, cy] = tools.pointToPixel(0, row)
        tools.appendText(cx, cy, String(row), "coord-text")
      }
    }

    return {
      ...tools,
      setupViewBox,
      renderFrame,
    }
  }

  function renderSideActionHex({
    boardSvg,
    point,
    toPlay,
    labelText = "",
    title = "",
    onClick = null,
    fill = rgbText(OFF_WHITE_RGB),
    stroke = GRID_EDGE,
    strokeWidth = "0.85",
    className = "board-hex side-action board-hex-face",
    primaryText = "",
    secondaryText = "",
    primaryTextClass = "cell-text",
    secondaryTextClass = "cell-stack-text",
    primaryTextFill = null,
    highlight = false,
  }) {
    if (!boardSvg || !point) {
      return null
    }
    const hoverColor = toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = turnRgbaText(toPlay, 0.12)
    const hitClasses = ["board-hover-hit"]
    if (typeof onClick === "function") {
      hitClasses.push("clickable", "hoverable")
    }
    const hoverHex = boardSvg.appendHex(point.col, point.row, {
      fill: "transparent",
      stroke: "none",
      className: hitClasses.join(" "),
      title,
      onClick,
      size: HEX_SIZE,
    })
    hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
    const hex = boardSvg.appendHex(point.col, point.row, {
      fill,
      stroke: highlight ? hoverColor : stroke,
      strokeWidth: highlight ? "1.6" : strokeWidth,
      className,
      title,
      onClick,
    })
    if (highlight) {
      hex.polygon.style.strokeWidth = "1.6"
    }
    hex.polygon.style.setProperty("--hover-outline", hoverColor)
    if (primaryText && secondaryText) {
      boardSvg.appendStackedText(hex.cx, hex.cy, primaryText, secondaryText, primaryTextClass, secondaryTextClass)
    } else if (primaryText || secondaryText) {
      boardSvg.appendText(hex.cx, hex.cy, primaryText || secondaryText, primaryTextClass, primaryTextFill)
    }
    if (labelText) {
      boardSvg.appendText(hex.cx, hex.cy - (HEX_SIZE * 1.28), labelText, "side-action-label")
    }
    return hex
  }

  function buildCandidateCellOverlays({
    boardSize,
    currentNode,
    displayRotation = 0,
    childLineForCandidate,
    buildOverlay,
    displayMoveForCandidate = null,
    displayLineForLookupLine = null,
    mirrorRootCandidates = true,
  }) {
    const node = currentNode || { line: "", candidates: [] }
    const overlayByKey = new Map()
    const displayMoveAtRotation = (move, rotation) => (
      typeof displayMoveForCandidate === "function"
        ? displayMoveForCandidate(move, { boardSize, displayRotation: rotation })
        : transformMove(move, boardSize, rotation)
    )
    const displayLineAtRotation = (line, rotation) => (
      typeof displayLineForLookupLine === "function"
        ? displayLineForLookupLine(line, { boardSize, displayRotation: rotation })
        : lookupLineToDisplayLine(line, { boardSize, displayRotation: rotation })
    )

    for (const candidate of node.candidates || []) {
      if (!candidate?.retained) {
        continue
      }
      const candidateAction = normalizeMoveToken(candidate.move)
      if (candidateAction === "pass" || candidateAction === "swap") {
        continue
      }
      const displayMove = displayMoveAtRotation(candidate.move, displayRotation)
      let point = tryParseCell(displayMove)
      if (!point) {
        const action = normalizeMoveToken(displayMove)
        if (action === "pass" || action === "swap") {
          continue
        }
        point = parseCell(displayMove)
      }
      const lookupChildLine = childLineForCandidate(node.line, candidate)
      const childLine = lookupChildLine
        ? displayLineAtRotation(lookupChildLine, displayRotation)
        : null
      const overlay = buildOverlay({
        node,
        candidate,
        displayMove,
        lookupChildLine,
        childLine,
        col: point.col,
        boardRow: point.row,
      })
      overlayByKey.set(pointKey(point.col, point.row), overlay)
      if (mirrorRootCandidates && !node.line) {
        const mirrorMove = displayMoveAtRotation(candidate.move, 180)
        if (mirrorMove !== displayMove) {
          const mirrorPoint = parseCell(mirrorMove)
          overlayByKey.set(pointKey(mirrorPoint.col, mirrorPoint.row), buildOverlay({
            node,
            candidate,
            displayMove: mirrorMove,
            lookupChildLine,
            childLine: lookupChildLine
              ? displayLineAtRotation(lookupChildLine, 180)
              : null,
            col: mirrorPoint.col,
            boardRow: mirrorPoint.row,
          }))
        }
      }
    }
    return overlayByKey
  }

  function renderMoveTreeBoard({
    boardSvg,
    boardSize,
    currentLine,
    currentNode,
    displayRotation = 0,
    extraViewboxPoints = null,
    childLineForCandidate,
    buildOverlay,
    candidateFill,
    overlayPrimaryText,
    overlaySecondaryText = null,
    onGoToLine,
    onActivateLastStone,
    displayMoveForCandidate = null,
    displayLineForLookupLine = null,
    mirrorRootCandidates = true,
    showCellCoords = false,
    showMoveNumbers = true,
    enableCellClicks = true,
    boardState = null,
    cellOverlays = null,
    isLegalCell = null,
  }) {
    boardSvg.clear()
    boardSvg.setupViewBox(boardSize, extraViewboxPoints)
    const positionBoard = boardState || materializeBoardState(parseMoves(currentLine), boardSize)
    const board = {
      ...positionBoard,
      cellsByKey: new Map(),
    }
    if (cellOverlays !== null && !(cellOverlays instanceof Map)) {
      throw new TypeError("Cell overlays must be a Map")
    }
    const overlayByKey = cellOverlays === null
      ? buildCandidateCellOverlays({
          boardSize,
          currentNode,
          displayRotation,
          childLineForCandidate,
          buildOverlay,
          displayMoveForCandidate,
          displayLineForLookupLine,
          mirrorRootCandidates,
        })
      : new Map(cellOverlays)

    const hoverColor = board.toPlay === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
    const hoverFill = turnRgbaText(board.toPlay, 0.12)

    for (let row = 1; row <= boardSize; row += 1) {
      for (let col = 1; col <= boardSize; col += 1) {
        const key = pointKey(col, row)
        const stone = board.occupied.get(key) || null
        const overlay = overlayByKey.get(key) || null
        const legal = typeof isLegalCell === "function"
          ? Boolean(isLegalCell(col, row, board))
          : Boolean(overlay?.childLine)
        let fill = rgbText(OFF_WHITE_RGB)
        let stroke = GRID_EDGE
        let strokeWidth = "0.85"
        let onClick = null
        let canActivate = false
        let className = "board-hex board-hex-face"

        if (overlay) {
          const nextFill = candidateFill(overlay)
          fill = typeof nextFill === "string" && nextFill ? nextFill : rgbText(OFF_WHITE_RGB)
          className = overlay.className || "board-hex board-hex-face"
          stroke = overlay.stroke ?? GRID_EDGE
          strokeWidth = overlay.strokeWidth ?? "0.85"
          if (overlay.childLine) {
            canActivate = true
            if (enableCellClicks) {
              onClick = () => {
                onGoToLine(overlay.childLine)
              }
            }
          }
        }

        if (stone) {
          fill = stone.color === "red" ? rgbText(RED_RGB) : rgbText(BLUE_RGB)
          stroke = "none"
          if (stone.isLast) {
            canActivate = true
            if (enableCellClicks) {
              onClick = () => {
                onActivateLastStone()
              }
            }
          }
        }

        const hitClasses = ["board-hover-hit"]
        if (canActivate || legal) {
          hitClasses.push("clickable")
        }
        if (legal && !stone) {
          hitClasses.push("hoverable")
        }
        const hoverHex = boardSvg.appendHex(col, row, {
          fill: "transparent",
          stroke: "none",
          className: hitClasses.join(" "),
          size: HEX_SIZE,
          title: formatCell(col, row),
          boardPoint: true,
          onClick,
        })
        hoverHex.polygon.style.setProperty("--hover-fill", hoverFill)
        const hex = boardSvg.appendHex(col, row, {
          fill,
          stroke,
          strokeWidth,
          className,
        })
        hex.polygon.style.setProperty("--hover-outline", hoverColor)
        board.cellsByKey.set(key, {
          col,
          row,
          hoverHex,
          hex,
          onClick,
          overlay,
          stone,
        })

        if (showCellCoords) {
          boardSvg.appendText(hex.cx, hex.cy, formatCell(col, row), "cell-text", stone ? stoneTextColor(stone) : null)
        } else if (overlay && !stone) {
          const primaryText = String(overlayPrimaryText(overlay) || "")
          const secondaryText = typeof overlaySecondaryText === "function"
            ? String(overlaySecondaryText(overlay) || "")
            : ""
          if (primaryText && secondaryText) {
            boardSvg.appendStackedText(hex.cx, hex.cy, primaryText, secondaryText)
          } else if (primaryText || secondaryText) {
            boardSvg.appendText(hex.cx, hex.cy, primaryText || secondaryText)
          }
        }

        if (!showCellCoords && stone && showMoveNumbers) {
          boardSvg.appendText(hex.cx, hex.cy, String(stone.ply), "cell-text", stoneTextColor(stone))
        } else if (!showCellCoords && stone?.isLast) {
          boardSvg.appendCircle(hex.cx, hex.cy, Math.max(2, HEX_SIZE * 0.14), {
            className: "last-move-dot",
            fill: rgbText(OFF_WHITE_RGB),
          })
        }
      }
    }

    boardSvg.renderFrame(boardSize)
    return board
  }

  window.HexMoveTree = {
    GRID_EDGE,
    clearHash,
    createBoardSvg,
    lineDisplay,
    numberText,
    parseHashState,
    renderHexWorldLink,
    renderLineMoveList,
    renderMoveTreeBoard,
    renderSideActionHex,
    setHashFromLine,
    swapControlPoint,
  }
})()
