(() => {
  const { formatCell, pointKey } = globalThis.HexPosition
  const { installHoldButton, renderAnalysisPv, shouldIgnoreGlobalKeydown } = globalThis.HexStudyUI
  const AWRN_STEPS = [0, 0.01, 0.02, 0.04, 0.1, 0.2, 0.5, 1, 2]
  const MAX_CACHED_POSITIONS = 128
  const SPEED_HALF_LIFE_MS = 1000
  const SPEED_DISPLAY_INTERVAL_MS = 500
  function shortcutHelpGroups(game) {
    return [
      {
        title: "Analysis",
        lines: [
          [{ keys: ["space"], text: "analysis" }, { keys: [","], text: "play best/PV" }],
          [
            { keys: ["t"], text: "priors" },
            ...(game === "yavalath" ? [{ keys: ["d"], text: "draw" }] : []),
            { keys: ["[", "]"], text: "set analysisWideRootNoise" },
          ],
          [{ keys: ["shift+c"], text: "clear cache" }, { keys: ["shift+b"], text: "batch" }],
        ],
      },
    ]
  }

  function compactCount(value) {
    return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : Math.round(value).toLocaleString()
  }

  function packCandidates(rows) {
    const moves = new Uint16Array(rows.length)
    const priors = new Float64Array(rows.length)
    const values = new Float64Array(rows.length * 4)
    // Even 42² candidates with 15 PV moves each fit in 16-bit offsets.
    const offsets = new Uint16Array(rows.length + 1)
    const variations = new Uint16Array(rows.reduce((length, row) => length + row.pv.length, 0))
    let offset = 0
    for (let i = 0; i < rows.length; i++) {
      const { move, visits, weight, prior, winrate, drawrate, pv } = rows[i]
      moves[i] = move
      priors[i] = prior
      const index = i * 4
      values[index] = visits
      values[index + 1] = weight
      values[index + 2] = winrate
      values[index + 3] = drawrate
      variations.set(pv, offset)
      offset += pv.length
      offsets[i + 1] = offset
    }
    return { moves, priors, values, offsets, variations }
  }

  function unpackCandidates({ moves, priors, values, offsets, variations }) {
    return Array.from(moves, (move, i) => ({
      move,
      visits: values[4 * i],
      weight: values[4 * i + 1],
      prior: priors[i],
      winrate: values[4 * i + 2],
      drawrate: values[4 * i + 3],
      pv: Array.from(variations.subarray(offsets[i], offsets[i + 1])),
    }))
  }

  function overlayText(row, mode) {
    if (!row) return { primary: "", secondary: "" }
    const value = mode === "prior" ? row.prior : mode === "draw" ? row.drawrate : row.winrate
    return {
      primary: Number.isFinite(value) ? (value * 100).toFixed(1) : "",
      secondary: mode === "prior" || !row.visits ? "" : compactCount(row.visits),
    }
  }

  function create(editor, model = null) {
    const toggle = document.getElementById("analysis-toggle")
    const batchButton = document.getElementById("analysis-batch")
    const modelChoice = document.getElementById("analysis-model")
    const selectedModel = () => modelChoice?.value || model
    const status = document.getElementById("analysis-status")
    const summary = document.getElementById("analysis-summary")
    const runtime = document.getElementById("analysis-runtime")
    const noise = document.getElementById("analysis-noise")
    const settings = document.getElementById("analysis-settings")
    const noiseDown = document.getElementById("analysis-noise-down")
    const noiseUp = document.getElementById("analysis-noise-up")
    const clearCacheButton = document.getElementById("analysis-clear-cache")
    const priorsButton = document.getElementById("analysis-priors")
    const drawButton = document.getElementById("analysis-draw")
    const overlayButtons = new Map([["prior", priorsButton], ...(drawButton ? [["draw", drawButton]] : [])])
    const overlayKeys = { t: "prior", ...(drawButton ? { d: "draw" } : {}) }
    const board = document.getElementById("board")
    const hoverControls = window.matchMedia("(hover: hover) and (pointer: fine)")
    const memoryLimit = () => (hoverControls.matches ? Infinity : 256 * 1024 * 1024)
    let overlays = new Map()
    let worker = null
    let position = editor.getPosition()
    let requestId = 0
    let ready = false
    let loading = false
    let active = false
    let displayedVisits = 0
    let searchSpeed = null
    let displayedSpeed = null
    let speedTimer = null
    let speedSample = null
    let hasStarted = false
    let finished = false
    let terminal = false
    let bestMove = null
    let overlayMode = "winrate"
    let pinnedMode = "winrate"
    const heldModes = []
    let wideRootNoise = 0.04
    let hoveredMove = null
    let frozenPv = null
    let boardOrientation = null
    let variation = null
    let batchRun = null
    let selectingBatchPosition = false
    const displayCache = new Map()
    const graph = createEvalGraph(editor)
    function postRequest(type, fields = {}) {
      worker?.postMessage({ type, id: ++requestId, model: selectedModel(), ...fields })
    }
    const view = {
      decorations: () => ({ analysisByKey: overlays }),
      overlayMode: () => overlayMode,
      renderPv: (boardSvg, toPlay, display) => {
        if (boardOrientation !== null && boardOrientation !== display.boardOrientation) {
          hoveredMove = null
          frozenPv = null
        }
        boardOrientation = display.boardOrientation
        if (!frozenPv) return
        const move = frozenPv[0]
        const firstMove = overlays.get(
          pointKey((move % position.boardSize) + 1, Math.floor(move / position.boardSize) + 1),
        )
        renderAnalysisPv(
          boardSvg,
          frozenPv,
          toPlay,
          position.boardSize,
          display.showCoords ? formatCell : null,
          overlayText(firstMove, overlayMode),
        )
      },
    }

    function setHoveredMove(move) {
      if (hoveredMove === move) return
      const previousPv = frozenPv
      hoveredMove = move
      const col = move === null ? null : (move % position.boardSize) + 1
      const row = move === null ? null : Math.floor(move / position.boardSize) + 1
      const pv = move === null ? null : overlays.get(pointKey(col, row))?.pv
      frozenPv = pv?.length > 1 ? pv : null
      if (frozenPv !== previousPv) editor.refreshBoard({ previewOnly: true })
    }

    function syncOverlayView() {
      for (const [mode, button] of overlayButtons) button.setAttribute("aria-pressed", String(pinnedMode === mode))
      const next = heldModes.at(-1) || pinnedMode
      if (overlayMode === next) return
      overlayMode = next
      editor.refreshBoard()
    }

    function holdOverlay(mode, held) {
      const index = heldModes.indexOf(mode)
      if (held && index < 0) heldModes.push(mode)
      else if (!held && index >= 0) heldModes.splice(index, 1)
      else return
      syncOverlayView()
    }

    function syncControls() {
      if (!active || batchRun) {
        speedSample = null
        clearTimeout(speedTimer)
        speedTimer = null
      }
      toggle.textContent = active ? "Stop" : "Start"
      toggle.disabled = !worker || (!active && finished)
      batchButton.disabled = !worker || !variation || variation.points.length < 2
      batchButton.textContent = batchRun ? "Stop" : "Batch"
      batchButton.setAttribute(
        "aria-label",
        batchRun ? "Stop batch analysis" : "Batch analysis of current sequence of moves",
      )
      toggle.setAttribute("aria-pressed", String(active))
      noise.textContent = `AWRN ${wideRootNoise}`
      const loadingModel = active && !ready
      summary.hidden = !hasStarted || loadingModel || Boolean(batchRun)
      noiseDown.disabled = !worker || wideRootNoise === AWRN_STEPS[0]
      noiseUp.disabled = !worker || wideRootNoise === AWRN_STEPS.at(-1)
      clearCacheButton.disabled = !worker
      for (const button of overlayButtons.values()) button.disabled = !worker
      runtime.hidden = !hasStarted || loadingModel || !runtime.textContent
    }

    function showProgress() {
      const speed = displayedSpeed === null ? "–" : compactCount(displayedSpeed)
      const text = `${compactCount(displayedVisits)} visits · ${speed} visits/s`
      if (summary.textContent !== text) summary.textContent = text
    }

    function resetSpeed() {
      clearTimeout(speedTimer)
      speedTimer = null
      speedSample = null
      searchSpeed = null
      displayedSpeed = null
      showProgress()
    }

    function refreshSpeed() {
      displayedSpeed = searchSpeed
      showProgress()
      speedTimer = setTimeout(() => {
        speedTimer = null
        if (searchSpeed !== displayedSpeed) refreshSpeed()
      }, SPEED_DISPLAY_INTERVAL_MS)
    }

    function updateProgress(visits, reportedAt, early) {
      showProgress()
      if (!active || batchRun || early) return
      if (!speedSample) {
        speedSample = { visits, time: reportedAt }
        return
      }
      const elapsed = reportedAt - speedSample.time
      if (elapsed <= 0) return
      const rate = (Math.max(0, visits - speedSample.visits) * 1000) / elapsed
      const blend = 1 - 2 ** (-elapsed / SPEED_HALF_LIFE_MS)
      searchSpeed = searchSpeed === null ? rate : searchSpeed + blend * (rate - searchSpeed)
      speedSample = { visits, time: reportedAt }
      if (speedTimer === null) refreshSpeed()
    }

    function clearResults() {
      overlays = new Map()
      frozenPv = null
      displayedVisits = 0
      resetSpeed()
      bestMove = null
      finished = false
      terminal = false
    }

    function showCandidates(candidates, snapshot) {
      const topVisits = Math.max(1, ...candidates.map((row) => row.visits))
      const topMove = candidates[0]?.move ?? null
      const topPrior = candidates.reduce((best, row) => (!best || row.prior > best.prior ? row : best), null)?.move ?? null
      overlays = new Map()
      for (const row of candidates) {
        const col = (row.move % position.boardSize) + 1
        const r = Math.floor(row.move / position.boardSize) + 1
        overlays.set(pointKey(col, r), {
          ...row,
          topVisits,
          isTopMove: row.move === topMove,
          isTopPrior: row.move === topPrior,
        })
      }
      if (hoveredMove !== null && !frozenPv) {
        const pv = candidates.find((row) => row.move === hoveredMove)?.pv
        if (pv?.length > 1) frozenPv = pv
      }
      bestMove = snapshot.bestMove
      displayedVisits = snapshot.visits
    }

    function restoreDisplaySnapshot() {
      // Batch previews and terminal positions can display historical reports.
      const cached = displayCache.get(position.key)
      if (!cached || (!selectingBatchPosition && !cached.terminal)) {
        clearResults()
        return
      }
      displayCache.delete(position.key)
      displayCache.set(position.key, cached)
      showCandidates(unpackCandidates(cached.candidates), cached)
      resetSpeed()
      terminal = cached.terminal
      finished = terminal
    }

    function rememberResult(result) {
      const previous = displayCache.get(position.key)
      const best = result.candidates[0]
      const cached = previous?.visits > result.visits ? previous : {
        visits: result.visits,
        winrate: best?.winrate ?? result.winrate,
        drawrate: best?.drawrate ?? result.drawrate,
        terminal: result.terminal,
        bestMove: result.bestMove,
        candidates: packCandidates(result.candidates),
      }
      graph.record(position, cached, cached.candidates.moves.length > 0)
      displayCache.delete(position.key)
      displayCache.set(position.key, cached)
      if (displayCache.size > MAX_CACHED_POSITIONS) displayCache.delete(displayCache.keys().next().value)
      showCandidates(result.candidates, result)
      terminal = result.terminal
      finished = result.finished
    }

    function fail(message, fatal = false) {
      active = false
      loading = false
      batchRun = null
      if (fatal) {
        ready = false
        runtime.textContent = ""
      }
      clearResults()
      status.textContent = message
      editor.refreshBoard()
      syncControls()
    }

    function start(refresh = true) {
      resetSpeed()
      batchRun = null
      active = true
      hasStarted = true
      if (!loading) status.textContent = ""
      postRequest("start", { position, wideRootNoise, memoryLimit: memoryLimit() })
      if (refresh) editor.refreshBoard()
      syncControls()
    }

    function selectBatchPosition(ply) {
      selectingBatchPosition = true
      try { batchRun.variation.points[ply].select() } finally { selectingBatchPosition = false }
    }

    function startBatch() {
      if (!worker || batchRun || variation.points.length < 2) return
      const startPly = variation.cursor === variation.points.length - 1 ? 0 : variation.cursor
      const plies = graph.missingBatchPlies(variation, startPly)
      batchRun = { variation, startPly, total: variation.points.length - startPly }
      hasStarted = true
      if (!plies.length) {
        selectBatchPosition(variation.points.length - 1)
        finishBatch()
        syncControls()
        return
      }
      active = true
      selectBatchPosition(plies[0])
      if (!loading) status.textContent = ready ? batchProgress(plies[0]) : ""
      postRequest("batch", { position, plies, moves: batchRun.variation.points.slice(1).map((point) => point.move) })
      syncControls()
    }

    function batchProgress(nextPly) {
      return `Batch analysis… ${nextPly - batchRun.startPly}/${batchRun.total}`
    }

    function finishBatch() {
      batchRun = null
      active = false
      loading = false
      restoreWorkerResults(true)
      status.textContent = ""
    }

    function stopAnalysis(restore = true) {
      const wasBatch = Boolean(batchRun)
      active = false
      loading = false
      batchRun = null
      postRequest("stop")
      if (restore && (wasBatch || overlays.size === 0)) restoreWorkerResults(wasBatch)
      status.textContent = ""
      syncControls()
    }

    function restoreWorkerResults(refresh = false) {
      if (refresh) {
        restoreDisplaySnapshot()
        editor.refreshBoard()
      }
      postRequest("restore", { position, wideRootNoise, memoryLimit: memoryLimit() })
    }

    function toggleBatch() {
      if (batchRun) stopAnalysis()
      else startBatch()
    }

    function moveText(move) {
      return formatCell((move % position.boardSize) + 1, Math.floor(move / position.boardSize) + 1)
    }

    editor.subscribePosition((next, { freshBoard = false } = {}) => {
      if (batchRun && !selectingBatchPosition) stopAnalysis(false)
      const wasFinished = finished
      hoveredMove = null
      frozenPv = null
      position = next
      if (selectingBatchPosition) {
        restoreDisplaySnapshot()
        return
      }
      if (freshBoard) {
        displayCache.clear()
        graph.clear()
        clearResults()
        postRequest("stop", { clear: true })
      } else restoreDisplaySnapshot()
      if (active) start(false)
      else {
        if (!freshBoard) restoreWorkerResults()
        if (ready || wasFinished) status.textContent = ""
        syncControls()
      }
    })
    editor.subscribeVariation((next) => {
      if (batchRun && !selectingBatchPosition && next !== batchRun.variation) stopAnalysis()
      variation = next
      syncControls()
    })

    function toggleAnalysis() {
      if (active) { stopAnalysis() } else start()
    }

    function clearAnalysisCache() {
      if (!worker) return
      if (batchRun) stopAnalysis(false)
      postRequest("stop", { clear: true })
      displayCache.clear()
      graph.clear()
      clearResults()
      if (active) start()
      else {
        status.textContent = ""
        editor.refreshBoard()
        syncControls()
      }
    }

    function stepWideRootNoise(direction) {
      const index = AWRN_STEPS.indexOf(wideRootNoise)
      const next = AWRN_STEPS[Math.max(0, Math.min(AWRN_STEPS.length - 1, index + direction))]
      if (next === wideRootNoise) return false
      wideRootNoise = next
      if (active && !batchRun) start()
      else {
        if (!batchRun) status.textContent = ""
        syncControls()
      }
      return true
    }

    toggle.addEventListener("click", toggleAnalysis)
    batchButton.addEventListener("click", toggleBatch)
    for (const eventName of ["contextmenu", "selectstart"]) {
      settings.addEventListener(eventName, (event) => event.preventDefault(), { capture: true })
    }
    installHoldButton(
      noiseDown,
      () => stepWideRootNoise(-1),
      () => noiseDown.disabled,
    )
    installHoldButton(
      noiseUp,
      () => stepWideRootNoise(1),
      () => noiseUp.disabled,
    )
    clearCacheButton.addEventListener("click", clearAnalysisCache)
    for (const [mode, button] of overlayButtons) {
      button.addEventListener("click", () => {
        pinnedMode = pinnedMode === mode ? "winrate" : mode
        syncOverlayView()
      })
    }
    board.addEventListener("pointermove", (event) => {
      if (!hoverControls.matches || event.pointerType === "touch") return
      const cell = event.target instanceof Element ? event.target.closest("[data-board-point='1']") : null
      if (!cell || !board.contains(cell)) {
        setHoveredMove(null)
        return
      }
      const col = Number(cell.getAttribute("data-q"))
      const row = Number(cell.getAttribute("data-r"))
      const move = (row - 1) * position.boardSize + col - 1
      setHoveredMove(move)
    })
    board.addEventListener("pointerleave", () => setHoveredMove(null))
    board.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "touch") setHoveredMove(null)
    })
    board.addEventListener("pointercancel", () => setHoveredMove(null))
    hoverControls.addEventListener("change", () => {
      if (!hoverControls.matches) setHoveredMove(null)
      if (!worker || (!active && !ready)) return
      finished = terminal
      if (batchRun) return
      if (active) start(false)
      else {
        status.textContent = ""
        restoreWorkerResults()
        syncControls()
      }
    })
    modelChoice?.addEventListener("change", () => {
      const wasActive = active && !batchRun
      batchRun = null
      active = false
      ready = false
      loading = false
      hasStarted = false
      runtime.textContent = ""
      displayCache.clear()
      graph.clear()
      clearResults()
      postRequest("stop", { clear: true })
      status.textContent = ""
      if (wasActive) start()
      else {
        editor.refreshBoard()
        syncControls()
      }
    })

    // Handle Space before focused controls can activate themselves.
    window.addEventListener(
      "keydown",
      (event) => {
        if (event.key !== " " || shouldIgnoreGlobalKeydown(event)) return
        event.preventDefault()
        event.stopPropagation()
        if (!event.repeat && !toggle.disabled) toggleAnalysis()
      },
      true,
    )

    window.addEventListener("keydown", (event) => {
      if (shouldIgnoreGlobalKeydown(event)) return
      const clearCache = event.shiftKey && event.key.toLowerCase() === "c"
      const batchKey = event.shiftKey && event.key.toLowerCase() === "b"
      const noiseDirection = event.key === "[" ? -1 : event.key === "]" ? 1 : 0
      const heldMode = overlayKeys[event.key.toLowerCase()]
      if (!clearCache && !batchKey && !noiseDirection && (event.shiftKey || (event.key !== "," && !heldMode))) return
      event.preventDefault()
      if (batchKey) {
        if (!event.repeat) toggleBatch()
        return
      }
      if (clearCache) {
        if (!event.repeat) clearAnalysisCache()
        return
      }
      if (noiseDirection) {
        stepWideRootNoise(noiseDirection)
        return
      }
      if (heldMode) {
        holdOverlay(heldMode, true)
        return
      }
      if (event.repeat) return
      if (bestMove !== null && !terminal) {
        const moves = frozenPv || [bestMove]
        editor.playLineFromCursor(`${position.line}${moves.map(moveText).join("")}`)
      }
    })
    window.addEventListener("keyup", (event) => {
      const mode = overlayKeys[event.key.toLowerCase()]
      if (mode) holdOverlay(mode, false)
    })
    window.addEventListener("blur", () => {
      heldModes.length = 0
      syncOverlayView()
      setHoveredMove(null)
    })

    if (!navigator.gpu) {
      fail("Analysis requires WebGPU. Try a browser and device with WebGPU support.", true)
      return view
    }
    try {
      worker = new Worker(new URL("./webgpu/analysis_worker.js", document.baseURI), { type: "module" })
      worker.onmessage = ({ data }) => {
        if (data.model && data.model !== selectedModel()) return
        if (data.id !== requestId) return
        if (!active && data.type === "result") return
        if (data.type === "loading") {
          if (!active) return
          const percent = data.percent === null ? "" : ` ${data.percent}%`
          const text = data.percent === 100 ? "Preparing engine…" : `Loading model…${percent}`
          if (status.textContent !== text) status.textContent = text
          if (loading) return
          ready = false
          loading = true
          runtime.textContent = ""
          resetSpeed()
        } else if (data.type === "ready") {
          ready = true
          loading = false
          runtime.textContent = `GPU: ${data.runtime}`
          runtime.title = data.runtimeHelp
          if (active) status.textContent = batchRun ? batchProgress(position.moves.length) : ""
        } else if (data.type === "error") {
          fail(data.message, data.fatal)
        } else if (data.type === "result" || data.type === "cached-result") {
          rememberResult(data)
          updateProgress(data.visits, data.reportedAt, data.early)
          editor.refreshBoard()
        } else if (data.type === "cache-miss") {
          const hadDecorations = overlays.size > 0 || frozenPv !== null
          restoreDisplaySnapshot()
          if (hadDecorations || overlays.size > 0) editor.refreshBoard()
        } else if (data.type === "batch-result" && batchRun) {
          graph.recordBatch(batchRun.variation, data.results)
          const ply = Math.min(data.nextPly, batchRun.variation.points.length - 1)
          selectBatchPosition(ply)
          status.textContent = batchProgress(data.nextPly)
        } else if (data.type === "batch-finished" && batchRun) {
          finishBatch()
        } else if (data.type === "finished") {
          active = false
          loading = false
          finished = true
          status.textContent = data.reason
        }
        syncControls()
      }
      worker.onerror = (event) => {
        worker.terminate()
        worker = null
        fail(`Analysis could not start. Reload the page to retry. ${event.message || ""}`, true)
      }
      status.textContent = ""
    } catch (error) {
      worker?.terminate()
      worker = null
      fail(error.message || "Analysis is unavailable in this browser.", true)
    }
    syncControls()
    window.addEventListener("pagehide", () => {
      if (active) { stopAnalysis() }
    })
    return view
  }

  function createEvalGraph(editor) {
    const BATCH_VALUE = 1
    const SEARCH_VALUE = 2
    const { rgbText, THEME } = window.HexStudyUI
    const MAX_EVALUATIONS = 4096
    const SVG_NS = "http://www.w3.org/2000/svg"

    const svg = document.getElementById("analysis-graph")
    const card = document.getElementById("analysis-graph-card")
    const label = document.createElement("p")
    const preview = document.createElementNS(SVG_NS, "svg")
    card.append(label, preview)
    const evaluations = new Map()
    let variation = { cursor: 0, points: [] }
    let width = 320
    let hoverPly = null
    let previewKey = null
    let boardOrientation = "flat"
    let pointer = null
    const height = 164, left = 34, right = 12, top = 14, bottom = 24
    const red = rgbText(THEME.RED_RGB)
    const yavalath = editor.getPosition().game === "yavalath"

    const xFor = (ply) => left + ((ply - 1) / (Math.max(10, variation.points.length - 1) - 1)) * (width - left - right)
    const yFor = (value) => top + (1 - value) * (height - top - bottom)

    function evaluationFor(point) {
      if (!point.ply) return undefined
      if (point.ply === 1 && variation.cursor >= 2 && variation.points[2]?.move === "swap") return undefined
      return evaluations.get(point.key)
    }

    function hideCard() {
      hoverPly = null
      pointer = null
      card.hidden = true
    }

    function drawPreview(point) {
      const { game, boardSize, board } = point.preview()
      window.BoardGraphPreview.draw(preview, { game, boardSize, board, orientation: boardOrientation })
    }

    function showCard() {
      const point = variation.points[hoverPly]
      if (!point || !pointer) {
        card.hidden = true
        return
      }
      const value = evaluationFor(point)
      const move = point.ply ? `${point.ply}. ${point.move}` : "Start"
      if (value) {
        label.textContent = `${move} · Red${yavalath ? " score" : ""} ${(100 * value.redWinrate).toFixed(1)}%`
        if (yavalath && Number.isFinite(value.drawrate))
          label.textContent += ` · Draw ${(100 * value.drawrate).toFixed(1)}%`
      } else label.textContent = `${move} · Not analyzed`
      if (previewKey !== point.key) {
        drawPreview(point)
        previewKey = point.key
      }
      window.BoardGraphPreview.place(card, pointer.x, pointer.y)
    }

    let renderFrame = null
    function scheduleRender() {
      if (renderFrame !== null) return
      renderFrame = requestAnimationFrame(() => {
        renderFrame = null
        render()
      })
    }

    function render() {
      width = Math.max(100, svg.getBoundingClientRect().width || 320)
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`)
      const content = document.createDocumentFragment()
      function append(tag, attributes, text = "") {
        const node = document.createElementNS(SVG_NS, tag)
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value))
        node.textContent = text
        content.appendChild(node)
      }
      for (const value of [0, 0.5, 1]) {
        const y = yFor(value)
        append("line", { x1: left, x2: width - right, y1: y, y2: y, class: "analysis-graph-grid" })
        append(
          "text",
          { x: left - 6, y: y + 4, "text-anchor": "end", class: "eval-graph-axis-label" },
          `${value * 100}`,
        )
      }
      const maxPly = Math.max(10, variation.points.length - 1)
      const step = Math.max(1, Math.ceil(maxPly / Math.max(2, Math.floor(width / 65))))
      for (let ply = 1; ply <= maxPly; ply += step) {
        append("text", { x: xFor(ply), y: height - 5, "text-anchor": "middle", class: "eval-graph-axis-label" }, ply)
      }
      if (variation.cursor > 0) {
        const cursorX = xFor(variation.cursor)
        append("line", { x1: cursorX, x2: cursorX, y1: top, y2: height - bottom, class: "analysis-graph-cursor" })
      }
      const series = [{ field: "redWinrate", color: red, className: "analysis-graph-score" }]
      if (yavalath) series.push({ field: "drawrate", color: rgbText(THEME.DRAW_RGB), className: "analysis-graph-draw" })
      for (const { field, color, className } of series) {
        let segment = []
        let dots = "", current = null
        function flush() {
          if (segment.length > 1)
            append("polyline", {
              points: segment.join(" "),
              fill: "none",
              stroke: color,
              "stroke-width": 2,
              class: className,
            })
          segment = []
        }
        for (const point of variation.points) {
          const value = evaluationFor(point)?.[field]
          if (!Number.isFinite(value)) {
            flush()
            continue
          }
          const x = xFor(point.ply), y = yFor(value)
          segment.push(`${x},${y}`)
          if (point.ply === variation.cursor) current = { x, y }
          else dots += `M${x - 2},${y}a2,2 0 1 0 4,0a2,2 0 1 0 -4,0z`
        }
        flush()
        if (dots) append("path", { d: dots, fill: color, class: className })
        if (current) append("circle", { cx: current.x, cy: current.y, r: 4, fill: color, class: className })
      }
      svg.replaceChildren(content)
      showCard()
    }

    function plyAt(event) {
      const rect = svg.getBoundingClientRect()
      const x = ((event.clientX - rect.left) * width) / rect.width
      const ply =
        1 + Math.round(((x - left) / (width - left - right)) * (Math.max(10, variation.points.length - 1) - 1))
      return Math.max(1, Math.min(variation.points.length - 1, ply))
    }
    svg.addEventListener("pointermove", (event) => {
      if (event.pointerType === "touch") {
        hideCard()
        return
      }
      hoverPly = plyAt(event)
      pointer = { x: event.clientX, y: event.clientY }
      showCard()
    })
    svg.addEventListener("pointerleave", hideCard)
    svg.addEventListener("pointercancel", hideCard)
    svg.addEventListener("click", (event) => {
      const point = variation.points[plyAt(event)]
      hideCard()
      point?.select()
    })
    svg.addEventListener("keydown", (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
      const ply = {
        ArrowLeft: variation.cursor - 1,
        ArrowRight: variation.cursor + 1,
        Home: 0,
        End: variation.points.length - 1,
      }[event.key]
      if (ply === undefined) return
      event.preventDefault()
      event.stopPropagation()
      hideCard()
      variation.points[Math.max(0, Math.min(variation.points.length - 1, ply))]?.select()
    })
    window.addEventListener("blur", hideCard)
    window.addEventListener("scroll", hideCard, true)
    new ResizeObserver(scheduleRender).observe(svg)
    editor.subscribeVariation((next) => {
      variation = next
      hideCard()
      scheduleRender()
    })
    editor.subscribeBoardOrientation((next) => {
      boardOrientation = next
      previewKey = null
      if (!card.hidden) showCard()
    })
    function remember(key, value) {
      const previous = evaluations.get(key)
      evaluations.delete(key)
      evaluations.set(key, previous && previous.priority > value.priority ? previous : value)
      if (evaluations.size > MAX_EVALUATIONS) evaluations.delete(evaluations.keys().next().value)
    }
    return {
      missingBatchPlies(line, startPly) {
        const plies = []
        // Reuse a range that fits in the cache so incoming results cannot evict skipped plies.
        const canReuse = line.points.length - startPly <= MAX_EVALUATIONS
        for (let ply = startPly; ply < line.points.length; ply++) {
          const { key } = line.points[ply]
          const value = canReuse ? evaluations.get(key) : null
          // A search root without visited moves may still have a symmetry-averaged value.
          if ((value?.priority ?? 0) < BATCH_VALUE) plies.push(ply)
          else remember(key, value)
        }
        return plies
      },
      recordBatch(line, results) {
        for (const row of results) {
          const key = line.points[row.ply].key
          remember(key, { ...row, priority: BATCH_VALUE })
        }
        scheduleRender()
      },
      record(position, { winrate, drawrate, terminal }, hasReplies) {
        if (!Number.isFinite(winrate)) return
        remember(position.key, {
          priority: hasReplies ? SEARCH_VALUE : terminal ? BATCH_VALUE : 0,
          redWinrate: position.toPlay === "red" ? winrate : 1 - winrate,
          drawrate: position.game === "yavalath" ? drawrate : undefined,
        })
        scheduleRender()
      },
      clear() {
        evaluations.clear()
        previewKey = null
        scheduleRender()
      },
    }
  }

  window.BoardAnalysis = { create, overlayText, shortcutHelpGroups }
})()
