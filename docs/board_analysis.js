(() => {
  const { formatCell, pointKey } = globalThis.HexPosition
  const { installHoldButton, renderAnalysisPv } = globalThis.HexStudyUI
  const AWRN_STEPS = [0, 0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1, 2]
  const MAX_CACHED_POSITIONS = 128
  const shortcutHelpGroups = [{
    title: "Analysis",
    lines: [
      [{ keys: ["enter", "space"], text: "start/stop" }, { keys: [","], text: "play best/PV" }],
      [{ keys: ["t"], text: "priors" }, { keys: ["[", "]"], text: "AWRN" }],
      [{ keys: ["shift+c"], text: "clear cache" }, { keys: ["shift+b"], text: "analyze moves" }],
    ],
  }]

  function compactCount(value) {
    return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : Math.round(value).toLocaleString()
  }

  function overlayText(row, showPriors) {
    const value = showPriors ? row.prior : row.winrate
    return {
      primary: Number.isFinite(value) ? (value * 100).toFixed(1) : "",
      secondary: showPriors || !row.visits ? "" : compactCount(row.visits),
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
    const board = document.getElementById("board")
    const mobileControls = window.matchMedia("(max-width: 620px), (hover: none) and (pointer: coarse)")
    const hoverControls = window.matchMedia("(hover: hover) and (pointer: fine)")
    let overlays = new Map()
    let worker = null
    let position = editor.getPosition()
    let requestId = 0
    let ready = false
    let active = false
    let hasStarted = false
    let finished = false
    let terminal = false
    let bestMove = null
    let showPriors = false
    let priorHeld = false
    let priorPinned = false
    let wideRootNoise = 0.04
    let hoveredMove = null
    let frozenPv = null
    let boardOrientation = null
    let variation = null
    let batchRun = null
    const cache = new Map()
    const graph = window.BoardEvalGraph.create(editor)
    const view = {
      decorations: () => ({ analysisByKey: overlays }),
      showPriors: () => showPriors,
      renderPv: (boardSvg, toPlay, display) => {
        if (boardOrientation !== null && boardOrientation !== display.boardOrientation) {
          hoveredMove = null
          frozenPv = null
        }
        boardOrientation = display.boardOrientation
        if (!frozenPv) return
        const move = frozenPv[0]
        const firstMove = overlays.get(pointKey(move % position.boardSize + 1, Math.floor(move / position.boardSize) + 1))
        renderAnalysisPv(boardSvg, frozenPv, toPlay, position.boardSize,
          display.showCoords ? formatCell : null, overlayText(firstMove, showPriors))
      },
    }

    function setHoveredMove(move) {
      if (hoveredMove === move) return
      const previousPv = frozenPv
      hoveredMove = move
      const col = move === null ? null : move % position.boardSize + 1
      const row = move === null ? null : Math.floor(move / position.boardSize) + 1
      const pv = move === null ? null : overlays.get(pointKey(col, row))?.pv
      frozenPv = pv?.length > 1 ? pv : null
      if (frozenPv !== previousPv) editor.refreshBoard()
    }

    function syncPriorView() {
      if (!mobileControls.matches) priorPinned = false
      priorsButton.setAttribute("aria-pressed", String(priorPinned))
      const next = priorHeld || priorPinned
      if (showPriors === next) return
      showPriors = next
      editor.refreshBoard()
    }

    function clearHeldPriorView() {
      if (!priorHeld) return
      priorHeld = false
      syncPriorView()
    }

    function syncControls() {
      toggle.textContent = active ? "Stop" : "Start"
      toggle.disabled = !worker || (!active && finished)
      batchButton.disabled = !worker
      batchButton.textContent = batchRun ? "Stop" : "Analyze moves"
      batchButton.setAttribute("aria-label", batchRun ? "Stop move analysis" : "Analyze current sequence of moves")
      batchButton.setAttribute("aria-pressed", String(Boolean(batchRun)))
      toggle.setAttribute("aria-pressed", String(active))
      noise.textContent = `AWRN ${wideRootNoise}`
      const loadingModel = active && !ready
      settings.hidden = !hasStarted
      summary.hidden = !hasStarted || loadingModel || Boolean(batchRun)
      noiseDown.disabled = !worker || wideRootNoise === AWRN_STEPS[0]
      noiseUp.disabled = !worker || wideRootNoise === AWRN_STEPS.at(-1)
      clearCacheButton.disabled = !worker
      priorsButton.disabled = !worker
      runtime.hidden = !hasStarted || loadingModel || !runtime.textContent
    }

    function showSummary(visits, visitsPerSecond) {
      const speed = visitsPerSecond > 0 ? compactCount(visitsPerSecond) : "–"
      summary.textContent = `${compactCount(visits)} visits · ${speed} visits/s`
    }

    function clearResults() {
      overlays = new Map()
      frozenPv = null
      showSummary(0, 0)
      bestMove = null
      finished = false
      terminal = false
    }

    function showCandidates(candidates) {
      const topVisits = Math.max(1, ...candidates.map((row) => row.visits))
      const visited = candidates.filter((row) => row.visits > 0)
      const topMove = visited[0]?.move ?? null
      const topPrior = visited.reduce((best, row) => !best || row.prior > best.prior ? row : best, null)?.move ?? null
      overlays = new Map()
      for (const row of candidates) {
        const col = row.move % position.boardSize + 1
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
      bestMove = candidates[0]?.move ?? null
    }

    function restoreResults() {
      const cached = cache.get(position.key)
      if (!cached) { clearResults(); return }
      cache.delete(position.key)
      cache.set(position.key, cached)
      showCandidates(cached.candidates)
      showSummary(cached.visits, cached.visitsPerSecond)
      terminal = cached.terminal
      finished = cached.finished
    }

    function rememberResult(result) {
      graph.record(position, result)
      const previous = cache.get(position.key)
      const cached = previous && previous.visits > result.visits ? previous : result
      cache.delete(position.key)
      cache.set(position.key, cached)
      if (cache.size > MAX_CACHED_POSITIONS) cache.delete(cache.keys().next().value)
      showCandidates(cached.candidates)
      showSummary(cached.visits, cached.visitsPerSecond)
      terminal = cached.terminal
    }

    function fail(message, fatal = false) {
      active = false
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
      batchRun = null
      active = true
      hasStarted = true
      status.textContent = ready ? "" : "Loading model…"
      worker.postMessage({ type: "start", id: ++requestId, model: selectedModel(), position, wideRootNoise })
      if (refresh) editor.refreshBoard()
      syncControls()
    }

    function startBatch() {
      if (!worker || batchRun) return
      const startPly = variation.cursor === variation.points.length - 1 ? 0 : variation.cursor
      batchRun = { variation, total: variation.points.length - startPly }
      active = true
      hasStarted = true
      status.textContent = ready ? `Fast analysis… 0/${batchRun.total}` : "Loading model…"
      worker.postMessage({ type: "batch", id: ++requestId, model: selectedModel(), position,
        startPly, moves: variation.points.slice(1).map((point) => point.move) })
      syncControls()
    }

    function stopAnalysis() {
      active = false
      batchRun = null
      requestId++
      worker?.postMessage({ type: "stop" })
      status.textContent = ""
      syncControls()
    }

    function toggleBatch() {
      if (batchRun) stopAnalysis()
      else startBatch()
    }

    function moveText(move) {
      return formatCell(move % position.boardSize + 1, Math.floor(move / position.boardSize) + 1)
    }

    editor.subscribePosition((next, { freshBoard = false } = {}) => {
      if (batchRun) stopAnalysis()
      hoveredMove = null
      frozenPv = null
      position = next
      requestId++
      if (freshBoard) {
        cache.clear()
        graph.clear()
        clearResults()
        worker?.postMessage({ type: "stop", clear: true })
      } else restoreResults()
      if (active) start(false)
      else {
        if (!freshBoard) worker?.postMessage({ type: "stop" })
        if (ready) status.textContent = ""
        syncControls()
      }
    })
    editor.subscribeVariation((next) => {
      if (batchRun && next !== batchRun.variation) stopAnalysis()
      variation = next
    })

    function toggleAnalysis() {
      if (active) {
        stopAnalysis()
      } else start()
    }

    function clearAnalysisCache() {
      if (!worker) return
      if (batchRun) stopAnalysis()
      requestId++
      worker.postMessage({ type: "stop", clear: true })
      cache.clear()
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
    installHoldButton(noiseDown, () => stepWideRootNoise(-1), () => noiseDown.disabled)
    installHoldButton(noiseUp, () => stepWideRootNoise(1), () => noiseUp.disabled)
    clearCacheButton.addEventListener("click", clearAnalysisCache)
    priorsButton.addEventListener("click", () => {
      priorPinned = !priorPinned
      syncPriorView()
    })
    mobileControls.addEventListener("change", syncPriorView)
    board.addEventListener("pointermove", (event) => {
      if (!hoverControls.matches || event.pointerType === "touch") return
      const cell = event.target instanceof Element ? event.target.closest("[data-board-point='1']") : null
      if (!cell || !board.contains(cell)) { setHoveredMove(null); return }
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
    })
    modelChoice?.addEventListener("change", () => {
      const wasActive = active && !batchRun
      batchRun = null
      active = false
      requestId++
      ready = false
      hasStarted = false
      runtime.textContent = ""
      cache.clear()
      graph.clear()
      clearResults()
      worker?.postMessage({ type: "stop", clear: true })
      status.textContent = ""
      if (wasActive) start()
      else {
        editor.refreshBoard()
        syncControls()
      }
    })

    window.addEventListener("keydown", (event) => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return
      const clearCache = event.shiftKey && event.key.toLowerCase() === "c"
      const batchKey = event.shiftKey && event.key.toLowerCase() === "b"
      const noiseDirection = event.key === "[" ? -1 : event.key === "]" ? 1 : 0
      const toggleKey = event.key === " " || event.key === "Enter"
      if (!clearCache && !batchKey && !noiseDirection && (event.shiftKey || (!toggleKey && event.key !== "," && event.key.toLowerCase() !== "t"))) return
      const target = event.target
      if (target instanceof Element) {
        if (target.closest("input, textarea") || target.isContentEditable) return
        if (target.closest("select") && target !== modelChoice) return
        if (toggleKey && target !== toggle && target !== batchButton && target.closest("button, a, [role=button]")) return
      }
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
      if (event.key.toLowerCase() === "t") {
        if (!priorHeld) {
          priorHeld = true
          syncPriorView()
        }
        return
      }
      if (event.repeat) return
      if (toggleKey) {
        if (!toggle.disabled) toggleAnalysis()
      } else if (bestMove !== null && !terminal) {
        const moves = frozenPv || [bestMove]
        editor.playLineFromCursor(`${position.line}${moves.map(moveText).join("")}`)
      }
    })
    window.addEventListener("keyup", (event) => {
      if (event.key.toLowerCase() === "t") clearHeldPriorView()
    })
    window.addEventListener("blur", () => {
      clearHeldPriorView()
      setHoveredMove(null)
    })

    if (!navigator.gpu) {
      fail("Analysis requires WebGPU. Try a browser and device with WebGPU support.", true)
      return view
    }
    try {
      worker = new Worker(new URL("./webgpu/katahex/worker.js", document.baseURI), { type: "module" })
      worker.onmessage = ({ data }) => {
        if (data.model && data.model !== selectedModel()) return
        if (data.id !== undefined && (data.id !== requestId || !active)) return
        if (data.type === "loading") {
          if (active) status.textContent = data.message
        } else if (data.type === "ready") {
          ready = true
          runtime.textContent = `GPU: ${data.runtime}`
          runtime.title = data.runtimeHelp
          if (active) status.textContent = batchRun ? `Fast analysis… 0/${batchRun.total}` : ""
        } else if (data.type === "error") {
          fail(data.message, data.fatal)
        } else if (data.type === "result") {
          rememberResult(data)
          editor.refreshBoard()
        } else if (data.type === "batch-result" && batchRun) {
          graph.recordBatch(batchRun.variation, data.results)
          status.textContent = `Fast analysis… ${data.completed}/${batchRun.total}`
        } else if (data.type === "batch-finished" && batchRun) {
          const run = batchRun
          batchRun = null
          active = false
          run.variation.points.at(-1).select()
          status.textContent = `Analyzed ${run.total} positions.`
        } else if (data.type === "finished") {
          active = false
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
      if (active) {
        stopAnalysis()
      }
    })
    return view
  }

  window.BoardAnalysis = { create, overlayText, shortcutHelpGroups }
})()
