import { readModel } from "./model.js"
import { Network, Fp16CnnMemoryError } from "./network.js"
import {
  GAME_OVER, replayBatchStates, resultFromLogits, stateFromPosition, terminalSearchReport, terminalValue,
} from "./search.js"
import { cachedSearchReport, searchSession, trimSearchCache } from "./search_cache.js"

let network = null
let loadedModel = null
let float32Retry = null
let desired = null
let task = null
let loading = null
let requestId = 0
let networkOwner = null
const searches = new Map()
const REPORT_INTERVAL_MS = 250
const yieldChannel = new MessageChannel()

function yieldToRequests() {
  return new Promise(resolve => {
    yieldChannel.port1.onmessage = resolve
    yieldChannel.port2.postMessage(null)
  })
}

function send(type, fields = {}) { self.postMessage({ type, id: requestId, ...fields }) }

function sendLoading() {
  const { model, percent } = loading
  send("loading", { model, percent })
}

function sendReady(model) {
  send("ready", { model, runtime: network.runtime, runtimeHelp: network.runtimeHelp })
}

function disposeNetwork() {
  networkOwner = null
  network.destroy()
  network = null
  loadedModel = null
  searches.clear()
}

async function requestAdapter(signal) {
  let onAbort
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason)
    signal.addEventListener("abort", onAbort, { once: true })
  })
  try {
    const adapter = await Promise.race([navigator.gpu.requestAdapter(), aborted])
    signal.throwIfAborted()
    if (!adapter) throw new Error("No WebGPU adapter is available on this device.")
    return adapter
  } finally {
    signal.removeEventListener("abort", onAbort)
  }
}

async function load() {
  const { model: modelName, position: { game } } = desired
  const forceFloat32 = float32Retry === modelName
  float32Retry = null
  const abort = new AbortController()
  const { signal } = abort
  loading = { model: modelName, abort, percent: null }
  let progressTimer
  try {
    if (!navigator.gpu) throw new Error("Analysis needs a browser with WebGPU support.")
    sendLoading()
    const [adapter, model] = await Promise.all([
      requestAdapter(signal),
      (async () => {
        // Published weights can change at the same URL.
        const response = await fetch(new URL(`./models/${modelName}.q7.gz`, import.meta.url), {
          signal,
          cache: "no-cache",
        })
        signal.throwIfAborted()
        if (!response.ok) throw new Error("The analysis model could not be loaded. Try Start again.")
        const total = Number(response.headers.get("Content-Length"))
        const encoding = response.headers.get("Content-Encoding")
        const hasTotal = Number.isFinite(total) && total > 0 &&
          (!encoding || encoding.toLowerCase() === "identity")
        let received = 0
        let percent = null
        let nextProgressAt = performance.now() + REPORT_INTERVAL_MS
        function reportProgress() {
          progressTimer = undefined
          if (signal.aborted) return
          nextProgressAt = performance.now() + REPORT_INTERVAL_MS
          loading.percent = percent
          sendLoading()
        }
        const stream = response.body.pipeThrough(new TransformStream({
          transform(chunk, controller) {
            if (hasTotal && !signal.aborted) {
              received += chunk.byteLength
              percent = Math.min(99, Math.floor(100 * received / total))
              if (percent !== loading.percent && progressTimer === undefined) {
                const wait = nextProgressAt - performance.now()
                if (wait <= 0) reportProgress()
                else progressTimer = setTimeout(reportProgress, Math.ceil(wait))
              }
            }
            controller.enqueue(chunk)
          },
          flush() {
            clearTimeout(progressTimer)
            percent = 100
            reportProgress()
          },
        }))
        return readModel(stream, signal)
      })(),
    ])
    signal.throwIfAborted()
    const owner = {}
    networkOwner = owner
    const prepared = await Network.create(adapter, model, (error) => {
      if (networkOwner !== owner || signal.aborted) return
      networkOwner = null
      desired = null
      if (loading?.abort === abort) abort.abort(error)
      send("error", { model: modelName, fatal: true, message: error.message || String(error) })
      pump()
    }, game, signal, forceFloat32)
    if (signal.aborted) {
      prepared.destroy()
      return
    }
    network = prepared
    loadedModel = modelName
    sendReady(modelName)
  } catch (error) {
    if (!signal.aborted) {
      if (!forceFloat32 && error instanceof Fp16CnnMemoryError) {
        float32Retry = modelName
      } else {
        desired = null
        send("error", { model: modelName, fatal: true, message: error.message || String(error) })
      }
    }
  } finally {
    clearTimeout(progressTimer)
    if (!network) {
      networkOwner = null
      abort.abort()
    }
    loading = null
    pump()
  }
}

function pump() {
  // Work using the network must unwind before releasing its device or cached searches.
  if (task || loading) return
  if (network && !networkOwner) disposeNetwork()
  if (!desired) return
  if (!network) {
    void load()
    return
  }
  task = (async () => {
    while (desired && networkOwner) {
      const request = desired
      try {
        if (request.type === "batch") {
          await runBatch(request)
          continue
        }
        if (request.type === "restore") {
          const report = cachedSearchReport(searches, request.position, network, request.wideRootNoise, request.memoryLimit)
          if (report) send("cached-result", { id: request.id, ...report, reportedAt: performance.now() })
          else send("cache-miss", { id: request.id })
          desired = null
          continue
        }
        const { search, key } = searchSession(searches, request.position, network, request.wideRootNoise)
        search.memoryLimit = request.memoryLimit
        const report = search.report()
        send("result", {
          id: request.id, ...report, reportedAt: performance.now(), finished: search.finishedReason !== null,
        })
        search.wideRootNoise = request.wideRootNoise
        trimSearchCache(searches, key)
        let lastReport = -Infinity
        let sentChildVisit = search.root.visits > 1
        while (desired === request) {
          const evaluated = await search.step(() => desired !== request)
          if (desired !== request) break
          const now = performance.now()
          const reason = search.finishedReason
          const done = reason !== null
          const reportDue = now - lastReport >= REPORT_INTERVAL_MS || done
          const hasChildVisit = search.root.visits > 1
          if (reportDue || (!sentChildVisit && hasChildVisit)) {
            const report = search.report()
            if (reportDue) trimSearchCache(searches, key)
            send("result", {
              id: request.id, ...report, reportedAt: performance.now(), finished: done, early: !reportDue,
            })
            sentChildVisit ||= hasChildVisit
            if (reportDue) lastReport = now
          }
          if (done) {
            desired = null
            send("finished", { id: request.id, reason })
            break
          }
          // Terminal-only traversals need a task boundary to receive Stop or
          // a new position. Repeated zero-delay timers are clamped to 4 ms.
          if (!evaluated) await yieldToRequests()
        }
      } catch (error) {
        if (desired === request) {
          if (error instanceof Fp16CnnMemoryError) {
            float32Retry = loadedModel
            disposeNetwork()
          } else {
            desired = null
            send("error", { id: request.id, fatal: false, message: error.message || String(error) })
          }
        }
      }
    }
  })().finally(() => {
    task = null
    pump()
  })
}

async function runBatch(request) {
  const { game, boardSize } = request.position
  const { moves, plies } = request
  const cancelled = () => desired !== request
  // Amortize the temporary graph across long Batch runs.
  let useWide = network.largeCNN && network.half &&
    boardSize >= 14 && plies.length >= 64
  let graph = await network.prepare(boardSize)
  const batchSize = graph.batchSize
  try {
    const advance = replayBatchStates(game, boardSize, moves)
    for (let start = request.completedPlies ?? 0; start < plies.length;) {
      if (cancelled()) return
      const rows = [], states = []
      const count = start > 0 && useWide ? 2 * batchSize : batchSize
      const end = Math.min(plies.length, start + count)
      for (let offset = start; offset < end; offset++) {
        const ply = plies[offset]
        const state = advance(ply)
        const terminal = terminalValue(state, game, false)
        const index = states.length
        if (terminal === null) states.push({ ...state, board: state.board.slice() })
        rows.push({ ply, terminal, index })
      }
      if (useWide && states.length > batchSize && !graph.largeBatch) {
        graph = await network.prepareWideBatch(boardSize)
        useWide = graph.largeBatch
        if (cancelled()) return
      }
      const values = states.length
        ? graph.largeBatch
          ? await graph.evaluate(states, true, states.length > batchSize ? cancelled : null)
          : await network.evaluateValues(states, cancelled)
        : []
      if (cancelled()) return
      const results = rows.map(({ ply, terminal, index }) => {
        const { value, drawrate } =
          terminal === null
            ? resultFromLogits(values[index])
            : { value: terminal, drawrate: Number(terminal === 0) }
        const winrate = (1 + value) / 2
        return {
          ply,
          visits: 1,
          redWinrate: ply % 2 === 0 ? winrate : 1 - winrate,
          ...(game === "yavalath" ? { drawrate } : {}),
        }
      })
      send("batch-result", { id: request.id, results, nextPly: plies[end] ?? moves.length + 1 })
      // A precision retry can follow a later graph allocation. Resume after the published rows.
      request.completedPlies = end
      start = end
      if (!states.length) await new Promise((resolve) => setTimeout(resolve, 0))
    }
  } finally {
    // The next evaluation builds the graph at its ordinary width.
    if (graph.largeBatch) network.releaseGraph()
  }
  if (cancelled()) return
  desired = null
  send("batch-finished", { id: request.id })
}

self.onmessage = ({ data }) => {
  requestId = data.id
  if (network && loadedModel !== data.model) networkOwner = null
  if (data.type === "restore") {
    desired = networkOwner && network ? data : null
    if (!desired) send("cache-miss")
  } else if (data.type === "start" || data.type === "batch") {
    desired = data
    if (!(network && networkOwner) && data.type === "start") {
      try {
        const terminal = terminalValue(stateFromPosition(data.position), data.position.game, false)
        if (terminal !== null) {
          desired = null
          send("result", { ...terminalSearchReport(terminal, 1), reportedAt: performance.now(), finished: true })
          send("finished", { reason: GAME_OVER })
        }
      } catch (error) {
        desired = null
        send("error", { fatal: false, message: error.message || String(error) })
      }
    }
    if (networkOwner && network) sendReady(data.model)
  } else if (data.type === "stop") {
    desired = null
    if (data.clear) searches.clear()
  }
  if (loading) {
    if (loading.model !== desired?.model) loading.abort.abort()
    // A newer position can continue the same load under its own request ID.
    else if (!loading.abort.signal.aborted) sendLoading()
  }
  pump()
}
