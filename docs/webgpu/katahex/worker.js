import { readModel } from "./model.js"
import { Network } from "./gpu.js"
import { preloadWinogradWeights } from "./winograd_kernels.js"
import { stateFromPosition, terminalValue, resultFromLogits } from "./search.js"
import { searchSession, trimSearchCache } from "./search_cache.js"

let network = null
let loadedModel = null
let desired = null
let task = null
let loading = null
let requestId = 0
let networkOwner = null
const searches = new Map()
const REPORT_INTERVAL_MS = 250

function send(type, fields = {}) { self.postMessage({ type, id: requestId, ...fields }) }

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
  const abort = new AbortController()
  const { signal } = abort
  loading = { model: modelName, abort }
  try {
    if (!navigator.gpu) throw new Error("Analysis needs a browser with WebGPU support.")
    send("loading", { model: modelName, message: "Loading model…" })
    const [adapter, model] = await Promise.all([
      requestAdapter(signal).then((adapter) => {
        signal.throwIfAborted()
        if (modelName === "katahex-20240812" && adapter.features.has("shader-f16")) {
          preloadWinogradWeights()
        }
        return adapter
      }),
      (async () => {
        const response = await fetch(new URL(`../models/${modelName}.q7.gz`, import.meta.url), {
          signal,
          cache: "no-cache",
        })
        signal.throwIfAborted()
        if (!response.ok) throw new Error("The analysis model could not be loaded. Try Start again.")
        send("loading", { model: modelName, message: "Reading model…" })
        return readModel(response.body, signal)
      })(),
    ])
    signal.throwIfAborted()
    send("loading", { model: modelName, message: "Preparing GPU…" })
    const owner = {}
    networkOwner = owner
    const prepared = await Network.create(adapter, model, (error) => {
      if (networkOwner !== owner || signal.aborted) return
      networkOwner = null
      desired = null
      if (loading?.abort === abort) abort.abort(error)
      send("error", { model: modelName, fatal: true, message: error.message || String(error) })
      pump()
    }, modelName, game, signal)
    if (signal.aborted) {
      prepared.destroy()
      return
    }
    network = prepared
    loadedModel = modelName
    sendReady(modelName)
  } catch (error) {
    if (!signal.aborted) {
      desired = null
      send("error", { model: modelName, fatal: true, message: error.message || String(error) })
    }
  } finally {
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
        const key = request.position.key
        const session = searchSession(
          searches,
          request.position,
          network,
          request.wideRootNoise,
          request.type !== "restore",
        )
        if (session) session.search.memoryLimit = request.memoryLimit
        if (session) {
          const report = session.search.report()
          send(request.type === "restore" ? "cached-result" : "result", {
            id: request.id,
            ...report,
            reportedAt: performance.now(),
            finished: session.search.finishedReason !== null,
          })
        }
        if (request.type === "restore") {
          desired = null
          trimSearchCache(searches, key)
          continue
        }
        session.search.wideRootNoise = request.wideRootNoise
        trimSearchCache(searches, key)
        const search = session.search
        let lastReport = -Infinity
        while (desired === request) {
          const evaluated = await search.step(() => desired !== request)
          if (desired !== request) break
          const now = performance.now()
          const reason = search.finishedReason
          const done = reason !== null
          if (now - lastReport >= REPORT_INTERVAL_MS || done) {
            const report = search.report()
            trimSearchCache(searches, key)
            send("result", { id: request.id, ...report, reportedAt: performance.now(), finished: done })
            lastReport = now
          }
          if (done) {
            desired = null
            send("finished", { id: request.id, reason })
            break
          }
          // Terminal-only traversals must yield to worker messages too.
          if (!evaluated) await new Promise((resolve) => setTimeout(resolve, 0))
        }
      } catch (error) {
        if (desired === request) {
          desired = null
          send("error", { id: request.id, fatal: false, message: error.message || String(error) })
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
  const cancelled = () => desired !== request
  const batchSize = network.prepare(boardSize).batchSize
  for (let start = request.startPly; start <= request.moves.length; start += batchSize) {
    if (cancelled()) return
    const rows = [], states = []
    const end = Math.min(request.moves.length + 1, start + batchSize)
    for (let ply = start; ply < end; ply++) {
      const moves = request.moves.slice(0, ply)
      const board = globalThis.HexPosition.materializeBoardState(
        moves,
        boardSize,
        game !== "hex" ? { swapStone: (stone) => ({ ...stone, color: "blue", ply: "S" }) } : {},
      )
      const state = stateFromPosition({ game, boardSize, moves, stones: board.stones, toPlay: board.toPlay })
      const terminal = terminalValue(state, game, false)
      const index = states.length
      if (terminal === null) states.push(state)
      rows.push({ ply, toPlay: board.toPlay, terminal, index })
    }
    const evaluations = states.length ? await network.evaluate(states, cancelled) : []
    if (cancelled()) return
    const results = rows.map(({ ply, toPlay, terminal, index }) => {
      const { value, drawrate } =
        terminal === null
          ? resultFromLogits(evaluations[index].value)
          : { value: terminal, drawrate: Number(terminal === 0) }
      const winrate = (1 + value) / 2
      return {
        ply,
        visits: 1,
        redWinrate: toPlay === "red" ? winrate : 1 - winrate,
        ...(game === "yavalath" ? { drawrate } : {}),
      }
    })
    send("batch-result", { id: request.id, results, completed: end - request.startPly })
    if (!states.length) await new Promise((resolve) => setTimeout(resolve, 0))
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
  } else if (data.type === "start" || data.type === "batch") {
    desired = data
    if (networkOwner && network) sendReady(data.model)
  } else if (data.type === "stop") {
    desired = null
    if (data.clear) searches.clear()
  }
  if (loading && loading.model !== desired?.model) loading.abort.abort()
  pump()
}
