// SPDX-License-Identifier: MIT
import { readModel } from "./model.js"
import { Network } from "./gpu.js"
import { stateFromPosition, terminalValue, valueFromLogits } from "./search.js"
import { searchSession, trimSearchCache } from "./search_cache.js"

let network = null
let loadedModel = null
let desired = null
let task = null
let loading = false
let loadingModel = null
let loadingAbort = null
let failed = false
const searches = new Map()
const REPORT_INTERVAL_MS = 250

function send(type, fields = {}) {
  self.postMessage({ type, ...fields })
}

function gpuFailed(error, model) {
  if (failed) return
  failed = true
  desired = null
  send("error", { model, fatal: true, message: error.message || String(error) })
}

async function load() {
  loading = true
  await task
  const modelName = desired?.model
  if (!modelName) { loading = false; return }
  loadingModel = modelName
  loadingAbort = new AbortController()
  const obsolete = () => desired?.model !== modelName
  const game = desired.position.game
  failed = false
  network?.destroy()
  network = null
  loadedModel = null
  searches.clear()
  try {
    if (!navigator.gpu) throw new Error("Analysis needs a browser with WebGPU support.")
    send("loading", { model: modelName, message: "Loading model…" })
    const archive = {
      "b5nbt-14": "b5nbt-14.q8.gz",
      "katahex-20240812": "katahex-20240812.q8.gz",
      "b10nbt-24": "b10nbt-24.q8.gz",
    }[modelName]
    const response = await fetch(new URL(`../models/${archive}`, import.meta.url), { signal: loadingAbort.signal })
    if (obsolete()) return
    if (!response.ok) throw new Error("The analysis model could not be loaded. Try Start again.")
    const file = await response.blob()
    if (obsolete()) return
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error("No WebGPU adapter is available on this device.")
    const packedInt8 = !adapter.features.has("shader-f16") &&
      navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") || false
    send("loading", { model: modelName, message: "Reading model…" })
    let model = await readModel(file, packedInt8)
    if (obsolete()) return
    send("loading", { model: modelName, message: "Preparing GPU…" })
    const onLost = (error) => {
      if (desired ? desired.model === modelName : loadedModel === modelName) gpuFailed(error, modelName)
    }
    if (packedInt8) {
      try {
        network = await Network.create(model, onLost, modelName, true, game)
      } catch (error) {
        if (obsolete()) return
        console.warn("8-bit GPU setup failed; using float GPU.", error)
        send("loading", { model: modelName, message: "Preparing float GPU…" })
        model = await readModel(file)
        if (obsolete()) return
        network = await Network.create(model, onLost, modelName, false, game)
      }
    } else network = await Network.create(model, onLost, modelName, false, game)
    if (obsolete()) {
      network.destroy()
      network = null
      return
    }
    loadedModel = modelName
    if (!failed) send("ready", { model: modelName, runtime: network.runtime, runtimeHelp: network.runtimeHelp })
  } catch (error) {
    network?.destroy()
    network = null
    loadedModel = null
    if (desired?.model === modelName && !loadingAbort.signal.aborted) {
      desired = null
      send("error", { model: modelName, fatal: true, message: error.message || String(error) })
    }
  } finally {
    loadingModel = null
    loadingAbort = null
    loading = false
    if (desired && loadedModel !== desired.model) void load()
    else pump()
  }
}

function pump() {
  if (task || loading || !network || failed || !desired || loadedModel !== desired.model) return
  task = (async () => {
    while (desired && !failed && !loading && loadedModel === desired.model) {
      const request = desired
      if (request.type === "batch") {
        try {
          await runBatch(request)
        } catch (error) {
          if (desired === request) {
            desired = null
            send("error", { id: request.id, fatal: false, message: error.message || String(error) })
          }
        }
        continue
      }
      const key = request.position.key
      const session = searchSession(searches, request.position, network, request.wideRootNoise, request.type !== "restore")
      if (session) session.search.memoryLimit = request.memoryLimit
      if (session) {
        const report = session.search.report()
        send(request.type === "restore" ? "cached-result" : "result", {
          id: request.id, ...report,
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
      try {
        while (desired === request && !failed) {
          const evaluated = await search.step(() => desired !== request || failed)
          if (desired !== request || failed) break
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
  })().finally(() => { task = null; pump() })
}

async function runBatch(request) {
  const { game, boardSize } = request.position
  const cancelled = () => desired !== request || failed
  const batchSize = network.prepare(boardSize).batchSize
  for (let start = request.startPly; start <= request.moves.length; start += batchSize) {
    if (cancelled()) return
    const rows = [], states = []
    const end = Math.min(request.moves.length + 1, start + batchSize)
    for (let ply = start; ply < end; ply++) {
      const moves = request.moves.slice(0, ply)
      const board = globalThis.HexPosition.materializeBoardState(moves, boardSize, game === "y"
        ? { swapStone: (stone) => ({ ...stone, color: "blue", ply: "S" }) } : {})
      const state = stateFromPosition({ game, boardSize, moves, stones: board.stones, toPlay: board.toPlay })
      const terminal = terminalValue(state, game)
      const index = states.length
      if (terminal === null) states.push(state)
      rows.push({ ply, toPlay: board.toPlay, terminal, index })
    }
    const evaluations = states.length ? await network.evaluate(states, cancelled) : []
    if (cancelled()) return
    const results = rows.map(({ ply, toPlay, terminal, index }) => {
      const value = terminal ?? valueFromLogits(evaluations[index].value)
      const winrate = (1 + value) / 2
      return { ply, visits: 1, redWinrate: toPlay === "red" ? winrate : 1 - winrate }
    })
    send("batch-result", { id: request.id, results, completed: end - request.startPly })
    if (!states.length) await new Promise((resolve) => setTimeout(resolve, 0))
  }
  if (cancelled()) return
  desired = null
  send("batch-finished", { id: request.id })
}

self.onmessage = ({ data }) => {
  if (data.type === "restore") {
    desired = network && !failed && loadedModel === data.model ? data : null
    pump()
  } else if (data.type === "start" || data.type === "batch") {
    desired = data
    if (loading && loadingModel && loadingModel !== data.model) loadingAbort?.abort()
    if (!loading && network && !failed && loadedModel === data.model) {
      send("ready", { model: data.model, runtime: network.runtime, runtimeHelp: network.runtimeHelp })
      pump()
    } else if (!loading) void load()
  } else if (data.type === "stop") {
    desired = null
    loadingAbort?.abort()
    if (data.clear) searches.clear()
  }
}
