// SPDX-License-Identifier: MIT
// PUCT selection, first-play urgency, and Hex board rules adapted from KataHex.
// See THIRD_PARTY_NOTICES.md.
import { CaptureMoves } from "./capture_moves.js"
import { canonicalConnectionBoard, rootSymmetryPruning, symmetriesFor, symmetryPoint } from "./symmetry.js"
import {
  createNode,
  positionKey,
  childKey,
  copyGraph,
  retainedVisits,
  EMPTY_VISITS,
  graphBytes,
} from "./search_graph.js"
import { childWeight, recompute, explorationScale, selectionWeights } from "./search_stats.js"
import {
  COLD_HEADER_BYTES, candidateBytes, coldLogitSize, expandedCandidateBytes,
  packCandidates, packColdCandidates, unpackColdCandidates,
} from "./search_candidates.js"
import { Y_CAPTURE_TABLE, bentYCaptureNeighbors, hasSideContact, yStrategicWinner } from "./y_rules.js"
import "../../position.js"
import "../../benty_geometry.js"
import "../../board_geometry.js"
import "../../yavalath_rules.js"

const NEIGHBORS = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, -1], [-1, 1]]
const CAPTURE_NEIGHBORS = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
const CONNECTION_NEIGHBORS = [...CAPTURE_NEIGHBORS, CAPTURE_NEIGHBORS[0]]
const JUMP_NEIGHBORS = [[1, -2], [2, -1], [1, 1], [-1, 2], [-2, 1], [-1, -1]]
const PV_LENGTH = 15

function buildCaptureTable() {
  const table = new Uint8Array(4096)
  const seeds = [[[1, 1, 1, 1, 0, 0], 4, 5], [[1, 1, 1, 0, 2, 0], 3, 5], [[1, 1, 0, 2, 2, 0], 2, 5]]
  for (const [pattern, first, second] of seeds) {
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        pattern[first] = a
        pattern[second] = b
        let id = 0
        for (let i = 0; i < 6; i++) id |= pattern[i] << (2 * i)
        table[id] = 1
      }
    }
  }
  for (let id = 0; id < table.length; id++) {
    if (table[id] !== 1) continue
    let rotated = id
    for (let i = 0; i < 6; i++) {
      rotated = ((rotated & 3) << 10) | (rotated >> 2)
      table[rotated] = 1
    }
  }
  for (let id = 0; id < table.length; id++) {
    if (table[id] !== 1) continue
    let inverse = 0
    for (let i = 0; i < 6; i++) {
      const color = (id >> (2 * i)) & 3
      inverse |= (color === 1 ? 2 : color === 2 ? 1 : 0) << (2 * i)
    }
    table[inverse] = 1
  }
  for (let id = 0; id < table.length; id++) {
    if (table[id] !== 1) continue
    for (let i = 0; i < 6; i++) {
      const captured = id & ~(3 << (2 * i))
      if (table[captured] === 0) table[captured] = 2
    }
  }
  const outsideMasks = Array.from({ length: 64 }, (_, bits) => {
    let mask = 0
    for (let i = 0; i < 6; i++) mask = (mask << 2) | ((bits >> i) & 1 ? 3 : 0)
    return mask
  })
  for (let id = 0; id < table.length; id++) {
    const type = table[id]
    if (!type) continue
    for (const mask of outsideMasks) {
      const withOutside = id | mask
      if (table[withOutside] === 0 || (table[withOutside] === 2 && type === 1)) { table[withOutside] = type }
    }
  }
  return table
}

const CAPTURE_TABLE = buildCaptureTable()

function captureNeighbors(size, game) {
  if (game === "bentY") return bentYCaptureNeighbors(globalThis.BentYGeometry.topology(size))
  // Negative entries encode Hex edge colors or an unused Y-family slot (-3).
  const isY = game === "y"
  const neighbors = new Int16Array(size * size * 6)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let offset = (y * size + x) * 6
      for (const [dx, dy] of CAPTURE_NEIGHBORS) {
        const nx = x + dx, ny = y + dy
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
          neighbors[offset++] = isY && nx + ny >= size ? -3 : ny * size + nx
        } else if (isY) neighbors[offset++] = -3
        else if (nx === -1 && ny >= 1 && ny < size) neighbors[offset++] = -2
        else if (nx === size && ny >= 0 && ny < size - 1) neighbors[offset++] = -2
        else if (ny === -1 && nx >= 1 && nx < size) neighbors[offset++] = -1
        else if (ny === size && nx >= 0 && nx < size - 1) neighbors[offset++] = -1
        else neighbors[offset++] = -3
      }
    }
  }
  return neighbors
}

// Leaf collection is synchronous. Pending evaluations own copies of their move sets.
let captureCache = null

function captureFor({ board, size }, game) {
  if (game === "yavalath") return null
  if (captureCache?.board === board) return captureCache.moves

  const neighbors = captureCache?.game === game && captureCache.size === size
    ? captureCache.neighbors
    : captureNeighbors(size, game)
  const moves = new CaptureMoves(board, neighbors, game === "hex" ? CAPTURE_TABLE : Y_CAPTURE_TABLE)
  captureCache = { game, size, board, neighbors, moves }
  return moves
}

function candidateMoves(board, allowed) {
  const moves = []
  if (allowed) {
    for (let i = 0; i < allowed.length; i++) {
      for (let word = allowed[i]; word; word &= word - 1) {
        moves.push(i * 32 + 31 - Math.clz32(word & -word))
      }
    }
  }
  // If every empty cell is pruned, search them all.
  if (!moves.length) {
    for (let i = 0; i < board.length; i++) {
      if (!board[i]) moves.push(i)
    }
  }
  return moves
}

function connected(board, size, color, game) {
  const seen = new Uint8Array(board.length)
  const stack = []
  for (let i = 0; i < size; i++) {
    const pos = game === "y" || color === 1 ? i : i * size
    if (board[pos] !== color || seen[pos]) continue
    stack.push(pos)
    seen[pos] = 1
    let sides = game === "y" ? 1 : 0
    while (stack.length) {
      const current = stack.pop(), x = current % size, y = Math.floor(current / size)
      if (game === "y") {
        if (x === 0) sides |= 2
        if (x + y === size - 1) sides |= 4
        if (sides === 7) return true
      } else if ((color === 1 ? y : x) === size - 1) return true
      for (const [dx, dy] of NEIGHBORS) {
        const nx = x + dx, ny = y + dy, next = ny * size + nx
        if (nx >= 0 && nx < size && ny >= 0 && ny < size && !seen[next] && board[next] === color) {
          seen[next] = 1
          stack.push(next)
        }
      }
    }
  }
  return false
}

function hexConnected(board, size, color) {
  const cells = new Uint8Array(board.length)
  for (let i = 0; i < board.length; i++) {
    const point = color === 2 ? (i % size) * size + Math.floor(i / size) : i
    cells[point] = board[i] === color ? 1 : board[i] ? 2 : 0
  }
  const get = (x, y) => (x >= 0 && y >= 0 && x < size && y < size ? cells[y * size + x] : 2)
  const visit = (x, y) => {
    cells[y * size + x] = 3
    for (let i = 0; i < 6; i++) {
      const [dx, dy] = CONNECTION_NEIGHBORS[i]
      const nx = x + dx, ny = y + dy, value = get(nx, ny)
      if (value === 4 || (value === 1 && visit(nx, ny))) return true
    }
    for (let i = 0; i < 6; i++) {
      const [dx, dy] = JUMP_NEIGHBORS[i]
      const nx = x + dx, ny = y + dy, value = get(nx, ny)
      const [ax, ay] = CONNECTION_NEIGHBORS[i]
      const [bx, by] = CONNECTION_NEIGHBORS[i + 1]
      if ((value !== 1 && value !== 4) || get(x + ax, y + ay) !== 0 || get(x + bx, y + by) !== 0) continue
      if (value === 4) return true
      cells[(y + ay) * size + x + ax] = 2
      cells[(y + by) * size + x + bx] = 2
      if (visit(nx, ny)) return true
    }
    return false
  }
  for (let x = 0; x < size; x++) {
    if (cells[(size - 1) * size + x] === 1) cells[(size - 1) * size + x] = 4
  }
  for (let x = 1; x < size; x++) {
    if (cells[(size - 2) * size + x] === 1 && get(x, size - 1) === 0 && get(x - 1, size - 1) === 0) {
      cells[(size - 2) * size + x] = 4
    }
  }
  for (let x = 0; x < size; x++) {
    if (get(x, 0) === 1 && visit(x, 0)) return true
  }
  for (let x = 0; x < size - 1; x++) {
    if (get(x, 1) === 1 && get(x, 0) === 0 && get(x + 1, 0) === 0 && visit(x, 1)) return true
  }
  return false
}

function hexStrategicWinner({ board, size }) {
  const bottom = (size - 1) * size
  let oriented
  for (const color of [1, 2]) {
    // The two Hex sides for this color need physical or bridge contact.
    const contactsBothSides = color === 1
      ? hasSideContact(board, size, color, 0, 1, size) &&
        hasSideContact(board, size, color, bottom, 1, bottom - size + 1)
      : hasSideContact(board, size, color, 0, size, 1) &&
        hasSideContact(board, size, color, size - 1, size, 2 * size - 2)
    if (!contactsBothSides) continue
    oriented ??= canonicalConnectionBoard(board, size, "hex")
    if (hexConnected(oriented, size, color)) return color
  }
  return 0
}

function rootNoiseBonus(amount) {
  if (Math.random() >= 0.5) return 0
  const radius = Math.sqrt(-2 * Math.log(1 - Math.random()))
  return amount * Math.abs(radius * Math.cos(2 * Math.PI * Math.random()))
}

function select(parent, isRoot, wideRootNoise, duplicates) {
  const { moves, priors, nodes, visits } = parent.children
  let visitedMass = 0, totalChildWeight = 0
  let bestNew = -1
  for (let i = 0; i < (isRoot ? moves.length : nodes.length); i++) {
    const child = nodes[i]
    if (!child) {
      if (!duplicates?.[moves[i]] && (bestNew < 0 || priors[i] > priors[bestNew])) bestNew = i
      continue
    }
    totalChildWeight += childWeight(child, visits[i])
    visitedMass += priors[i]
  }
  if (!isRoot && bestNew < 0) {
    for (let i = nodes.length; i < moves.length; i++) {
      if (!duplicates?.[moves[i]]) {
        bestNew = i
        break
      }
    }
  }
  const exploration = explorationScale(parent, totalChildWeight)
  const parentWeight = Math.min(1, visitedMass * visitedMass)
  const parentValue = parent.value
  const fpu =
    parentWeight * parentValue +
    (1 - parentWeight) * parent.initialValue -
    (isRoot ? 0.1 : 0.2) * Math.sqrt(visitedMass)
  let bestIndex = -1, bestScore = -Infinity
  const candidates = isRoot ? moves.length : Math.max(nodes.length, bestNew + 1)
  for (let i = 0; i < candidates; i++) {
    if (duplicates?.[moves[i]]) continue
    const child = nodes[i]
    // Sampling AWRN for every new move would favor low-prior outliers.
    if (!child && i !== bestNew) continue
    const childVisits = child?.visits || 0
    const inFlight = child?.inFlight || 0
    const weight = child ? childWeight(child, visits[i]) : 0
    const count = weight + inFlight
    let utility = childVisits && weight > 0 ? -child.value : fpu
    if (inFlight) utility += ((-1 - utility) * inFlight) / (inFlight + Math.max(0.25, weight))
    const prior = isRoot && wideRootNoise > 0 ? priors[i] ** (1 / (4 * wideRootNoise + 1)) : priors[i]
    if (isRoot && wideRootNoise > 0) utility += rootNoiseBonus(wideRootNoise)
    const score = utility + (exploration * prior) / (1 + count)
    if (score > bestScore) {
      bestIndex = i
      bestScore = score
    }
  }
  return bestIndex
}

export function resultFromLogits(logits) {
  const max = Math.max(...logits)
  const values = Array.from(logits, (x) => Math.exp(x - max))
  const total = values[0] + values[1] + values[2]
  return { value: (values[0] - values[1]) / total, drawrate: values[2] / total }
}

function averagedResult(evaluations) {
  let value = 0, drawrate = 0
  for (const evaluation of evaluations) {
    const result = resultFromLogits(evaluation.value)
    value += result.value
    drawrate += result.drawrate
  }
  return { value: value / evaluations.length, drawrate: drawrate / evaluations.length }
}

function normalizedPriors(moves, policy) {
  let max = -Infinity
  for (const move of moves) max = Math.max(max, policy[move])
  const priors = new Float64Array(moves.length)
  let sum = 0
  for (let i = 0; i < moves.length; i++) {
    priors[i] = Math.exp((policy[moves[i]] - max) / 1.1)
    sum += priors[i]
  }
  for (let i = 0; i < priors.length; i++) priors[i] /= sum
  return priors
}

function expand(leaf, policies, isRoot, half) {
  const { current, state, moves } = leaf
  if (!isRoot) {
    // Readback widens the native logits to Float32Array without changing their values.
    current.children = packColdCandidates(leaf, policies[0], state.board.length, half)
    return
  }
  setChildren(current, moves, averagedPriors(moves, policies), state.board.length, true)
}

function setChildren(parent, moves, priors, points, isRoot) {
  // A typed move array has room for every index as well as every cell ID.
  const ranked = moves.map((_, i) => i)
  ranked.sort((a, b) => priors[b] - priors[a])
  parent.children = {
    ...packCandidates(moves, priors, points, { order: ranked }),
    nodes: [],
    visits: EMPTY_VISITS,
    order: isRoot ? [] : null,
  }
}

function averagedPriors(moves, policies) {
  const priors = normalizedPriors(moves, policies[0])
  for (let s = 1; s < policies.length; s++) {
    const next = normalizedPriors(moves, policies[s])
    for (let i = 0; i < priors.length; i++) priors[i] += next[i]
  }
  if (policies.length > 1) {
    for (let i = 0; i < priors.length; i++) priors[i] /= policies.length
  }
  return priors
}

export function stateFromPosition(position) {
  // Hex search uses native colors and coordinates after swap.
  const swapped = position.game === "hex" && position.moves[1] === "swap"
  const size = position.boardSize
  const board = new Uint8Array(size * size)
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      if (!globalThis.BoardGeometry.isPlayable(position.game, col, row, size)) board[row * size + col] = 3
    }
  }
  for (const stone of position.stones) {
    const col = swapped ? stone.row : stone.col
    const row = swapped ? stone.col : stone.row
    const color = stone.color === "red" ? 1 : 2
    board[(row - 1) * size + col - 1] = swapped ? 3 - color : color
  }
  const toPlay = position.toPlay === "red" ? 1 : 2
  return { board, size, toPlay: swapped ? 3 - toPlay : toPlay }
}

export function terminalValue(state, game, includeVirtualConnections) {
  if (game === "yavalath") { return globalThis.YavalathRules.terminalValue(state) }
  const { board, size, toPlay } = state
  let winner
  if (includeVirtualConnections && game === "hex" && size >= 4) {
    winner = hexStrategicWinner(state)
  } else if (includeVirtualConnections && (game === "y" || game === "bentY")) {
    winner = yStrategicWinner(state, game)
  } else if (game === "bentY") {
    winner = globalThis.BentYGeometry.winner(board, size)
  } else {
    winner = connected(board, size, 1, game) ? 1 : connected(board, size, 2, game) ? 2 : 0
  }
  return winner ? (winner === toPlay ? 1 : -1) : null
}

export function subtreeQuery(position) {
  const { game } = position
  const state = stateFromPosition(position)
  const { duplicates } = rootSymmetryPruning(state, game)
  const candidates = symmetriesFor(game).map((symmetry) => {
    const board = new Uint8Array(state.board.length)
    const inverse = new Uint16Array(board.length)
    for (let point = 0; point < board.length; point++) {
      const mapped = symmetryPoint(point, state.size, game, symmetry)
      board[mapped] = state.board[point]
      inverse[mapped] = point
    }
    return { key: positionKey({ board, toPlay: state.toPlay }), transform: (point) => inverse[point] }
  })
  return { candidates, duplicates }
}

export class Search {
  constructor(position, network, wideRootNoise) {
    this.game = position.game
    this.swapped = this.game === "hex" && position.moves[1] === "swap"
    this.state = stateFromPosition(position)
    this.rootSymmetry = rootSymmetryPruning(this.state, this.game)
    this.network = network
    this.evaluationSymmetries = network.model?.graphAttention ? [0] : symmetriesFor(this.game)
    this.wideRootNoise = wideRootNoise
    this.root = createNode(positionKey(this.state))
    this.nodes = new Map([[this.root.key, this.root]])
    this.needsRootEvaluation = false
    this.policyBytes = 0
    this.childCapacity = 0
    this.visitLimit = Infinity
    this.memoryLimit = Infinity
    const points = this.state.board.length
    const policyBytes = expandedCandidateBytes(points, points)
    // Reserve a new node, child-array growth and at most one older policy promotion.
    // A new edge to a visited transposition stops at catch-up.
    const promotionBytes = policyBytes - coldLogitSize(network.half) * points
    this.expansionBytes = graphBytes(1, policyBytes + COLD_HEADER_BYTES + promotionBytes, points, this.root.key.length)
  }

  get estimatedBytes() {
    return graphBytes(this.nodes.size, this.policyBytes, this.childCapacity, this.root.key.length)
  }

  get finishedReason() {
    if (this.root.terminal !== null) return "Game over."
    if (this.root.visits >= this.visitLimit) return `Stopped at ${this.visitLimit.toLocaleString()} visits.`
    if (this.estimatedBytes + this.expansionBytes > this.memoryLimit) return "Stopped at the search memory limit."
    return null
  }

  subtree({ candidates, duplicates }) {
    let best = null
    for (const { key, transform } of candidates) {
      const root = this.nodes.get(key)
      if (!root?.visits) continue
      const visits = retainedVisits(root, transform, duplicates)
      if (visits && (!best || visits > best.visits)) best = { root, transform, visits }
    }
    return best
  }

  inherit({ root, transform }) {
    const graph = copyGraph(root, this.state.board.length, transform, this.rootSymmetry.duplicates)
    this.root = graph.root
    this.nodes = graph.nodes
    this.policyBytes = graph.policyBytes
    if (this.root.children instanceof ArrayBuffer) this.materialize(this.root, true)
    this.childCapacity = graph.childCapacity
    this.root.terminal = terminalValue(this.state, this.game, false)
    if (this.root.children) recompute(this.root)
    this.needsRootEvaluation = this.root.children !== null && this.root.terminal === null
  }

  materialize(node, isRoot) {
    const points = this.state.board.length
    const oldBytes = candidateBytes(node.children)
    const { moves, policy } = unpackColdCandidates(node.children, points, node.coldMapping)
    setChildren(node, moves, normalizedPriors(moves, policy), points, isRoot)
    node.coldMapping = null
    this.policyBytes += candidateBytes(node.children) - oldBytes
  }

  backup(path, deferRoot = false) {
    for (let i = path.length - 1; i >= 0; i--) {
      const { node, index } = path[i]
      if (index >= 0) node.children.visits[index]++
      node.visits++
      if (i > 0 || !deferRoot) recompute(node)
    }
  }

  async evaluateRoot(cancelled) {
    const state = this.state
    const symmetries = this.evaluationSymmetries
    if (symmetries.length === 1 || this.rootSymmetry.symmetries.length === 1) {
      return this.network.evaluate(symmetries.map((symmetry) => ({ ...state, symmetry })), cancelled)
    }

    // The playable mask and side to move are invariant across these orientations.
    // Identical oriented stones can share one evaluation.
    const { board, size } = state
    const occupied = []
    for (let point = 0; point < board.length; point++) {
      if (board[point] === 1 || board[point] === 2) occupied.push(point)
    }
    const transform = (point, symmetry) => symmetryPoint(point, size, this.game, symmetry)
    const byPosition = new Map()
    const inputs = []
    const sources = symmetries.map((symmetry) => {
      const stones = occupied.map((point) => transform(point, symmetry) * 2 + board[point] - 1)
      stones.sort((a, b) => a - b)
      const key = String.fromCharCode(...stones)
      let index = byPosition.get(key)
      if (index === undefined) {
        index = inputs.length
        byPosition.set(key, index)
        inputs.push({ ...state, symmetry })
      }
      return index
    })

    const evaluations = await this.network.evaluate(inputs, cancelled)
    if (!evaluations || cancelled()) return null
    const inverses = new Map()
    // Undo the representative's policy orientation, then apply each original orientation.
    return sources.map((index, i) => {
      const { policy, value } = evaluations[index]
      const representative = inputs[index].symmetry
      const symmetry = symmetries[i]
      if (symmetry === representative) return evaluations[index]
      let inverse = inverses.get(representative)
      if (!inverse) {
        inverse = new Uint16Array(board.length)
        for (let point = 0; point < board.length; point++) inverse[transform(point, representative)] = point
        inverses.set(representative, inverse)
      }
      const oriented = Float32Array.from(policy, (_, point) => policy[inverse[transform(point, symmetry)]])
      return { policy: oriented, value }
    })
  }

  async step(cancelled) {
    if (this.needsRootEvaluation) {
      const evaluations = await this.evaluateRoot(cancelled)
      if (!evaluations || cancelled()) return true
      const { value, drawrate } = averagedResult(evaluations)
      Object.assign(
        this.root.children,
        packCandidates(
          this.root.children.moves,
          averagedPriors(
            this.root.children.moves,
            evaluations.map((evaluation) => evaluation.policy),
          ),
          this.state.board.length,
        ),
      )
      this.root.initialValue = value
      this.root.initialDrawrate = drawrate
      recompute(this.root)
      this.needsRootEvaluation = false
      return true
    }
    if (this.finishedReason) return false
    const pending = []
    const capture = captureFor(this.state, this.game)
    const fullBatch = this.network.graph?.searchBatchSize ?? 1
    const batchSize =
      this.root.visits === 0 ? 1 : Math.min(fullBatch, Math.ceil(Math.sqrt(fullBatch * this.root.visits)))
    const capacity = Math.floor((this.memoryLimit - this.estimatedBytes) / this.expansionBytes)
    const limit = Math.min(batchSize, this.visitLimit - this.root.visits, capacity)
    let rootNeedsRecompute = false
    try {
      for (let i = 0; i < limit; i++) {
        let current = this.root
        const state = { ...this.state, board: this.state.board.slice() }
        capture?.reset()
        const path = [{ node: current, index: -1 }]
        let caughtUp = false
        while (current.children) {
          if (current.children instanceof ArrayBuffer) this.materialize(current, current === this.root)
          if (!current.children.moves.length) break
          const isRoot = current === this.root
          const index = select(current, isRoot, this.wideRootNoise, isRoot ? this.rootSymmetry.duplicates : null)
          if (index < 0) {
            current = null
            break
          }
          const children = current.children
          let child = children.nodes[index]
          const move = children.moves[index]
          if (!child) {
            const key = childKey(current.key, move, state.toPlay)
            child = this.nodes.get(key)
            if (!child) {
              child = createNode(key)
              this.nodes.set(key, child)
            }
            if (index >= children.visits.length) {
              const visits = new Float64Array(
                Math.min(children.moves.length, Math.max(index + 1, children.visits.length * 2, 1)),
              )
              visits.set(children.visits)
              this.childCapacity += visits.length - children.visits.length
              children.visits = visits
            }
            if (!children.nodes.length && index === 0) children.nodes = [child]
            else children.nodes[index] = child
            children.order?.push(index)
          }
          path[path.length - 1].index = index
          if (children.visits[index] < child.visits) {
            this.backup(path)
            caughtUp = true
            break
          }
          state.board[move] = state.toPlay
          capture?.play(move, state.toPlay, state.board)
          state.toPlay = 3 - state.toPlay
          current = child
          path.push({ node: current, index: -1 })
        }
        if (caughtUp) continue
        if (!current || (current.inFlight && !current.children)) break
        let value = current.terminal
        if (value === null) {
          value = terminalValue(state, this.game, current !== this.root)
          current.terminal = value
        }
        if (value !== null) {
          this.backup(path)
          if (current === this.root) break
        } else {
          const allowed = capture?.allowed
          const moveBitmap = current !== this.root && allowed?.some(Boolean) ? allowed.slice() : null
          const moves = moveBitmap ? null : candidateMoves(state.board, allowed)
          pending.push({ current, state, path, moves, moveBitmap })
          for (const { node } of path) node.inFlight++
        }
      }
      if (!pending.length || cancelled()) return false
      const atRoot = pending.length === 1 && pending[0].current === this.root
      const evaluations = atRoot
        ? await this.evaluateRoot(cancelled)
        : await this.network.evaluate(pending.map(({ state }) => {
          const symmetries = this.evaluationSymmetries
          return { ...state, symmetry: symmetries[Math.floor(Math.random() * symmetries.length)] }
        }), cancelled)
      if (!evaluations || cancelled()) return true
      // Descendants never read the root, and selection resumes after the whole batch.
      pending.forEach((leaf, i) => {
        const { current, path } = leaf
        const policies = atRoot ? evaluations.map((evaluation) => evaluation.policy) : [evaluations[i].policy]
        const { value, drawrate } = atRoot ? averagedResult(evaluations) : resultFromLogits(evaluations[i].value)
        current.initialValue = value
        current.initialDrawrate = drawrate
        expand(leaf, policies, current === this.root, this.network.half)
        this.policyBytes += candidateBytes(current.children)
        rootNeedsRecompute = true
        this.backup(path, true)
      })
      return true
    } finally {
      for (const { path } of pending) {
        for (const { node } of path) node.inFlight--
      }
      if (rootNeedsRecompute) recompute(this.root)
    }
  }

  report() {
    const children = this.root.children
    const rootWeights = children ? selectionWeights(this.root, true, false) : null
    const weights = children ? selectionWeights(this.root, true, true) : null
    let reportedValue = this.root.value
    let reportedDrawrate = this.root.drawrate
    if (children && rootWeights.some((weight) => weight > 0)) {
      let value = this.root.initialValue, drawrate = this.root.initialDrawrate, total = 1
      for (let i = 0; i < rootWeights.length; i++) {
        if (!rootWeights[i]) continue
        value -= rootWeights[i] * children.nodes[i].value
        drawrate += rootWeights[i] * children.nodes[i].drawrate
        total += rootWeights[i]
      }
      reportedValue = value / total
      reportedDrawrate = drawrate / total
    }
    const displayMove = (move) =>
      this.swapped ? (move % this.state.size) * this.state.size + Math.floor(move / this.state.size) : move
    const pvWeights = new Map()
    const candidates = children
      ? Array.from(children.moves, (move, i) => {
          if (this.rootSymmetry.duplicates[move]) return null
          const child = children.nodes[i]
          const visits = children.visits[i] || 0
          const pv = visits ? [move] : null
          let current = child
          while (pv && pv.length < PV_LENGTH && current?.children?.priors) {
            const branches = current.children
            let branchWeights = pvWeights.get(current)
            if (!branchWeights) {
              branchWeights = selectionWeights(current, false, true)
              pvWeights.set(current, branchWeights)
            }
            let best = -1, bestWeight = 0
            for (let j = 0; j < branches.nodes.length; j++) {
              const count = branchWeights[j]
              if (
                count > bestWeight ||
                (count > 0 && count === bestWeight && branches.priors[j] > branches.priors[best])
              ) {
                best = j
                bestWeight = count
              }
            }
            if (best < 0) break
            pv.push(branches.moves[best])
            current = branches.nodes[best]
          }
          return {
            move,
            visits,
            weight: weights[i],
            prior: children.priors[i],
            winrate: visits ? (1 - child.value) / 2 : null,
            drawrate: visits ? child.drawrate : null,
            pv,
          }
        }).filter(Boolean)
      : []
    candidates.sort((a, b) => b.weight - a.weight || b.prior - a.prior)
    // Equivalent cells share one subtree's visits.
    const displayed = []
    const seen = new Set()
    for (const row of candidates) {
      for (const symmetry of this.rootSymmetry.symmetries) {
        const transform = (move) => displayMove(symmetryPoint(move, this.state.size, this.game, symmetry))
        const move = transform(row.move)
        if (seen.has(move)) continue
        seen.add(move)
        displayed.push({ ...row, move, pv: row.pv?.map(transform) || null })
      }
    }
    return {
      visits: this.root.visits,
      winrate: this.root.visits ? (1 + reportedValue) / 2 : null,
      drawrate: this.root.visits ? reportedDrawrate : null,
      terminal: this.root.terminal !== null,
      candidates: displayed,
    }
  }
}
