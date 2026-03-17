// SPDX-License-Identifier: MIT
// PUCT selection, first-play urgency, and capture patterns adapted from KataHex.
// See THIRD_PARTY_NOTICES.md.
import { rootSymmetryPruning, symmetriesFor, symmetryPoint } from "./symmetry.js"

// Scale the visit cap inversely with board area.
export function visitLimit(size, game = "hex") {
  const points = game === "y" ? size * (size + 1) / 2 : size * size
  return Math.min(150000, Math.max(2500, Math.floor(50000 * 11 * 11 / points)))
}
const NEIGHBORS = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, -1], [-1, 1]]
const CAPTURE_NEIGHBORS = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
const PV_LENGTH = 15

function buildCaptureTable() {
  const table = new Uint8Array(4096)
  const seeds = [
    [[1, 1, 1, 1, 0, 0], 4, 5],
    [[1, 1, 1, 0, 2, 0], 3, 5],
    [[1, 1, 0, 2, 2, 0], 2, 5],
  ]
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
      if (table[withOutside] === 0 || (table[withOutside] === 2 && type === 1)) {
        table[withOutside] = type
      }
    }
  }
  return table
}

const CAPTURE_TABLE = buildCaptureTable()

function captureNeighbors(size, game) {
  // Negative entries encode edge colors: black 1, white 2, wall 3.
  // Y's -4 means the native interior-only rule does not apply.
  const isY = game === "y"
  const neighbors = new Int16Array(size * size * 6)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let offset = (y * size + x) * 6
      for (const [dx, dy] of CAPTURE_NEIGHBORS) {
        const nx = x + dx, ny = y + dy
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
          neighbors[offset++] = isY && nx + ny >= size ? -4 : ny * size + nx
        } else if (isY) neighbors[offset++] = -4
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

function isDeadOrCaptured(board, neighbors, move) {
  let pattern = 0
  const offset = move * 6
  for (let i = 0; i < 6; i++) {
    const neighbor = neighbors[offset + i]
    if (neighbor === -4) return false
    pattern = (pattern << 2) | (neighbor < 0 ? -neighbor : board[neighbor])
  }
  return CAPTURE_TABLE[pattern] !== 0
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

function play(state, move) {
  const board = state.board.slice()
  board[move] = state.toPlay
  return { board, size: state.size, toPlay: 3 - state.toPlay }
}

function node(move = null) {
  return { move, visits: 0, sum: 0, inFlight: 0, initialValue: 0, children: null, terminal: null }
}

function rootNoiseBonus(amount) {
  if (Math.random() >= 0.5) return 0
  const radius = Math.sqrt(-2 * Math.log(1 - Math.random()))
  return amount * Math.abs(radius * Math.cos(2 * Math.PI * Math.random()))
}

function select(parent, isRoot, wideRootNoise, duplicates) {
  const { moves, priors, nodes } = parent.children
  let visitedMass = 0, visits = 0
  let bestNew = -1
  for (let i = 0; i < nodes.length; i++) {
    const child = nodes[i]
    if (!child) {
      if (!duplicates?.[moves[i]] && (bestNew < 0 || priors[i] > priors[bestNew])) bestNew = i
      continue
    }
    visits += child.visits
    if (child.visits || child.inFlight) visitedMass += priors[i]
  }
  const exploration = (0.9 + 0.4 * Math.log((visits + 500) / 500)) * Math.sqrt(visits + 0.01)
  // KataHex's analysis defaults blend searched utility with the NN estimate
  // using squared visited policy mass. FPU must track revised search values.
  const parentWeight = Math.min(1, visitedMass * visitedMass)
  const parentValue = parent.sum / parent.visits
  const fpu = parentWeight * parentValue + (1 - parentWeight) * parent.initialValue
    - (isRoot ? 0.1 : 0.2) * Math.sqrt(visitedMass)
  let bestIndex = -1, bestScore = -Infinity
  for (let i = 0; i < moves.length; i++) {
    if (duplicates?.[moves[i]]) continue
    const child = nodes[i]
    // Native search considers only the highest-prior new child. Sampling AWRN
    // for every unallocated move would favor random low-prior outliers.
    if (!child && i !== bestNew) continue
    const childVisits = child?.visits || 0
    const inFlight = child?.inFlight || 0
    const count = childVisits + inFlight
    // Blend pending losses with FPU for a child still awaiting its first result,
    // as native KataHex does, rather than assigning it a certain loss.
    let utility = childVisits ? -child.sum / childVisits : fpu
    if (inFlight) {
      utility += (-1 - utility) * inFlight / (inFlight + Math.max(0.25, childVisits))
    }
    const prior = isRoot && wideRootNoise > 0
      ? priors[i] ** (1 / (4 * wideRootNoise + 1))
      : priors[i]
    if (isRoot && wideRootNoise > 0) utility += rootNoiseBonus(wideRootNoise)
    const score = utility + exploration * prior / (1 + count)
    if (score > bestScore) { bestIndex = i; bestScore = score }
  }
  if (bestIndex < 0) return null
  if (!nodes[bestIndex]) nodes[bestIndex] = node(moves[bestIndex])
  return nodes[bestIndex]
}

export function valueFromLogits(logits) {
  const max = Math.max(...logits)
  const values = Array.from(logits, (x) => Math.exp(x - max))
  return (values[0] - values[1]) / (values[0] + values[1] + values[2])
}

function normalizedPriors(moves, policy) {
  let max = -Infinity
  for (const move of moves) max = Math.max(max, policy[move])
  const priors = new Float64Array(moves.length)
  let sum = 0
  for (let i = 0; i < moves.length; i++) {
    priors[i] = Math.exp(policy[moves[i]] - max)
    sum += priors[i]
  }
  for (let i = 0; i < priors.length; i++) priors[i] /= sum
  return priors
}

function expand(parent, state, policies, neighbors) {
  const moves = []
  const policy = policies[0]
  let max = -Infinity
  for (let i = 0; i < state.board.length; i++) {
    if (!state.board[i] && !isDeadOrCaptured(state.board, neighbors, i)) {
      moves.push(i)
      max = Math.max(max, policy[i])
    }
  }
  // Native search retains all legal moves if every empty point is masked.
  if (!moves.length) {
    for (let i = 0; i < state.board.length; i++) {
      if (!state.board[i]) { moves.push(i); max = Math.max(max, policy[i]) }
    }
  }
  const priors = new Float64Array(moves.length)
  let sum = 0
  for (let i = 0; i < moves.length; i++) {
    priors[i] = Math.exp(policy[moves[i]] - max)
    sum += priors[i]
  }
  for (let i = 0; i < priors.length; i++) priors[i] /= sum
  // Average probabilities rather than logits across root orientations.
  for (let s = 1; s < policies.length; s++) {
    const next = normalizedPriors(moves, policies[s])
    for (let i = 0; i < priors.length; i++) priors[i] += next[i]
  }
  if (policies.length > 1) {
    for (let i = 0; i < priors.length; i++) priors[i] /= policies.length
  }
  // Keep every legal prior, but allocate a search node only when selected.
  parent.children = { moves: Uint16Array.from(moves), priors, nodes: new Array(moves.length) }
  return moves.length
}

export function stateFromPosition(position) {
  // Restore the native Hex representation after swap; inference values remain
  // relative to the player to move in both native and displayed coordinates.
  const swapped = position.game === "hex" && position.moves[1] === "swap"
  const size = position.boardSize
  const board = new Uint8Array(size * size)
  if (position.game === "y") {
    for (let row = 0; row < size; row++) {
      for (let col = size - row; col < size; col++) board[row * size + col] = 3
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

export function terminalValue(state, game) {
  const winner = connected(state.board, state.size, 1, game)
    ? 1 : connected(state.board, state.size, 2, game) ? 2 : 0
  return winner ? (winner === state.toPlay ? 1 : -1) : null
}

export class Search {
  constructor(position, network, wideRootNoise) {
    this.game = position.game
    this.swapped = this.game === "hex" && position.moves[1] === "swap"
    this.state = stateFromPosition(position)
    this.rootSymmetry = rootSymmetryPruning(this.state, this.game)
    this.captureNeighbors = captureNeighbors(position.boardSize, this.game)
    this.network = network
    this.wideRootNoise = wideRootNoise
    this.root = node()
    this.edgeCount = 0
    this.visitLimit = visitLimit(position.boardSize, this.game)
  }

  backup(path, value) {
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visits++
      path[i].sum += value
      value = -value
    }
  }

  async step(cancelled) {
    const pending = []
    const fullBatch = this.network.graph?.searchBatchSize ?? 1
    const batchSize = this.root.visits === 0 ? 1
      : Math.min(fullBatch, Math.ceil(Math.sqrt(fullBatch * this.root.visits)))
    const limit = Math.min(batchSize, this.visitLimit - this.root.visits)
    try {
      for (let i = 0; i < limit; i++) {
        let current = this.root, state = this.state
        const path = [current]
        while (current.children?.moves.length) {
          const isRoot = current === this.root
          current = select(current, isRoot, this.wideRootNoise, isRoot ? this.rootSymmetry.duplicates : null)
          if (!current) break
          state = play(state, current.move)
          path.push(current)
        }
        // The root initially needs one evaluation; a pending leaf cannot be
        // expanded until its result arrives. Submit a partial batch if needed.
        if (!current || (current.inFlight && !current.children)) break
        let value = current.terminal
        if (value === null) {
          value = terminalValue(state, this.game)
          current.terminal = value
        }
        if (value !== null) {
          this.backup(path, value)
          if (current === this.root) break
        } else {
          for (const ancestor of path) ancestor.inFlight++
          pending.push({ current, state, path })
        }
      }
      if (!pending.length || cancelled()) return false
      const atRoot = pending.length === 1 && pending[0].current === this.root
      const inputs = atRoot
        ? symmetriesFor(this.game).map((symmetry) => ({ ...pending[0].state, symmetry }))
        : pending.map(({ state }) => state)
      const evaluations = await this.network.evaluate(inputs, cancelled)
      if (cancelled()) return true
      pending.forEach(({ current, state, path }, i) => {
        const policies = atRoot ? evaluations.map((evaluation) => evaluation.policy) : [evaluations[i].policy]
        const value = atRoot
          ? evaluations.reduce((sum, evaluation) => sum + valueFromLogits(evaluation.value), 0) / evaluations.length
          : valueFromLogits(evaluations[i].value)
        current.initialValue = value
        this.edgeCount += expand(current, state, policies, this.captureNeighbors)
        this.backup(path, value)
      })
      return true
    } finally {
      for (const { path } of pending) {
        for (const ancestor of path) ancestor.inFlight--
      }
    }
  }

  report() {
    const children = this.root.children
    const displayMove = (move) => this.swapped
      ? (move % this.state.size) * this.state.size + Math.floor(move / this.state.size)
      : move
    const candidates = children ? Array.from(children.moves, (move, i) => {
      if (this.rootSymmetry.duplicates[move]) return null
      const child = children.nodes[i]
      const visits = child?.visits || 0
      const pv = visits ? [move] : null
      let current = child
      while (pv && pv.length < PV_LENGTH && current?.children) {
        const branches = current.children
        let best = -1, bestVisits = 0
        for (let j = 0; j < branches.moves.length; j++) {
          const count = branches.nodes[j]?.visits || 0
          if (count > bestVisits || (count > 0 && count === bestVisits && branches.priors[j] > branches.priors[best])) {
            best = j
            bestVisits = count
          }
        }
        if (best < 0) break
        pv.push(branches.moves[best])
        current = branches.nodes[best]
      }
      return {
        move,
        visits, prior: children.priors[i],
        winrate: visits ? (1 - child.sum / visits) / 2 : null,
        pv,
      }
    }).filter(Boolean) : []
    candidates.sort((a, b) => b.visits - a.visits || b.prior - a.prior)
    // Mirror analysis, not search visits: equivalent cells share one subtree.
    // Transform the entire PV before applying the editor's swap coordinates.
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
      winrate: this.root.visits ? (1 + this.root.sum / this.root.visits) / 2 : null,
      terminal: this.root.terminal !== null, candidates: displayed,
    }
  }
}
