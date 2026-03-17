// SPDX-License-Identifier: MIT

// Includes objects, typed-array views, map entries, and engine allocation slack.
const NODE_BYTES = 1024

export function graphBytes(nodes, edgeCount, childCapacity, keyLength) {
  return nodes * (NODE_BYTES + keyLength * 2) + edgeCount * 6 + childCapacity * 16
}

export function positionKey({ board, toPlay }) {
  const packed = new Uint8Array(1 + Math.ceil(board.length / 4))
  packed[0] = toPlay
  for (let i = 0; i < board.length; i += 4) {
    packed[1 + (i >> 2)] = board[i] | (board[i + 1] << 2) | (board[i + 2] << 4) | (board[i + 3] << 6)
  }
  return String.fromCharCode(...packed)
}

function transformedKey(key, mapping) {
  const board = new Uint8Array(mapping.length)
  for (let i = 0; i < mapping.length; i++) board[mapping[i]] = (key.charCodeAt(1 + (i >> 2)) >> ((i & 3) * 2)) & 3
  return positionKey({ board, toPlay: key.charCodeAt(0) })
}

export function createNode(key) {
  return {
    key, visits: 0, value: 0, valueSq: 0, weight: 0, weightSq: 0,
    inFlight: 0, initialValue: 0, children: null, terminal: null,
  }
}

export function retainedVisits(root, transform, duplicates) {
  if (!root.children) return 0
  const { moves, visits, nodes } = root.children
  let retained = 1, filtered = false
  for (let i = 0; i < nodes.length; i++) {
    if (!nodes[i]) continue
    if (duplicates[transform(moves[i])]) filtered = true
    else retained += visits[i]
  }
  return filtered ? retained : root.visits
}

export function reachableBytes(root, transform, duplicates) {
  const seen = new Set()
  let edges = 0, capacity = 0
  const visit = (node) => {
    if (seen.has(node)) return
    seen.add(node)
    const children = node.children
    if (!children) return
    edges += children.moves.length
    capacity += children.nodes.length
    for (let i = 0; i < children.nodes.length; i++) {
      if (!children.nodes[i] || (node === root && duplicates[transform(children.moves[i])])) continue
      visit(children.nodes[i])
    }
  }
  visit(root)
  return graphBytes(seen.size, edges, capacity, root.key.length)
}

export function copyGraph(root, points, transform, duplicates) {
  const copies = new Map(), nodes = new Map()
  const mapping = Uint16Array.from({ length: points }, (_, point) => transform(point))
  const identity = mapping.every((point, i) => point === i)
  let edgeCount = 0, childCapacity = 0
  const copy = (source) => {
    if (copies.has(source)) return copies.get(source)
    const key = identity ? source.key : transformedKey(source.key, mapping)
    const target = { ...source, key, inFlight: 0 }
    if (source === root) target.visits = retainedVisits(source, transform, duplicates)
    copies.set(source, target)
    nodes.set(key, target)
    if (source.children) {
      const children = source.children
      const candidates = identity ? children : packCandidates(children.moves.map((point) => mapping[point]), children.priors)
      const { moves, priors } = candidates
      const visits = children.visits.length ? children.visits.slice(0, children.nodes.length) : EMPTY_VISITS
      const branches = new Array(children.nodes.length), order = source === root ? [] : null
      const count = children.order?.length ?? children.nodes.length
      for (let j = 0; j < count; j++) {
        const i = children.order ? children.order[j] : j
        if (!children.nodes[i]) continue
        if (source === root && duplicates[moves[i]]) { visits[i] = 0; continue }
        branches[i] = copy(children.nodes[i])
        order?.push(i)
      }
      target.children = { moves, priors, visits, nodes: branches, order }
      edgeCount += moves.length
      childCapacity += visits.length
    }
    return target
  }
  const copiedRoot = copy(root)
  return { root: copiedRoot, nodes, edgeCount, childCapacity }
}

export const EMPTY_VISITS = new Float64Array(0)

// Candidate data is immutable between root evaluations and shared by snapshots.
export function packCandidates(moves, priors) {
  const buffer = new ArrayBuffer(moves.length * 6)
  const packedPriors = new Float32Array(buffer, 0, moves.length)
  const packedMoves = new Uint16Array(buffer, moves.length * 4, moves.length)
  packedPriors.set(priors)
  packedMoves.set(moves)
  return { moves: packedMoves, priors: packedPriors }
}
