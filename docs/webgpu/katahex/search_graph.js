import { candidateBytes, coldCandidateCount, expandedCandidateBytes, packCandidates } from "./search_candidates.js"

// Includes objects, typed-array views, map entries, and engine allocation slack.
const NODE_BYTES = 1024

export function graphBytes(nodes, policyBytes, childCapacity, keyLength) {
  return nodes * (NODE_BYTES + keyLength * 2) + policyBytes + childCapacity * 16
}

export function positionKey({ board, toPlay }) {
  const packed = new Uint8Array(1 + Math.ceil(board.length / 4))
  packed[0] = toPlay
  for (let i = 0; i < board.length; i += 4) {
    packed[1 + (i >> 2)] = board[i] | (board[i + 1] << 2) | (board[i + 2] << 4) | (board[i + 3] << 6)
  }
  return String.fromCharCode(...packed)
}

// A legal play changes the side to move and sets one zero-valued two-bit cell.
export function childKey(key, move, toPlay) {
  const byte = 1 + (move >> 2)
  const value = key.charCodeAt(byte) | (toPlay << ((move & 3) * 2))
  return String.fromCharCode(3 - toPlay) + key.slice(1, byte) +
    String.fromCharCode(value) + key.slice(byte + 1)
}

function transformedKey(key, mapping) {
  const packed = new Uint8Array(key.length)
  packed[0] = key.charCodeAt(0)
  for (let i = 0; i < mapping.length; i++) {
    const point = mapping[i]
    const color = (key.charCodeAt(1 + (i >> 2)) >> ((i & 3) * 2)) & 3
    packed[1 + (point >> 2)] |= color << ((point & 3) * 2)
  }
  return String.fromCharCode(...packed)
}

export function createNode(key) {
  return {
    key,
    visits: 0,
    value: 0,
    valueSq: 0,
    drawrate: 0,
    weight: 0,
    weightSq: 0,
    inFlight: 0,
    initialValue: 0,
    initialDrawrate: 0,
    // Null before evaluation, a compact policy for leaves, or ranked candidates with links.
    children: null,
    coldMapping: null,
    terminal: null,
  }
}

export function retainedVisits(root, transform, duplicates) {
  if (!root.children) return 0
  if (root.children instanceof ArrayBuffer) return root.visits
  const { moves, visits, nodes } = root.children
  let retained = 1, filtered = false
  for (let i = 0; i < nodes.length; i++) {
    if (!nodes[i]) continue
    if (duplicates[transform(moves[i])]) filtered = true
    else retained += visits[i]
  }
  return filtered ? retained : root.visits
}

export function reachableBytes(root, points, transform, duplicates) {
  const seen = new Set(), mappings = new Set()
  let policyBytes = 0, capacity = 0
  let remapsMoves = false
  for (let point = 0; point < points && !remapsMoves; point++) remapsMoves = transform(point) !== point
  const visit = (node) => {
    if (seen.has(node)) return
    seen.add(node)
    const children = node.children
    if (!children) return
    if (children instanceof ArrayBuffer) {
      if (node === root) policyBytes += expandedCandidateBytes(coldCandidateCount(children), points)
      else {
        policyBytes += children.byteLength
        if (remapsMoves || node.coldMapping) mappings.add(node.coldMapping)
      }
      return
    }
    policyBytes += candidateBytes(children)
    capacity += children.nodes.length
    for (let i = 0; i < children.nodes.length; i++) {
      if (!children.nodes[i] || (node === root && duplicates[transform(children.moves[i])])) continue
      visit(children.nodes[i])
    }
  }
  visit(root)
  policyBytes += mappings.size * points * Uint16Array.BYTES_PER_ELEMENT
  return graphBytes(seen.size, policyBytes, capacity, root.key.length)
}

export function copyGraph(root, points, transform, duplicates) {
  const copies = new Map(), nodes = new Map()
  const mapping = Uint16Array.from({ length: points }, (_, point) => transform(point))
  const identity = mapping.every((point, i) => point === i)
  const compositions = identity ? null : new Map([[null, mapping]])
  const retainedMappings = new Set()
  let policyBytes = 0, childCapacity = 0
  const copy = (source) => {
    if (copies.has(source)) return copies.get(source)
    const key = identity ? source.key : transformedKey(source.key, mapping)
    const target = { ...source, key, inFlight: 0 }
    if (source === root) target.visits = retainedVisits(source, transform, duplicates)
    copies.set(source, target)
    nodes.set(key, target)
    if (source.children) {
      const children = source.children
      if (children instanceof ArrayBuffer) {
        const original = source.coldMapping
        if (compositions) {
          if (!compositions.has(original)) compositions.set(original, original.map((point) => mapping[point]))
          target.coldMapping = compositions.get(original)
        }
        if (source !== root && target.coldMapping) retainedMappings.add(target.coldMapping)
        policyBytes += children.byteLength
        return target
      }
      const candidates = identity
        ? children
        : packCandidates(children.moves, children.priors, points, { mapping })
      const { moves, priors } = candidates
      const visits = children.visits.length ? children.visits.slice(0, children.nodes.length) : EMPTY_VISITS
      const branches = new Array(children.nodes.length), order = source === root ? [] : null
      const count = children.order?.length ?? children.nodes.length
      for (let j = 0; j < count; j++) {
        const i = children.order ? children.order[j] : j
        if (!children.nodes[i]) continue
        if (source === root && duplicates[moves[i]]) {
          visits[i] = 0
          continue
        }
        branches[i] = copy(children.nodes[i])
        order?.push(i)
      }
      target.children = { moves, priors, visits, nodes: branches, order }
      policyBytes += moves.buffer.byteLength
      childCapacity += visits.length
    }
    return target
  }
  const copiedRoot = copy(root)
  // Keep the original map charge as cold nodes materialize.
  policyBytes += retainedMappings.size * mapping.byteLength
  return { root: copiedRoot, nodes, policyBytes, childCapacity }
}

export const EMPTY_VISITS = new Float64Array(0)
