import { candidateBytes, coldCandidateCount, expandedCandidateBytes, packCandidates } from "./search_candidates.js"

// Includes objects, typed-array views, map entries, and engine allocation slack.
const NODE_BYTES = 1024

export function graphBytes(nodes, policyBytes, childCapacity, keyChars) {
  return nodes * NODE_BYTES + keyChars * 2 + policyBytes + childCapacity * 16
}

export function positionKey({ board, toPlay }) {
  const packed = new Uint8Array(1 + Math.ceil(board.length / 4))
  packed[0] = toPlay
  for (let i = 0; i < board.length; i += 4) {
    packed[1 + (i >> 2)] = board[i] | (board[i + 1] << 2) | (board[i + 2] << 4) | (board[i + 3] << 6)
  }
  return String.fromCharCode(...packed)
}

// The root board is common to every graph node. A sorted colored move set
// identifies every descendant exactly, including transpositions.
export function childKey(key, move, toPlay) {
  const stone = 2 * move + toPlay
  let i = key.length
  while (i > 0 && key.charCodeAt(i - 1) > stone) i--
  return key.slice(0, i) + String.fromCharCode(stone) + key.slice(i)
}

export function relativeKey(base, query) {
  if (base.length !== query.length) return null
  let result = ""
  for (let i = 1; i < base.length; i++) {
    const a = base.charCodeAt(i), b = query.charCodeAt(i)
    if (a === b) continue
    for (let k = 0; k < 4; k++) {
      const before = (a >>> (2 * k)) & 3, after = (b >>> (2 * k)) & 3
      if (before === after) continue
      if (before || (after !== 1 && after !== 2)) return null
      result += String.fromCharCode(8 * (i - 1) + 2 * k + after)
    }
  }
  const side = base.charCodeAt(0)
  return query.charCodeAt(0) === (result.length % 2 ? 3 - side : side) ? result : null
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

export function reachableBytes(root, points, transform) {
  const seen = new Set(), mappings = new Set()
  let policyBytes = 0, capacity = 0, keyChars = 0
  let remapsMoves = false
  for (let point = 0; point < points && !remapsMoves; point++) remapsMoves = transform(point) !== point
  const visit = (node) => {
    if (seen.has(node)) return
    seen.add(node)
    keyChars += node.key.length - root.key.length
    const children = node.children
    if (!children) return
    const cold = children instanceof ArrayBuffer ? children : children.cold
    if (cold && node === root) policyBytes += expandedCandidateBytes(coldCandidateCount(cold), points)
    else {
      policyBytes += candidateBytes(children)
      if (cold && (remapsMoves || node.coldMapping)) mappings.add(node.coldMapping)
    }
    if (children instanceof ArrayBuffer) return
    capacity += children.nodes.length
    for (let i = 0; i < children.nodes.length; i++) {
      if (!children.nodes[i]) continue
      visit(children.nodes[i])
    }
  }
  visit(root)
  policyBytes += mappings.size * points * Uint16Array.BYTES_PER_ELEMENT
  return graphBytes(seen.size, policyBytes, capacity, 1 + Math.ceil(points / 4) + keyChars)
}

export function copyGraph(root, points, transform, rootToPlay, copyDescendants) {
  const nodes = new Map()
  const mapping = Uint16Array.from({ length: points }, (_, point) => transform(point))
  const identity = mapping.every((point, i) => point === i)
  const compositions = identity ? null : new Map([[null, mapping]])
  const retainedMappings = new Set()
  let policyBytes = 0, childCapacity = 0, keyChars = 0
  const copy = (source, key) => {
    const existing = nodes.get(key)
    if (existing) return existing
    const target = { ...source, key, inFlight: 0 }
    nodes.set(key, target)
    keyChars += key.length
    if (source.children) {
      const children = source.children
      if (children instanceof ArrayBuffer || children.cold) {
        const original = source.coldMapping
        if (compositions) {
          if (!compositions.has(original)) compositions.set(original, original.map((point) => mapping[point]))
          target.coldMapping = compositions.get(original)
        }
        if (source !== root && target.coldMapping) retainedMappings.add(target.coldMapping)
        if (children instanceof ArrayBuffer) {
          policyBytes += children.byteLength
          return target
        }
      }
      const candidates = identity
        ? children
        : packCandidates(children.moves, children.priors, points, { mapping })
      const { moves, priors } = candidates
      const visits = children.visits.length ? children.visits.slice(0, children.nodes.length) : EMPTY_VISITS
      const branches = new Array(children.nodes.length), order = source === root ? [] : null
      const count = children.order?.length ?? children.nodes.length
      const toPlay = key.length % 2 ? 3 - rootToPlay : rootToPlay
      for (let j = 0; j < count; j++) {
        const i = children.order ? children.order[j] : j
        if (!children.nodes[i]) continue
        branches[i] = copyDescendants ? copy(children.nodes[i], childKey(key, moves[i], toPlay)) : children.nodes[i]
        order?.push(i)
      }
      target.children = { moves, priors, visits, nodes: branches, order, cold: children.cold }
      policyBytes += candidateBytes(target.children)
      childCapacity += visits.length
    }
    return target
  }
  const copiedRoot = copy(root, "")
  // Keep the original map charge as cold nodes materialize.
  policyBytes += retainedMappings.size * mapping.byteLength
  return { root: copiedRoot, nodes, policyBytes, childCapacity, keyChars }
}

export const EMPTY_VISITS = new Float64Array(0)
