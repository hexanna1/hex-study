// SPDX-License-Identifier: MIT
// Y-family inferior-move certificates and virtual connections.
import "../../benty_geometry.js"
import { canonicalConnectionBoard } from "./symmetry.js"

const DIRECTIONS = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
const DEAD = 1

function neighborhood(present) {
  for (const count of [6, 5, 4, 3, 2]) {
    for (let start = 0; start < 6; start++) {
      let path = 0
      for (let i = 0; i < count; i++) path |= 1 << ((start + i) % 6)
      if (path !== present) continue
      const adjacent = new Uint8Array(6), end = (start + count - 1) % 6
      for (let i = 0; i < count - 1; i++) {
        const a = (start + i) % 6, b = (start + i + 1) % 6
        adjacent[a] |= 1 << b
        adjacent[b] |= 1 << a
      }
      const contacts = new Uint8Array(6)
      if (count >= 5) {
        adjacent[start] |= 1 << end
        adjacent[end] |= 1 << start
        return { sides: 0, contacts, adjacent }
      }
      contacts[start] = 1
      contacts[end] = count === 4 ? 1 : 2
      return { sides: count === 4 ? 1 : 3, contacts, adjacent }
    }
  }
  return null
}

function preservesConnections(colored, { sides, contacts, adjacent }) {
  if (!colored) return true
  let connected = colored & -colored
  while (true) {
    let expanded = connected
    for (let d = 0; d < 6; d++) if (connected & (1 << d)) expanded |= adjacent[d] & colored
    if (expanded === connected) break
    connected = expanded
  }
  if (connected !== colored) return false
  let touched = 0
  for (let d = 0; d < 6; d++) if (colored & (1 << d)) touched |= contacts[d]
  return (touched & sides) === sides
}

function deadPattern(pattern) {
  let present = 0, red = 0, blue = 0, empty = 0
  for (let d = 0; d < 6; d++) {
    const color = (pattern >> (2 * d)) & 3
    if (color === 3) continue // An unused slot, never an owned edge or wildcard.
    present |= 1 << d
    if (color === 1) red |= 1 << d
    else if (color === 2) blue |= 1 << d
    else empty |= 1 << d
  }
  const geometry = neighborhood(present)
  if (!geometry) return false
  // Every completion must preserve both colors' connectivity and side contacts.
  let completion = empty
  while (true) {
    if (!preservesConnections(red | completion, geometry) ||
        !preservesConnections(blue | (empty & ~completion), geometry)) return false
    if (!completion) break
    completion = (completion - 1) & empty
  }
  return true
}

function buildCaptureTable() {
  const table = new Uint8Array(4096)
  for (let pattern = 0; pattern < table.length; pattern++) {
    if (deadPattern(pattern)) table[pattern] = DEAD
  }
  for (let pattern = 0; pattern < table.length; pattern++) {
    if (table[pattern] === DEAD) continue
    for (let d = 0; d < 6; d++) {
      if (((pattern >> (2 * d)) & 3) !== 0) continue
      for (const color of [1, 2]) {
        // One real move making the center dead certifies an inferior move.
        if (table[pattern | (color << (2 * d))] === DEAD) {
          table[pattern] = 2
        }
      }
    }
  }
  return table
}

export const Y_CAPTURE_TABLE = buildCaptureTable()

const bentYNeighbors = new WeakMap()

export function bentYCaptureNeighbors(graph) {
  if (bentYNeighbors.has(graph)) return bentYNeighbors.get(graph)
  const ordered = new Int16Array(graph.neighbors.length * 6).fill(-3)
  for (const point of graph.playable) {
    const near = graph.neighbors[point]
    const start = near.find(next => graph.neighbors[next].filter(adjacent => near.includes(adjacent)).length === 1) ?? near[0]
    let previous = -1, current = start
    for (let d = 0; d < near.length; d++) {
      ordered[point * 6 + d] = current
      const next = graph.neighbors[current].find(adjacent => adjacent !== previous && near.includes(adjacent))
      previous = current
      current = next
    }
  }
  bentYNeighbors.set(graph, ordered)
  return ordered
}

const bentYLinks = new WeakMap()

function bentYConnections(graph) {
  if (bentYLinks.has(graph)) return bentYLinks.get(graph)
  const bridges = Array.from(graph.neighbors, () => [])
  const sideContacts = Array.from(graph.neighbors, () => [])
  for (const point of graph.playable) {
    const near = graph.neighbors[point]
    const candidates = Array.from(new Set(near.flatMap(next => graph.neighbors[next]))).sort((a, b) => a - b)
    for (const next of candidates) {
      if (next === point || near.includes(next)) continue
      const common = near.filter(adjacent => graph.neighbors[next].includes(adjacent))
      for (let i = 0; i < common.length; i++) {
        for (let j = i + 1; j < common.length; j++) bridges[point].push({ next, a: common[i], b: common[j] })
      }
    }
    for (const side of [1, 2, 4]) {
      if (graph.sideMasks[point] & side) continue
      const contacts = near.filter(next => graph.sideMasks[next] & side)
      for (let i = 0; i < contacts.length; i++) {
        for (let j = i + 1; j < contacts.length; j++) sideContacts[point].push({ side, a: contacts[i], b: contacts[j] })
      }
    }
  }
  const boundary = graph.playable.filter(point => graph.sideMasks[point] || sideContacts[point].length)
  const links = { bridges, sideContacts, boundary }
  bentYLinks.set(graph, links)
  return links
}

export function hasSideContact(board, size, color, edge, step, inner) {
  for (let i = 0; i < size; i++) {
    const point = edge + i * step
    if (board[point] === color) return true
    if (i + 1 < size && !board[point] && !board[point + step] && board[inner + i * step] === color) return true
  }
  return false
}

function bentYHasSideContacts(board, color, graph, links) {
  let sides = 0
  for (const point of links.boundary) {
    if (board[point] !== color) continue
    sides |= graph.sideMasks[point]
    for (const { side, a, b } of links.sideContacts[point]) {
      if (!board[a] && !board[b]) sides |= side
    }
    if (sides === 7) return true
  }
  return false
}

export function yStrategicWinner({ board, size }, game) {
  const graph = game === "bentY" ? globalThis.BentYGeometry.topology(size) : null
  const links = graph ? bentYConnections(graph) : null
  let oriented
  for (const color of [1, 2]) {
    // Check the top, left and diagonal sides before orienting or traversing the board.
    const contactsAllSides = graph ? bentYHasSideContacts(board, color, graph, links) :
      hasSideContact(board, size, color, 0, 1, size) &&
      hasSideContact(board, size, color, 0, size, 1) &&
      hasSideContact(board, size, color, size - 1, size - 1, size - 2)
    if (!contactsAllSides) continue
    oriented ??= canonicalConnectionBoard(board, size, game)
    if (yConnected(oriented, size, color, graph, links)) return color
  }
  return 0
}

function yConnected(board, size, color, graph, links) {
  const seen = new Uint8Array(board.length)
  const stack = []
  const playable = (x, y) => x >= 0 && y >= 0 && x + y < size
  for (let root = 0; root < board.length; root++) {
    if (board[root] !== color || seen[root]) continue
    let sides = 0
    stack.push(root)
    seen[root] = 1
    while (stack.length) {
      const move = stack.pop(), x = move % size, y = Math.floor(move / size)
      if (graph) sides |= graph.sideMasks[move]
      else {
        if (y === 0) sides |= 1
        if (x === 0) sides |= 2
        if (x + y === size - 1) sides |= 4
      }
      // Sides are terminal contacts. Disjoint carriers make all replies compatible.
      const edgeContact = (side, a, b) => {
        if (!(sides & side) && !board[a] && !board[b] && !seen[a] && !seen[b]) {
          seen[a] = seen[b] = 2
          sides |= side
        }
      }
      if (links) {
        for (const { side, a, b } of links.sideContacts[move]) edgeContact(side, a, b)
      } else {
        if (y === 1) edgeContact(1, x, x + 1)
        if (x === 1) edgeContact(2, y * size, (y + 1) * size)
        if (x + y === size - 2) edgeContact(4, move + 1, move + size)
      }
      if (sides === 7) return true
      const connect = (next) => {
        if (board[next] === color && !seen[next]) {
          seen[next] = 1
          stack.push(next)
        }
      }
      if (graph) {
        for (const next of graph.neighbors[move]) connect(next)
      } else {
        for (const [dx, dy] of DIRECTIONS) {
          const nx = x + dx, ny = y + dy
          if (playable(nx, ny)) connect(ny * size + nx)
        }
      }
      const bridge = (next, a, b) => {
        if (board[next] === color && !seen[next] && !board[a] && !board[b] && !seen[a] && !seen[b]) {
          seen[a] = seen[b] = 2
          seen[next] = 1
          stack.push(next)
        }
      }
      if (links) {
        for (const { next, a, b } of links.bridges[move]) bridge(next, a, b)
      } else {
        for (let d = 0; d < 6; d++) {
          const [ax, ay] = DIRECTIONS[d], [bx, by] = DIRECTIONS[(d + 1) % 6]
          const nx = x + ax + bx, ny = y + ay + by
          if (playable(nx, ny)) bridge(ny * size + nx, (y + ay) * size + x + ax, (y + by) * size + x + bx)
        }
      }
    }
  }
  return false
}
