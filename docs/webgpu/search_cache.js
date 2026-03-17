import { Search, stateFromPosition, subtreeQuery, usesSwappedHexCoordinates } from "./search.js"
import { positionKey, reachableBytes } from "./search_graph.js"

const MAX_CACHED_SEARCHES = 32
const CACHE_MEMORY_BYTES = 192 * 1024 * 1024

function cachedBytes(searches) {
  let bytes = 0
  for (const session of searches.values()) bytes += session.search.estimatedBytes
  return bytes
}

export function trimSearchCache(searches, activeKey, reservedBytes = 0) {
  let bytes = reservedBytes + cachedBytes(searches)
  while (searches.size > MAX_CACHED_SEARCHES || bytes > CACHE_MEMORY_BYTES) {
    let victimKey = null, victimVisits = Infinity
    for (const [key, session] of searches) {
      if (key === activeKey) continue
      const count = session.search.root.visits
      if (count < victimVisits) {
        victimKey = key
        victimVisits = count
      }
    }
    if (victimKey === null) break
    bytes -= searches.get(victimKey).search.estimatedBytes
    searches.delete(victimKey)
  }
}

function findCachedSearch(searches, position) {
  // Search depends on the visible stones and side to play. Different move
  // orders can share its graph when their display coordinates also match.
  const context = `${position.game}:${position.boardSize}:${Number(usesSwappedHexCoordinates(position))}`
  const state = stateFromPosition(position)
  const key = context + ":" + positionKey(state)
  const session = searches.get(key)
  const sessionVisits = session?.search.root.visits ?? 0
  let inherited = null
  let query = null
  for (const candidate of searches.values()) {
    if (candidate === session) continue
    if (candidate.search.game !== position.game || candidate.search.state.size !== position.boardSize) continue
    query ??= subtreeQuery(state, position.game)
    const subtree = candidate.search.subtree(query)
    if (!subtree) continue
    const bestVisits = inherited?.visits ?? sessionVisits
    // Prefer an exact orientation once it has a searched reply.
    const bestExactReply = inherited?.exactReply ?? (sessionVisits > 1)
    if (subtree.exactReply === bestExactReply ? subtree.visits > bestVisits : subtree.exactReply) {
      inherited = { ...subtree, sourceKey: candidate.key, sourceBytes: candidate.search.estimatedBytes }
    }
  }
  return { key, session, inherited }
}

export function cachedSearchReport(searches, position, network, wideRootNoise, memoryLimit) {
  const { key, session, inherited } = findCachedSearch(searches, position)
  if (inherited && inherited.sourceBytes <= CACHE_MEMORY_BYTES) {
    // Keep the graph that backs the displayed replies available for Start and siblings.
    trimSearchCache(searches, inherited.sourceKey)
    const report = new Search(position, network, wideRootNoise).reportInherited(inherited)
    return { ...report, finished: report.terminal }
  }
  // A parent that cannot fit gives way to a resumable search at the displayed position.
  const restored = inherited ? installSearch(searches, position, network, wideRootNoise, key, inherited) : session
  if (!inherited) trimSearchCache(searches, key)
  if (!restored) return null
  const { search } = restored
  search.memoryLimit = memoryLimit
  return { ...search.report(), finished: search.finishedReason !== null }
}

export function searchSession(searches, position, network, wideRootNoise) {
  const { key, session, inherited } = findCachedSearch(searches, position)
  if (session && !inherited) return session
  return installSearch(searches, position, network, wideRootNoise, key, inherited)
}

function installSearch(searches, position, network, wideRootNoise, key, inherited) {
  const search = new Search(position, network, wideRootNoise)
  if (inherited) searches.delete(key)
  // Inheritance can warm one cold root and add one coordinate map.
  const upperBound = inherited ? inherited.sourceBytes + search.expansionBytes : search.estimatedBytes
  if (searches.size > MAX_CACHED_SEARCHES || cachedBytes(searches) + upperBound > CACHE_MEMORY_BYTES) {
    const reservedBytes = inherited
      ? reachableBytes(inherited.root, search.state.board.length, inherited.transform)
      : search.estimatedBytes
    trimSearchCache(searches, key, reservedBytes)
  }
  if (inherited) search.inherit(inherited)
  const session = { key, search }
  searches.set(key, session)
  trimSearchCache(searches, key)
  return session
}
