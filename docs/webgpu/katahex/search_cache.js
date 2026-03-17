// SPDX-License-Identifier: MIT
import { Search } from "./search.js"
import { reachableBytes } from "./search_graph.js"

const MAX_CACHED_SEARCHES = 32
const CACHE_MEMORY_BYTES = 192 * 1024 * 1024

export function trimSearchCache(searches, activeKey, reservedBytes = 0) {
  let bytes = reservedBytes
  for (const session of searches.values()) bytes += session.search.estimatedBytes
  while (searches.size > MAX_CACHED_SEARCHES || bytes > CACHE_MEMORY_BYTES) {
    let victimKey = null, victimVisits = Infinity
    for (const [key, session] of searches) {
      if (key === activeKey) continue
      const count = session.search.root.visits
      if (count < victimVisits) { victimKey = key; victimVisits = count }
    }
    if (victimKey === null) break
    bytes -= searches.get(victimKey).search.estimatedBytes
    searches.delete(victimKey)
  }
}

export function searchSession(searches, position, network, wideRootNoise, create = true) {
  let session = searches.get(position.key)
  let inherited = null
  for (const candidate of searches.values()) {
    if (candidate === session) continue
    const subtree = candidate.search.subtree(position)
    if (subtree && subtree.visits > (inherited?.visits ?? session?.search.root.visits ?? 0)) {
      inherited = subtree
    }
  }
  if (inherited || (!session && create)) {
    const search = new Search(position, network, wideRootNoise)
    if (inherited) {
      searches.delete(position.key)
      session = null
    }
    const reservedBytes = inherited
      ? reachableBytes(inherited.root, inherited.transform, search.rootSymmetry.duplicates)
      : search.estimatedBytes
    trimSearchCache(searches, position.key, reservedBytes)
    if (inherited) search.inherit(inherited)
    session = { search }
    searches.set(position.key, session)
    trimSearchCache(searches, position.key)
  }
  return session
}
