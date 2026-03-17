window.createJosekiCodec = (AVAILABLE_FAMILIES) => {
  const { decodeThousandths, readAsciiMagic, readBitplaneValue } = window.HexStudyUI

  const BUNDLE_MAGIC = "HJB"
  const FAMILY_CODE_TO_NAME = { 1: "A", 2: "O" }
  const HEADER_SIZE = 29
  const LOCAL_COUNT_MASK = 0x0f
  const CORE = 0x10
  const TENUKI_RETAINED = 0x20
  const TENUKI_CHILD = 0x40
  const LOCAL_CHILDREN = 0x80

  const LOCAL_BOARD_SIZE = 10
  const LOCAL_DELTA_MAX = { A: 127, O: 64 }

  function localDelta(local, family) {
    const dq = Number(local[0]) - 1
    const dr = Number(local[1]) - 1
    if (family === "O") { return dq * dq - dq * dr + dr * dr }
    return dq * dq + dq * dr + dr * dr
  }

  function normalizedFamily(family) {
    const familyValue = String(family || "") .trim() .toUpperCase()
    if (!AVAILABLE_FAMILIES.includes(familyValue)) { return null }
    return familyValue
  }

  function parseLineForFamily(line, family, { validateRegion = false, rejectDuplicates = false } = {}) {
    const familyValue = normalizedFamily(family)
    if (!familyValue) { return null }
    const raw = String(line || "").replace(/\s+/g, "")
    if (!raw) { return null }
    const match = /^([AO])\[(.*)\]$/.exec(raw)
    if (!match || match[1] !== familyValue) { return null }
    const entries = []
    const occupied = rejectDuplicates ? new Set() : null
    for (const token of match[2].split(":")) {
      if (token === "") {
        entries.push(null)
        continue
      }
      const parts = /^([1-9][0-9]*),([1-9][0-9]*)$/.exec(token)
      if (!parts) { return null }
      const entry = [Number(parts[1]), Number(parts[2])]
      if (validateRegion) {
        const point = localToDisplay(entry, familyValue)
        if (!displayCellInLocalRegion(point.col, point.row, familyValue)) { return null }
      }
      if (occupied) {
        const key = `${entry[0]},${entry[1]}`
        if (occupied.has(key)) { return null }
        occupied.add(key)
      }
      entries.push(entry)
    }
    return { family: familyValue, entries }
  }

  function parseEntries(line, family) { return parseLineForFamily(line, family)?.entries || [] }

  function normalizeLineForFamily(line, family) {
    const parsed = parseLineForFamily(line, family, { validateRegion: true, rejectDuplicates: true })
    if (!parsed || parsed.entries.length === 0) { return "" }
    return formatLine(parsed.family, parsed.entries)
  }

  function decodeLocalMove(rawMove) {
    if (!Number.isInteger(rawMove) || rawMove < 0 || rawMove >= 100) {
      throw new Error("Unsupported joseki local move")
    }
    return [Math.floor(rawMove / 10) + 1, (rawMove % 10) + 1]
  }

  function retainedCount(control) {
    return (control & LOCAL_COUNT_MASK) + ((control & TENUKI_RETAINED) >>> 5)
  }

  function normalizeLoadedData(rawBuffer) {
    if (!(rawBuffer instanceof ArrayBuffer) || rawBuffer.byteLength < HEADER_SIZE) {
      throw new Error("Unsupported joseki data format")
    }
    const view = new DataView(rawBuffer)
    if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) { throw new Error("Unsupported joseki data format") }
    const family = FAMILY_CODE_TO_NAME[view.getUint8(3)]
    if (!AVAILABLE_FAMILIES.includes(family)) { throw new Error("Unsupported joseki data format") }
    const boardSize = view.getUint8(4)
    const nodeCount = view.getUint32(5, true)
    const localRowCount = view.getUint32(9, true)
    const linkCount = view.getUint32(13, true)
    const totalContinuations = view.getUint32(17, true)
    const firstDropCount = view.getUint32(21, true)
    const siblingDropCount = view.getUint32(25, true)
    const rootIndex = nodeCount - 1
    const bytes = new Uint8Array(rawBuffer)
    const localMoveOffset = HEADER_SIZE + nodeCount
    const firstDropBitmapOffset = localMoveOffset + localRowCount
    const controls = bytes.slice(HEADER_SIZE, localMoveOffset)
    const localMoves = bytes.slice(localMoveOffset, firstDropBitmapOffset)
    let firstDropOffset = firstDropBitmapOffset + Math.ceil(nodeCount / 8)
    let siblingDropOffset = firstDropOffset + firstDropCount
    const tenukiZeroCount = nodeCount - firstDropCount
    const tenukiZeroLowOffset = siblingDropOffset + siblingDropCount
    const tenukiZeroHighOffset = tenukiZeroLowOffset + tenukiZeroCount
    const tenukiNonzeroLowOffset = tenukiZeroHighOffset + Math.ceil(tenukiZeroCount / 4)
    const tenukiNonzeroHighOffset = tenukiNonzeroLowOffset + firstDropCount
    const redirectBitmapOffset = tenukiNonzeroHighOffset + Math.ceil(firstDropCount / 4)
    let redirectOffset = redirectBitmapOffset + Math.ceil(linkCount / 8)

    const NodeIndexArray = nodeCount <= 0xffff ? Uint16Array : Uint32Array
    const OffsetArray = Math.max(linkCount, localRowCount) <= 0xffff ? Uint16Array : Uint32Array
    const CountArray = totalContinuations < 0xffff ? Uint16Array : Uint32Array
    const localStarts = new OffsetArray(nodeCount)
    const linkStarts = new OffsetArray(nodeCount)
    const childTargets = new NodeIndexArray(linkCount)
    const treeSpans = new NodeIndexArray(nodeCount)
    const localDrops = new Uint8Array(localRowCount)
    const tenukiDrops = new Uint16Array(nodeCount)
    const continuationCounts = new CountArray(nodeCount)
    const coreCounts = new CountArray(nodeCount)
    const localCoreDescendants = new CountArray(nodeCount)
    const leafCounts = new CountArray(nodeCount)
    const localLeafDescendants = new CountArray(nodeCount)

    let localStart = 0
    let link = 0
    let tenukiZeroIndex = 0
    let tenukiNonzeroIndex = 0
    // Children precede their parent. Read each edge once to fill the adjacency
    // index and derive all navigation counts, including the random pools.
    for (let node = 0; node < nodeCount; node += 1) {
      const control = controls[node]
      const localCount = control & LOCAL_COUNT_MASK
      const edgeCount = retainedCount(control)
      const localLinkCount = control & LOCAL_CHILDREN ? localCount : 0
      const linkEnd = link + localLinkCount + ((control & TENUKI_CHILD) >>> 6)
      const localLinkEnd = link + localLinkCount
      localStarts[node] = localStart
      linkStarts[node] = link

      const hasFirstDrop = bytes[firstDropBitmapOffset + (node >>> 3)] & (1 << (node & 7))
      let drop = hasFirstDrop ? bytes[firstDropOffset++] : 0
      if (localCount > 0) {
        localDrops[localStart++] = drop
        for (let edge = 1; edge < localCount; edge += 1) {
          drop = (drop + bytes[siblingDropOffset++]) & 0xff
          localDrops[localStart++] = drop
        }
      }
      if (hasFirstDrop) {
        const high = readBitplaneValue(view, tenukiNonzeroHighOffset, firstDropCount, tenukiNonzeroIndex, 2)
        tenukiDrops[node] = bytes[tenukiNonzeroLowOffset + tenukiNonzeroIndex++] | (high << 8)
      } else {
        const high = readBitplaneValue(view, tenukiZeroHighOffset, tenukiZeroCount, tenukiZeroIndex, 2)
        tenukiDrops[node] = bytes[tenukiZeroLowOffset + tenukiZeroIndex++] | (high << 8)
      }

      let continuations = edgeCount
      let cores = node < rootIndex && (control & CORE) ? 1 : 0
      let localCores = 0
      let leaves = edgeCount === 0 ? 1 : edgeCount - (linkEnd - link)
      let localLeaves = localCount - localLinkCount
      let treeChild = node - 1
      for (; link < linkEnd; link += 1) {
        let target
        const redirect = bytes[redirectBitmapOffset + (link >>> 3)] & (1 << (link & 7))
        if (redirect) {
          target = 0
          let shift = 0
          let next
          do {
            next = bytes[redirectOffset++]
            target |= (next & 0x7f) << shift
            shift += 7
          } while (next & 0x80)
        } else {
          target = treeChild
          treeChild -= treeSpans[target]
        }
        if (target < 0 || target >= node) { throw new Error("Joseki topology is invalid") }
        childTargets[link] = target
        continuations += continuationCounts[target]
        cores += coreCounts[target]
        leaves += leafCounts[target]
        localCores += localCoreDescendants[target]
        localLeaves += localLeafDescendants[target]
        // A shared node can end both a local and a tenuki line. Its own entry
        // belongs to the local-only pool only along a local incoming edge.
        if (link < localLinkEnd) {
          if (controls[target] & CORE) { localCores += 1 }
          if (retainedCount(controls[target]) === 0) { localLeaves += 1 }
        }
      }
      treeSpans[node] = node - treeChild
      continuationCounts[node] = continuations
      coreCounts[node] = cores
      localCoreDescendants[node] = localCores
      leafCounts[node] = leaves
      localLeafDescendants[node] = localLeaves
    }
    if (
      treeSpans[rootIndex] !== nodeCount || localStart !== localRowCount || link !== linkCount ||
      redirectOffset !== bytes.length || continuationCounts[rootIndex] !== totalContinuations
    ) { throw new Error("Joseki bundle size mismatch") }

    return {
      family,
      board_size: boardSize,
      controls,
      nodeCount,
      rootIndex,
      localStarts,
      linkStarts,
      childTargets,
      localMoves,
      localDrops,
      tenukiDrops,
      continuationCounts,
      coreCounts,
      localCoreDescendants,
      leafCounts,
      localLeafDescendants,
    }
  }

  function targetAt(data, nodeIndex, edge) {
    const control = data.controls[nodeIndex]
    const localCount = control & LOCAL_COUNT_MASK
    const start = data.linkStarts[nodeIndex]
    if (edge < localCount) {
      return control & LOCAL_CHILDREN ? data.childTargets[start + edge] : null
    }
    if (!(control & TENUKI_CHILD)) { return null }
    return data.childTargets[start + (control & LOCAL_CHILDREN ? localCount : 0)]
  }

  function appendMove(line, family, local) {
    const move = local ? `${local[0]},${local[1]}` : ""
    return line ? `${line.slice(0, -1)}:${move}]` : `${family}[${move}]`
  }

  function decodeJosekiNode(data, nodeIndex, line) {
    const control = data.controls[nodeIndex]
    const localCount = control & LOCAL_COUNT_MASK
    const localStart = data.localStarts[nodeIndex]
    const candidates = []
    for (let edge = 0; edge < localCount; edge += 1) {
      const local = decodeLocalMove(data.localMoves[localStart + edge])
      candidates.push({
        kind: "local",
        local,
        stone_fraction: decodeThousandths(1000 - data.localDrops[localStart + edge]),
        retained: true,
        childIndex: targetAt(data, nodeIndex, edge),
        childLine: appendMove(line, data.family, local),
      })
    }
    if (nodeIndex !== data.rootIndex) {
      const tenukiDrop = data.tenukiDrops[nodeIndex]
      const retained = (control & TENUKI_RETAINED) !== 0
      candidates.push({
        kind: "tenuki",
        stone_fraction: decodeThousandths(1000 - tenukiDrop),
        retained,
        childIndex: targetAt(data, nodeIndex, localCount),
        childLine: retained ? appendMove(line, data.family, null) : null,
      })
    }
    return { index: nodeIndex, is_core: (control & CORE) !== 0, line, candidates }
  }

  function findJosekiNode(data, line, entries) {
    if (!data || (line && entries.length === 0)) { return null }
    let node = data.rootIndex
    for (const entry of entries) {
      const localCount = data.controls[node] & LOCAL_COUNT_MASK
      let edge = localCount
      if (entry) {
        const code = (entry[0] - 1) * 10 + entry[1] - 1
        const start = data.localStarts[node]
        edge = 0
        while (edge < localCount && data.localMoves[start + edge] !== code) { edge += 1 }
        if (edge === localCount) { return null }
      }
      node = targetAt(data, node, edge)
      if (node === null) { return null }
    }
    return decodeJosekiNode(data, node, line)
  }

  function formatLine(family, entries) {
    if (!entries.length) { return "" }
    return `${family}[${entries.map((entry) => (entry ? `${entry[0]},${entry[1]}` : "")).join(":")}]`
  }

  function localToDisplay(local, family) {
    const x = Number(local[0])
    const y = Number(local[1])
    if (family === "O") { return { col: y, row: LOCAL_BOARD_SIZE - x + 1 } }
    return { col: LOCAL_BOARD_SIZE - y + 1, row: LOCAL_BOARD_SIZE - x + 1 }
  }

  function displayToLocal(col, row, family) {
    if (family === "O") { return [LOCAL_BOARD_SIZE - Number(row) + 1, Number(col)] }
    return [LOCAL_BOARD_SIZE - Number(row) + 1, LOCAL_BOARD_SIZE - Number(col) + 1]
  }

  function displayCellInLocalRegion(col, row, family) {
    if (Number(col) < 1 || Number(col) > LOCAL_BOARD_SIZE || Number(row) < 1 || Number(row) > LOCAL_BOARD_SIZE) {
      return false
    }
    return localDelta(displayToLocal(col, row, family), family) <= Number(LOCAL_DELTA_MAX[family] || LOCAL_DELTA_MAX.A)
  }

  function josekiLineForRandomRank(data, kind, localOnly, rank) {
    const core = kind === "core"
    const counts = core
      ? (localOnly ? data.localCoreDescendants : data.coreCounts)
      : (localOnly ? data.localLeafDescendants : data.leafCounts)
    let node = data.rootIndex
    let line = ""
    let endsLocal = false
    while (true) {
      const control = data.controls[node]
      const localCount = control & LOCAL_COUNT_MASK
      const edgeCount = retainedCount(control)
      if (core && node < data.rootIndex && (control & CORE) && (!localOnly || endsLocal)) {
        if (rank === 0) { return line }
        rank -= 1
      }
      if (!core && edgeCount === 0) { return line }
      let selected = false
      for (let edge = 0; edge < edgeCount; edge += 1) {
        const target = targetAt(data, node, edge)
        const isLocal = edge < localCount
        let weight
        if (target === null) {
          weight = !core && (!localOnly || isLocal) ? 1 : 0
        } else {
          weight = counts[target]
          if (localOnly && isLocal) {
            const targetControl = data.controls[target]
            if (core ? targetControl & CORE : retainedCount(targetControl) === 0) { weight += 1 }
          }
        }
        if (rank < weight) {
          const local = isLocal
            ? decodeLocalMove(data.localMoves[data.localStarts[node] + edge])
            : null
          line = appendMove(line, data.family, local)
          if (target === null) { return line }
          node = target
          endsLocal = isLocal
          selected = true
          break
        }
        rank -= weight
      }
      if (!selected) { throw new Error("Unsupported joseki random index") }
    }
  }

  return {
    LOCAL_BOARD_SIZE,
    parseEntries,
    normalizeLineForFamily,
    normalizeLoadedData,
    findJosekiNode,
    formatLine,
    localToDisplay,
    displayCellInLocalRegion,
    josekiLineForRandomRank,
  }
}
