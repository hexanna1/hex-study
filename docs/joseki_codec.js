window.createJosekiCodec = (AVAILABLE_FAMILIES) => {
  const {
    bitmapRankBefore,
    bitmapRankCheckpoints,
    decodeThousandths,
    readAsciiMagic,
    readBitplaneValue,
    readPackedWordAtBit,
  } = window.HexStudyUI

  const BUNDLE_MAGIC = "HJB"
  const FAMILY_CODE_TO_NAME = { 1: "A", 2: "O" }
  const HEADER_SIZE = 17
  const JOSEKI_INDEX_CHECKPOINT_INTERVAL = 256
  const PACKED_NODE_LOCAL_COUNT_BITS = 4
  const PACKED_NODE_LOCAL_COUNT_MASK = (1 << PACKED_NODE_LOCAL_COUNT_BITS) - 1
  const PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_LOCAL_COUNT_BITS
  const PACKED_NODE_TENUKI_RETAINED_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
  const PACKED_NODE_TENUKI_CHILD_SHIFT = PACKED_NODE_TENUKI_RETAINED_SHIFT + 1
  const PACKED_NODE_LOCAL_CHILDREN_SHIFT = PACKED_NODE_TENUKI_CHILD_SHIFT + 1
  const PACKED_TENUKI_DROP_HIGH_BITS = 2

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
    const occupied = new Set()
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
      const key = `${entry[0]},${entry[1]}`
      if (rejectDuplicates && occupied.has(key)) { return null }
      occupied.add(key)
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

  function josekiRedirectAt(data, edgeIndex) {
    const byte = data.redirectBitmap[edgeIndex >>> 3]
    return ((byte >> (edgeIndex % 8)) & 0x1) !== 0
  }

  function josekiRedirectRankBefore(data, edgeIndex) {
    return bitmapRankBefore(data.redirectBitmap, data.redirectCheckpoints, edgeIndex, JOSEKI_INDEX_CHECKPOINT_INTERVAL)
  }

  function josekiRedirectTarget(data, redirectIndex) {
    return readPackedWordAtBit(data.view, data.redirectTargetOffset, redirectIndex * data.nodeIdBits, data.nodeIdBits)
  }

  function josekiChildIndices(data, nodeIndex) {
    const rawNode = data.rawNodes[nodeIndex]
    const edgeStart = data.edgeStarts[nodeIndex]
    let redirectIndex = josekiRedirectRankBefore(data, edgeStart)
    let childIndex = nodeIndex + 1
    const targets = []
    const childPresence = [
      ...Array(rawNode.localCount).fill(rawNode.localChildren),
      ...(rawNode.tenukiRetained ? [rawNode.tenukiChildPresent] : []),
    ]
    for (let index = 0; index < childPresence.length; index += 1) {
      const edgeIndex = edgeStart + index
      if (josekiRedirectAt(data, edgeIndex)) {
        targets.push(josekiRedirectTarget(data, redirectIndex))
        redirectIndex += 1
      } else if (childPresence[index]) {
        targets.push(childIndex)
        childIndex += data.subtreeNodes[childIndex]
      } else { targets.push(null) }
    }
    return targets
  }

  function normalizeLoadedData(rawBuffer) {
    if (!(rawBuffer instanceof ArrayBuffer)) { throw new Error("Unsupported joseki data format") }
    if (rawBuffer.byteLength < HEADER_SIZE) { throw new Error("Unsupported joseki data format") }
    const view = new DataView(rawBuffer)
    if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) { throw new Error("Unsupported joseki data format") }
    const family = FAMILY_CODE_TO_NAME[view.getUint8(3)]
    if (!AVAILABLE_FAMILIES.includes(family)) { throw new Error("Unsupported joseki data format") }
    const boardSize = view.getUint8(4)
    const nodeCount = view.getUint32(5, true)
    const localRowCount = view.getUint32(9, true)
    const redirectCount = view.getUint32(13, true)
    const nodeControlOffset = HEADER_SIZE
    const tenukiDropLowOffset = nodeControlOffset + nodeCount
    const tenukiDropHighOffset = tenukiDropLowOffset + nodeCount
    const localMoveOffset = tenukiDropHighOffset + Math.ceil((nodeCount * PACKED_TENUKI_DROP_HIGH_BITS) / 8)
    const firstLocalDropOffset = localMoveOffset + localRowCount
    const rawNodes = []
    const localStarts = new Uint32Array(nodeCount + 1)
    const firstLocalStarts = new Uint32Array(nodeCount + 1)
    const edgeStarts = new Uint32Array(nodeCount + 1)
    let firstLocalDropCount = 0
    let localTotal = 0
    let edgeTotal = 0
    for (let idx = 0; idx < nodeCount; idx += 1) {
      localStarts[idx] = localTotal
      firstLocalStarts[idx] = firstLocalDropCount
      edgeStarts[idx] = edgeTotal
      const word = view.getUint8(nodeControlOffset + idx)
      const tenukiDrop =
        view.getUint8(tenukiDropLowOffset + idx) +
        readBitplaneValue(view, tenukiDropHighOffset, nodeCount, idx, PACKED_TENUKI_DROP_HIGH_BITS) * 256
      const tenukiPresent = idx > 0
      if (!tenukiPresent && tenukiDrop !== 0) { throw new Error("Unsupported joseki root tenuki drop") }
      const tenukiStoneFraction = tenukiPresent ? 1000 - tenukiDrop : 0
      if (tenukiStoneFraction < 0 || tenukiStoneFraction > 1000) { throw new Error("Unsupported joseki tenuki drop") }
      const localCount = word & PACKED_NODE_LOCAL_COUNT_MASK
      if (localCount > 0) { firstLocalDropCount += 1 }
      const rawNode = {
        localCount,
        isCore: (Math.trunc(word / 2 ** PACKED_NODE_IS_CORE_SHIFT) & 0x1) !== 0,
        tenukiRetained: (Math.trunc(word / 2 ** PACKED_NODE_TENUKI_RETAINED_SHIFT) & 0x1) !== 0,
        tenukiPresent,
        tenukiChildPresent: (Math.trunc(word / 2 ** PACKED_NODE_TENUKI_CHILD_SHIFT) & 0x1) !== 0,
        tenukiStoneFraction,
        localChildren: (Math.trunc(word / 2 ** PACKED_NODE_LOCAL_CHILDREN_SHIFT) & 0x1) !== 0,
      }
      rawNodes.push(rawNode)
      localTotal += localCount
      edgeTotal += localCount + (rawNode.tenukiRetained ? 1 : 0)
    }
    localStarts[nodeCount] = localTotal
    firstLocalStarts[nodeCount] = firstLocalDropCount
    edgeStarts[nodeCount] = edgeTotal
    const siblingLocalDropOffset = firstLocalDropOffset + firstLocalDropCount
    const redirectBitmapOffset = siblingLocalDropOffset + localRowCount - firstLocalDropCount
    const redirectTargetOffset = redirectBitmapOffset + Math.ceil(edgeTotal / 8)
    const nodeIdBits = Math.max(1, Math.ceil(Math.log2(Math.max(1, nodeCount))))
    const expectedSize = redirectTargetOffset + Math.ceil((redirectCount * nodeIdBits) / 8)
    if (localTotal !== localRowCount || rawBuffer.byteLength !== expectedSize) {
      throw new Error("Joseki bundle size mismatch")
    }
    const redirectBitmap = new Uint8Array(rawBuffer, redirectBitmapOffset, Math.ceil(edgeTotal / 8))
    const redirectCheckpoints = bitmapRankCheckpoints(redirectBitmap, edgeTotal, JOSEKI_INDEX_CHECKPOINT_INTERVAL)

    const subtreeNodes = new Uint32Array(nodeCount)
    const data = {
      family,
      board_size: Number(boardSize || 19),
      view,
      nodeCount,
      nodeIdBits,
      rawNodes,
      localStarts,
      firstLocalStarts,
      edgeStarts,
      localMoveOffset,
      firstLocalDropOffset,
      siblingLocalDropOffset,
      redirectBitmap,
      redirectTargetOffset,
      redirectCheckpoints,
      subtreeNodes,
      continuationCounts: null,
      nodeCache: new Map(),
      nodeByLine: new Map(),
      subtreeCore: null,
      subtreeCoreLocal: null,
      subtreeLeaves: null,
      subtreeLeavesLocal: null,
      randomIndexScheduled: false,
    }
    for (let nodeIndex = nodeCount - 1; nodeIndex >= 0; nodeIndex -= 1) {
      const rawNode = rawNodes[nodeIndex]
      let subtreeSize = 1
      let childIndex = nodeIndex + 1
      const childPresence = [
        ...Array(rawNode.localCount).fill(rawNode.localChildren),
        ...(rawNode.tenukiRetained ? [rawNode.tenukiChildPresent] : []),
      ]
      for (let edgeOffset = 0; edgeOffset < childPresence.length; edgeOffset += 1) {
        if (!childPresence[edgeOffset] || josekiRedirectAt(data, edgeStarts[nodeIndex] + edgeOffset)) { continue }
        if (childIndex >= nodeCount || subtreeNodes[childIndex] === 0) {
          throw new Error("Joseki node topology is incomplete")
        }
        subtreeSize += subtreeNodes[childIndex]
        childIndex += subtreeNodes[childIndex]
      }
      subtreeNodes[nodeIndex] = subtreeSize
    }
    if (nodeCount === 0 || subtreeNodes[0] !== nodeCount) { throw new Error("Joseki node topology is incomplete") }

    const continuationCounts = new Uint32Array(nodeCount)
    const continuationState = new Uint8Array(nodeCount)
    function countContinuations(nodeIndex) {
      if (continuationState[nodeIndex] === 2) { return continuationCounts[nodeIndex] }
      if (continuationState[nodeIndex] === 1) { throw new Error("Joseki redirect topology is cyclic") }
      continuationState[nodeIndex] = 1
      let count = edgeStarts[nodeIndex + 1] - edgeStarts[nodeIndex]
      for (const target of josekiChildIndices(data, nodeIndex)) {
        if (target !== null) {
          if (target < 0 || target >= nodeCount) { throw new Error("Joseki redirect topology is invalid") }
          count += countContinuations(target)
        }
      }
      if (!Number.isSafeInteger(count) || count > 0xffffffff) {
        throw new Error("Joseki continuation count is too large")
      }
      continuationCounts[nodeIndex] = count
      continuationState[nodeIndex] = 2
      return count
    }
    countContinuations(0)
    if ([...continuationState].some((value) => value !== 2)) {
      throw new Error("Joseki graph contains unreachable nodes")
    }
    data.continuationCounts = continuationCounts
    return data
  }

  function decodeJosekiNode(data, nodeIndex, line) {
    let structure = data.nodeCache.get(nodeIndex)
    if (!structure) {
      const rawNode = data.rawNodes[nodeIndex]
      const childIndices = josekiChildIndices(data, nodeIndex)
      const candidates = []
      let childOffset = 0
      const localStart = data.localStarts[nodeIndex]
      const firstLocalStart = data.firstLocalStarts[nodeIndex]
      const siblingLocalStart = localStart - firstLocalStart
      let previousDrop = 0
      for (let index = 0; index < rawNode.localCount; index += 1) {
        if (index === 0) { previousDrop = data.view.getUint8(data.firstLocalDropOffset + firstLocalStart) } else {
          previousDrop =
            (previousDrop + data.view.getUint8(data.siblingLocalDropOffset + siblingLocalStart + index - 1)) & 0xff
        }
        candidates.push({
          kind: "local",
          local: decodeLocalMove(data.view.getUint8(data.localMoveOffset + localStart + index)),
          stone_fraction: decodeThousandths(1000 - previousDrop),
          retained: true,
          childIndex: childIndices[childOffset],
        })
        childOffset += 1
      }
      if (rawNode.tenukiPresent) {
        candidates.push({
          kind: "tenuki",
          stone_fraction: decodeThousandths(rawNode.tenukiStoneFraction),
          retained: rawNode.tenukiRetained,
          childIndex: rawNode.tenukiRetained ? childIndices[childOffset] : null,
        })
      }
      structure = { index: nodeIndex, is_core: rawNode.isCore, candidates }
      data.nodeCache.set(nodeIndex, structure)
    }
    const family = data.family
    const entries = line ? parseEntries(line, family) : []
    const candidates = structure.candidates.map((candidate) => ({
      ...candidate,
      childLine: candidate.retained
        ? formatLine(family, [...entries, candidate.kind === "local" ? candidate.local : null])
        : null,
    }))
    const node = { ...structure, line, candidates }
    data.nodeByLine.set(line, node)
    return node
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

  function ensureJosekiRandomIndex(data) {
    if (!data || (data.subtreeCore && data.subtreeCoreLocal && data.subtreeLeaves && data.subtreeLeavesLocal)) {
      return
    }
    const subtreeCore = new Uint32Array(data.nodeCount)
    const subtreeCoreLocal = new Uint32Array(data.nodeCount)
    const subtreeLeaves = new Uint32Array(data.nodeCount)
    const subtreeLeavesLocal = new Uint32Array(data.nodeCount)
    const endingKind = new Int8Array(data.nodeCount)
    const visited = new Uint8Array(data.nodeCount)
    endingKind[0] = -1

    function indexNode(nodeIndex) {
      if (visited[nodeIndex] === 2) { return }
      if (visited[nodeIndex] === 1) { throw new Error("Joseki redirect topology is cyclic") }
      visited[nodeIndex] = 1
      const rawNode = data.rawNodes[nodeIndex]
      const targets = josekiChildIndices(data, nodeIndex)
      const edgeCount = targets.length
      let coreCount = nodeIndex > 0 && rawNode.isCore ? 1 : 0
      let coreLocalCount = coreCount && endingKind[nodeIndex] === 1 ? 1 : 0
      let leafCount = edgeCount === 0 ? 1 : 0
      let leafLocalCount = leafCount && endingKind[nodeIndex] === 1 ? 1 : 0
      for (let edgeOffset = 0; edgeOffset < edgeCount; edgeOffset += 1) {
        const target = targets[edgeOffset]
        const isLocal = edgeOffset < rawNode.localCount
        if (target === null) {
          leafCount += 1
          leafLocalCount += isLocal ? 1 : 0
          continue
        }
        const targetKind = isLocal ? 1 : 2
        if (endingKind[target] !== 0 && endingKind[target] !== targetKind) {
          throw new Error("Joseki node has inconsistent incoming move kinds")
        }
        endingKind[target] = targetKind
        indexNode(target)
        coreCount += subtreeCore[target]
        coreLocalCount += subtreeCoreLocal[target]
        leafCount += subtreeLeaves[target]
        leafLocalCount += subtreeLeavesLocal[target]
      }
      for (const count of [coreCount, coreLocalCount, leafCount, leafLocalCount]) {
        if (!Number.isSafeInteger(count) || count > 0xffffffff) { throw new Error("Joseki random index is too large") }
      }
      subtreeCore[nodeIndex] = coreCount
      subtreeCoreLocal[nodeIndex] = coreLocalCount
      subtreeLeaves[nodeIndex] = leafCount
      subtreeLeavesLocal[nodeIndex] = leafLocalCount
      visited[nodeIndex] = 2
    }
    indexNode(0)
    data.subtreeCore = subtreeCore
    data.subtreeCoreLocal = subtreeCoreLocal
    data.subtreeLeaves = subtreeLeaves
    data.subtreeLeavesLocal = subtreeLeavesLocal
  }

  function josekiLineForRandomRank(data, kind, localOnly, rank) {
    let nodeIndex = 0
    let line = ""
    while (true) {
      const node = decodeJosekiNode(data, nodeIndex, line)
      const entries = line ? parseEntries(line, data.family) : []
      const endsLocal = entries.length > 0 && entries[entries.length - 1] !== null
      if (kind === "core" && nodeIndex > 0 && node.is_core && (!localOnly || endsLocal)) {
        if (rank === 0) { return line }
        rank -= 1
      }
      const retained = node.candidates.filter((candidate) => candidate.retained)
      if (kind === "leaf" && retained.length === 0) { return line }
      let selected = false
      for (const candidate of retained) {
        const weight =
          candidate.childIndex === null
            ? kind === "leaf" && (!localOnly || candidate.kind === "local")
              ? 1
              : 0
            : kind === "core"
              ? localOnly
                ? data.subtreeCoreLocal[candidate.childIndex]
                : data.subtreeCore[candidate.childIndex]
              : localOnly
                ? data.subtreeLeavesLocal[candidate.childIndex]
                : data.subtreeLeaves[candidate.childIndex]
        if (rank < weight) {
          if (candidate.childIndex === null) { return candidate.childLine }
          nodeIndex = candidate.childIndex
          line = candidate.childLine
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
    josekiChildIndices,
    normalizeLoadedData,
    decodeJosekiNode,
    formatLine,
    localToDisplay,
    displayCellInLocalRegion,
    ensureJosekiRandomIndex,
    josekiLineForRandomRank,
  }
}
