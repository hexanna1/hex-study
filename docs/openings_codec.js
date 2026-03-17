window.HexOpeningCodec = (() => {
  const {
    bitmapRankBefore,
    bitmapRankCheckpoints,
    decodeOptionalThousandths,
    decodeRedirectReferences,
    readAsciiMagic,
    readPackedWordAtBit,
  } = window.HexStudyUI

  const { appendMoveToLine, cellIdToMove } = globalThis.HexPosition

  const BUNDLE_MAGIC = "HOD"
  const PACKED_OPTIONAL_NULL = 1023
  const HEADER_SIZE = 17
  const NODE_ROW_SIZE = 1
  const PACKED_NODE_COUNT_BITS = 6
  const PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_COUNT_BITS
  const PACKED_NODE_HAS_CHILDREN_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
  const PACKED_NODE_COUNT_MASK = (1 << PACKED_NODE_COUNT_BITS) - 1
  const PACKED_CANDIDATE_METRIC_BITS = 10
  const PACKED_CANDIDATE_DELTA_BITS = 8
  const PACKED_CANDIDATE_DELTA_ESCAPE = 2 ** PACKED_CANDIDATE_DELTA_BITS - 1
  const PACKED_CANDIDATE_DELTA_MAX_ABS = Math.floor((PACKED_CANDIDATE_DELTA_ESCAPE - 1) / 2)
  const OPENING_INDEX_CHECKPOINT_INTERVAL = 256

  function decodeOptionalBundleMetric(rawValue) { return decodeOptionalThousandths(rawValue, PACKED_OPTIONAL_NULL) }

  function packedMoveIdBits(boardSize) {
    const size = Number(boardSize)
    if (!Number.isInteger(size) || size <= 0) { throw new Error("Unsupported opening board size") }
    return Math.floor(Math.log2(size * size - 1)) + 1
  }

  function readPackedCandidateWord(view, offset, candidateIndex, rowBits) {
    return readPackedWordAtBit(view, offset, candidateIndex * rowBits, rowBits)
  }

  function moverMetricFromRedMetric(redMetric, ply) {
    if (redMetric === PACKED_OPTIONAL_NULL || redMetric === null) { return PACKED_OPTIONAL_NULL }
    return Number(ply) % 2 === 0 ? redMetric : 1000 - redMetric
  }

  function openingRedirectAt(data, candidateIndex) {
    const byte = data.redirectBitmap[candidateIndex >>> 3]
    return ((byte >> (candidateIndex % 8)) & 0x1) !== 0
  }

  function openingRedirectRankBefore(data, candidateIndex) {
    return bitmapRankBefore(
      data.redirectBitmap,
      data.redirectCheckpoints,
      candidateIndex,
      OPENING_INDEX_CHECKPOINT_INTERVAL,
    )
  }

  function openingRedirectTarget(data, redirectIndex) { return data.redirectTargets[redirectIndex] }

  function normalizeLoadedData(rawBuffer) {
    if (!(rawBuffer instanceof ArrayBuffer)) { throw new Error("Unsupported opening data format") }
    if (rawBuffer.byteLength < HEADER_SIZE) { throw new Error("Unsupported opening data format") }
    const view = new DataView(rawBuffer)
    if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) { throw new Error("Unsupported opening data format") }
    const boardSize = view.getUint16(3, true)
    const nodeCount = view.getUint32(5, true)
    const candidateCount = view.getUint32(9, true)
    const redirectCount = view.getUint32(13, true)
    const moveIdBits = packedMoveIdBits(boardSize)
    const moveHighBits = Math.max(0, moveIdBits - 8)
    const moveLowOffset = HEADER_SIZE + nodeCount * NODE_ROW_SIZE
    let deltaCount = 0
    let firstPriorCount = 0
    const candidateStarts = new Uint32Array(nodeCount + 1)
    let candidateTotal = 0
    for (let idx = 0; idx < nodeCount; idx += 1) {
      candidateStarts[idx] = candidateTotal
      const word = view.getUint8(HEADER_SIZE + idx)
      const candidateCountForNode = word & PACKED_NODE_COUNT_MASK
      candidateTotal += candidateCountForNode
      if (candidateCountForNode > 1) { deltaCount += candidateCountForNode }
      if (candidateCountForNode > 0) { firstPriorCount += 1 }
    }
    candidateStarts[nodeCount] = candidateTotal
    const moveHighOffset = moveLowOffset + candidateCount
    const priorLowOffset = moveHighOffset + Math.ceil((candidateCount * moveHighBits) / 8)
    const priorHighOffset = priorLowOffset + Math.ceil((candidateCount * 2) / 8)
    const deltaOffset = priorHighOffset + candidateCount
    const exceptionOffset = deltaOffset + deltaCount
    const checkpointCount = Math.ceil(nodeCount / OPENING_INDEX_CHECKPOINT_INTERVAL) + 1
    const candidateCheckpoints = new Uint32Array(checkpointCount)
    const firstPriorCheckpoints = new Uint32Array(checkpointCount)
    const deltaCheckpoints = new Uint32Array(checkpointCount)
    const exceptionCheckpoints = new Uint32Array(checkpointCount)
    const subtreeNodes = new Uint32Array(nodeCount)
    let candidateCursor = 0
    let firstPriorCursor = 0
    let deltaCursor = 0
    let exceptionCursor = 0
    for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
      if (nodeIndex % OPENING_INDEX_CHECKPOINT_INTERVAL === 0) {
        const checkpointIndex = nodeIndex / OPENING_INDEX_CHECKPOINT_INTERVAL
        candidateCheckpoints[checkpointIndex] = candidateCursor
        firstPriorCheckpoints[checkpointIndex] = firstPriorCursor
        deltaCheckpoints[checkpointIndex] = deltaCursor
        exceptionCheckpoints[checkpointIndex] = exceptionCursor
      }
      const count = view.getUint8(HEADER_SIZE + nodeIndex) & PACKED_NODE_COUNT_MASK
      candidateCursor += count
      if (count > 0) { firstPriorCursor += 1 }
      if (count > 1) {
        for (let candidateIndex = 0; candidateIndex < count; candidateIndex += 1) {
          if (view.getUint8(deltaOffset + deltaCursor + candidateIndex) === PACKED_CANDIDATE_DELTA_ESCAPE) {
            exceptionCursor += 1
          }
        }
        deltaCursor += count
      }
    }
    const endCheckpoint = Math.ceil(nodeCount / OPENING_INDEX_CHECKPOINT_INTERVAL)
    candidateCheckpoints[endCheckpoint] = candidateCursor
    firstPriorCheckpoints[endCheckpoint] = firstPriorCursor
    deltaCheckpoints[endCheckpoint] = deltaCursor
    exceptionCheckpoints[endCheckpoint] = exceptionCursor
    const redirectOffset = exceptionOffset + Math.ceil((exceptionCursor * PACKED_CANDIDATE_METRIC_BITS) / 8)
    const redirects = decodeRedirectReferences(
      new Uint8Array(rawBuffer),
      redirectOffset,
      redirectCount,
      candidateStarts,
    )
    if (
      candidateCursor !== candidateCount ||
      firstPriorCursor !== firstPriorCount ||
      deltaCursor !== deltaCount ||
      rawBuffer.byteLength !== redirects.offset
    ) { throw new Error("Opening bundle size mismatch") }
    const redirectCheckpoints = bitmapRankCheckpoints(
      redirects.bitmap,
      candidateCount,
      OPENING_INDEX_CHECKPOINT_INTERVAL,
    )

    const topologyData = { redirectBitmap: redirects.bitmap }
    for (let nodeIndex = nodeCount - 1; nodeIndex >= 0; nodeIndex -= 1) {
      const word = view.getUint8(HEADER_SIZE + nodeIndex)
      const count = word & PACKED_NODE_COUNT_MASK
      const hasChildren = (Math.trunc(word / 2 ** PACKED_NODE_HAS_CHILDREN_SHIFT) & 0x1) !== 0
      let subtreeSize = 1
      if (hasChildren) {
        let childIndex = nodeIndex + 1
        for (let candidateIndex = 0; candidateIndex < count; candidateIndex += 1) {
          const absoluteCandidate = candidateStarts[nodeIndex] + candidateIndex
          if (openingRedirectAt(topologyData, absoluteCandidate)) { continue }
          if (childIndex >= nodeCount || subtreeNodes[childIndex] === 0) {
            throw new Error("Opening node topology is incomplete")
          }
          subtreeSize += subtreeNodes[childIndex]
          childIndex += subtreeNodes[childIndex]
        }
      }
      subtreeNodes[nodeIndex] = subtreeSize
    }
    if (nodeCount === 0 || subtreeNodes[0] !== nodeCount) { throw new Error("Opening node topology is incomplete") }

    const data = {
      board_size: boardSize,
      view,
      nodeCount,
      candidateCount,
      redirectCount,
      moveHighBits,
      moveLowOffset,
      moveHighOffset,
      priorLowOffset,
      priorHighOffset,
      deltaOffset,
      exceptionOffset,
      redirectBitmap: redirects.bitmap,
      redirectTargets: redirects.targets,
      firstPriorCount,
      candidateCheckpoints,
      firstPriorCheckpoints,
      deltaCheckpoints,
      exceptionCheckpoints,
      redirectCheckpoints,
      candidateStarts,
      subtreeNodes,
      continuationCounts: null,
      nodeCache: new Map(),
      nodeByLine: new Map(),
      subtreeCore: null,
      subtreeLeaves: null,
      randomIndexScheduled: false,
    }
    const continuationCounts = new Uint32Array(nodeCount)
    const continuationState = new Uint8Array(nodeCount)
    function countContinuations(nodeIndex) {
      if (continuationState[nodeIndex] === 2) { return continuationCounts[nodeIndex] }
      if (continuationState[nodeIndex] === 1) { throw new Error("Opening redirect topology is cyclic") }
      continuationState[nodeIndex] = 1
      let count = view.getUint8(HEADER_SIZE + nodeIndex) & PACKED_NODE_COUNT_MASK
      for (const target of openingChildIndices(data, nodeIndex)) {
        if (target !== null) {
          if (target < 0 || target >= nodeCount) { throw new Error("Opening redirect topology is invalid") }
          count += countContinuations(target)
        }
      }
      if (!Number.isSafeInteger(count) || count > 0xffffffff) {
        throw new Error("Opening continuation count is too large")
      }
      continuationCounts[nodeIndex] = count
      continuationState[nodeIndex] = 2
      return count
    }
    countContinuations(0)
    for (const visitState of continuationState) {
      if (visitState !== 2) { throw new Error("Opening graph contains unreachable nodes") }
    }
    data.continuationCounts = continuationCounts
    return data
  }

  function openingNodeCursors(data, nodeIndex) {
    const checkpointIndex = Math.floor(nodeIndex / OPENING_INDEX_CHECKPOINT_INTERVAL)
    const checkpointNode = checkpointIndex * OPENING_INDEX_CHECKPOINT_INTERVAL
    let candidate = data.candidateCheckpoints[checkpointIndex]
    let firstPrior = data.firstPriorCheckpoints[checkpointIndex]
    let delta = data.deltaCheckpoints[checkpointIndex]
    let exception = data.exceptionCheckpoints[checkpointIndex]
    for (let index = checkpointNode; index < nodeIndex; index += 1) {
      const count = data.view.getUint8(HEADER_SIZE + index) & PACKED_NODE_COUNT_MASK
      candidate += count
      if (count > 0) { firstPrior += 1 }
      if (count > 1) {
        for (let candidateIndex = 0; candidateIndex < count; candidateIndex += 1) {
          if (data.view.getUint8(data.deltaOffset + delta + candidateIndex) === PACKED_CANDIDATE_DELTA_ESCAPE) {
            exception += 1
          }
        }
        delta += count
      }
    }
    return { candidate, firstPrior, delta, exception }
  }

  function openingChildIndices(data, nodeIndex) {
    const word = data.view.getUint8(HEADER_SIZE + nodeIndex)
    const candidateCount = word & PACKED_NODE_COUNT_MASK
    const hasChildren = (Math.trunc(word / 2 ** PACKED_NODE_HAS_CHILDREN_SHIFT) & 0x1) !== 0
    if (!hasChildren) { return Array(candidateCount).fill(null) }
    const candidateStart = data.candidateStarts[nodeIndex]
    let redirectIndex = openingRedirectRankBefore(data, candidateStart)
    let childIndex = nodeIndex + 1
    const targets = []
    for (let index = 0; index < candidateCount; index += 1) {
      const absoluteCandidate = candidateStart + index
      if (openingRedirectAt(data, absoluteCandidate)) {
        targets.push(openingRedirectTarget(data, redirectIndex))
        redirectIndex += 1
        continue
      }
      targets.push(childIndex)
      childIndex += data.subtreeNodes[childIndex]
    }
    return targets
  }

  function openingDescendantCount(data, nodeIndex) {
    if (!data || !Number.isInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= data.nodeCount) { return 0 }
    return data.continuationCounts[nodeIndex]
  }

  function decodeOpeningNode(data, nodeIndex, line, parentEdgeRedMetric, ply) {
    let structure = data.nodeCache.get(nodeIndex)
    if (!structure) {
      const word = data.view.getUint8(HEADER_SIZE + nodeIndex)
      const candidateCount = word & PACKED_NODE_COUNT_MASK
      const cursors = openingNodeCursors(data, nodeIndex)
      const childIndices = openingChildIndices(data, nodeIndex)
      const dropPriorStart = data.firstPriorCount + cursors.candidate - cursors.firstPrior
      const candidates = []
      let previousPrior = null
      let exceptionIndex = cursors.exception
      for (let index = 0; index < candidateCount; index += 1) {
        const candidateIndex = cursors.candidate + index
        const moveLow = data.view.getUint8(data.moveLowOffset + candidateIndex)
        const moveHigh = data.moveHighBits
          ? readPackedCandidateWord(data.view, data.moveHighOffset, candidateIndex, data.moveHighBits)
          : 0
        const moveId = moveLow + moveHigh * 256
        const priorStreamIndex = index === 0 ? cursors.firstPrior : dropPriorStart + index - 1
        const priorLow = readPackedCandidateWord(data.view, data.priorLowOffset, priorStreamIndex, 2)
        const priorHigh = data.view.getUint8(data.priorHighOffset + priorStreamIndex)
        const priorValue = priorLow + priorHigh * 4
        const prior = index === 0 ? priorValue : previousPrior - priorValue
        if (!Number.isInteger(prior) || prior < 0 || prior >= 2 ** PACKED_CANDIDATE_METRIC_BITS) {
          throw new Error("Unsupported opening candidate prior")
        }
        previousPrior = prior
        let redMetric = parentEdgeRedMetric
        if (candidateCount !== 1) {
          const deltaCode = data.view.getUint8(data.deltaOffset + cursors.delta + index)
          if (deltaCode === PACKED_CANDIDATE_DELTA_ESCAPE) {
            redMetric = readPackedCandidateWord(
              data.view,
              data.exceptionOffset,
              exceptionIndex,
              PACKED_CANDIDATE_METRIC_BITS,
            )
            exceptionIndex += 1
          } else {
            if (parentEdgeRedMetric === null || parentEdgeRedMetric === PACKED_OPTIONAL_NULL) {
              throw new Error("Unsupported opening candidate metric")
            }
            redMetric = parentEdgeRedMetric + deltaCode - PACKED_CANDIDATE_DELTA_MAX_ABS
          }
        }
        candidates.push({
          move: cellIdToMove(moveId, data.board_size),
          childIndex: childIndices[index],
          redMetric,
          prior: decodeOptionalBundleMetric(prior),
          tree_mover_winrate: decodeOptionalBundleMetric(moverMetricFromRedMetric(redMetric, ply)),
          elo_loss: null,
          retained: true,
        })
      }
      structure = {
        index: nodeIndex,
        is_core: (Math.trunc(word / 2 ** PACKED_NODE_IS_CORE_SHIFT) & 0x1) !== 0,
        candidates,
      }
      data.nodeCache.set(nodeIndex, structure)
    }
    const node = {
      ...structure,
      line,
      candidates: structure.candidates.map((candidate) => ({
        ...candidate,
        childLine: appendMoveToLine(line, candidate.move),
      })),
    }
    data.nodeByLine.set(line, node)
    return node
  }

  function ensureOpeningRandomIndex(data) {
    if (!data || (data.subtreeCore && data.subtreeLeaves)) { return }
    const subtreeCore = new Uint32Array(data.nodeCount)
    const subtreeLeaves = new Uint32Array(data.nodeCount)
    const visited = new Uint8Array(data.nodeCount)
    function indexNode(nodeIndex) {
      if (visited[nodeIndex]) { return }
      visited[nodeIndex] = 1
      const word = data.view.getUint8(HEADER_SIZE + nodeIndex)
      const candidateCount = word & PACKED_NODE_COUNT_MASK
      const hasChildren = (Math.trunc(word / 2 ** PACKED_NODE_HAS_CHILDREN_SHIFT) & 0x1) !== 0
      let coreCount = nodeIndex > 0 && (Math.trunc(word / 2 ** PACKED_NODE_IS_CORE_SHIFT) & 0x1) !== 0 ? 1 : 0
      let leafCount = hasChildren ? 0 : Math.max(1, candidateCount)
      if (hasChildren) {
        for (const childIndex of openingChildIndices(data, nodeIndex)) {
          indexNode(childIndex)
          coreCount += subtreeCore[childIndex]
          leafCount += subtreeLeaves[childIndex]
        }
      }
      if (coreCount > 0xffffffff || leafCount > 0xffffffff) { throw new Error("Opening random index is too large") }
      subtreeCore[nodeIndex] = coreCount
      subtreeLeaves[nodeIndex] = leafCount
    }
    indexNode(0)
    data.subtreeCore = subtreeCore
    data.subtreeLeaves = subtreeLeaves
  }

  function openingLineForRandomRank(data, kind, rank) {
    let nodeIndex = 0
    let line = ""
    let parentEdgeRedMetric = null
    let ply = 0
    while (true) {
      const node = decodeOpeningNode(data, nodeIndex, line, parentEdgeRedMetric, ply)
      if (kind === "core" && nodeIndex > 0 && node.is_core) {
        if (rank === 0) { return line }
        rank -= 1
      }
      if (kind === "leaf" && node.candidates.length === 0) { return line }
      let selected = false
      for (const candidate of node.candidates) {
        const weight =
          candidate.childIndex === null
            ? kind === "leaf"
              ? 1
              : 0
            : kind === "core"
              ? data.subtreeCore[candidate.childIndex]
              : data.subtreeLeaves[candidate.childIndex]
        if (rank < weight) {
          if (candidate.childIndex === null) { return candidate.childLine }
          nodeIndex = candidate.childIndex
          line = candidate.childLine
          parentEdgeRedMetric = candidate.redMetric
          ply += 1
          selected = true
          break
        }
        rank -= weight
      }
      if (!selected) { throw new Error("Unsupported opening random index") }
    }
  }

  return {
    normalizeLoadedData,
    openingChildIndices,
    openingDescendantCount,
    decodeOpeningNode,
    ensureOpeningRandomIndex,
    openingLineForRandomRank,
  }
})()
