window.HexOpeningCodec = (() => {
  const {
    bitmapRankBefore,
    bitmapRankCheckpoints,
    decodeOptionalThousandths,
    readAsciiMagic,
    readUvarintAt,
  } = window.HexStudyUI

  const { cellIdToMove } = globalThis.HexPosition

  const BUNDLE_MAGIC = "HOD"
  const HEADER_SIZE = 37
  const NODE_COUNT_MASK = 63
  const NODE_IS_CORE = 64
  const NODE_HAS_CHILDREN = 128
  const OPTIONAL_NULL = 1023
  const LEAD_ESCAPE = 15
  const OTHER_ESCAPE = 255
  const MOVE_MODEL_BITS = 10
  const MOVE_MODEL_TOTAL = 1 << MOVE_MODEL_BITS
  const MOVE_MODEL_MASK = MOVE_MODEL_TOTAL - 1
  const MOVE_STATE_FLOOR = 1 << 23
  const NODE_INTERVAL = 32
  const RANK_INTERVAL = 64
  const LARGE_COUNT = 255
  const LINE_CACHE_CAPACITY = 128

  function packedNibble(bytes, offset, index) {
    return (bytes[offset + (index >>> 1)] >>> ((index & 1) * 4)) & 15
  }

  function packedU10(bytes, offset, index) {
    const bit = index * 10
    const start = offset + (bit >>> 3)
    return ((bytes[start] | (bytes[start + 1] << 8)) >>> (bit & 7)) & OPTIONAL_NULL
  }

  function decodeNodeRows(bytes, nodeCount, largeCount, lastParentLimit, flagCount) {
    const nodes = new Uint8Array(nodeCount)
    let overflow = HEADER_SIZE + Math.ceil(nodeCount / 4)
    let offset = overflow + largeCount
    for (let index = 0; index < nodeCount; index += 1) {
      let count = (bytes[HEADER_SIZE + (index >>> 2)] >>> ((index & 3) * 2)) & 3
      if (count === 3) { count = bytes[overflow++] }
      nodes[index] = count | (count > 0 && index < lastParentLimit ? NODE_HAS_CHILDREN : 0)
    }
    let index = -1
    for (let i = 0; i < flagCount; i += 1) {
      const flags = readUvarintAt(bytes, offset)
      offset = flags.offset
      index += Math.floor(flags.value / 4) + 1
      if (flags.value & 1) { nodes[index] |= NODE_IS_CORE }
      if (flags.value & 2) { nodes[index] &= ~NODE_HAS_CHILDREN }
    }
    return { nodes, offset }
  }

  function isRedirect(bitmap, edge) {
    return (bitmap[edge >>> 3] & (1 << (edge & 7))) !== 0
  }

  function decodeRedirects(bytes, offset, edgeCount, redirectCount) {
    const bitmap = new Uint8Array(Math.ceil(edgeCount / 8))
    const targets = new Uint32Array(redirectCount)
    let position = -1
    for (let rank = 0; rank < redirectCount; rank += 1) {
      const gap = readUvarintAt(bytes, offset)
      offset = gap.offset
      position += gap.value + 1
      bitmap[position >>> 3] |= 1 << (position & 7)
      targets[rank] = position - rank
    }
    for (let rank = 0; rank < redirectCount; rank += 1) {
      const distance = readUvarintAt(bytes, offset)
      offset = distance.offset
      targets[rank] -= distance.value
    }
    return { bitmap, targets, checkpoints: bitmapRankCheckpoints(bitmap, edgeCount, RANK_INTERVAL), offset }
  }

  function decodeOpeningMoves(wire, presentOffset, frequencyCount, size, nodes, candidateCount, redirectBitmap) {
    const cellCount = size * size
    const tableSize = (cellCount + 1) * cellCount
    const slots = new Uint32Array((cellCount + 1) * MOVE_MODEL_TOTAL)
    let frequencyOffset = presentOffset + Math.ceil(tableSize / 8)
    let inputOffset = frequencyOffset + frequencyCount
    let modelIndex = 0
    for (let context = 0; context <= cellCount; context += 1) {
      const firstSlot = context * MOVE_MODEL_TOTAL
      let start = 0
      for (let move = 0; move < cellCount; move += 1, modelIndex += 1) {
        if ((wire[presentOffset + (modelIndex >>> 3)] & (1 << (modelIndex & 7))) === 0) continue
        let minusOne = wire[frequencyOffset++]
        if (minusOne === 255) {
          minusOne = wire[inputOffset] | (wire[inputOffset + 1] << 8)
          inputOffset += 2
        }
        const frequency = minusOne + 1
        // Each inverse CDF slot stores the symbol, its interval start, and its frequency minus one.
        const code = move | (start << MOVE_MODEL_BITS) | (minusOne << (2 * MOVE_MODEL_BITS))
        slots.fill(code, firstSlot + start, firstSlot + start + frequency)
        start += frequency
      }
    }

    const transpose = new Uint16Array(cellCount)
    for (let move = 0; move < cellCount; move += 1) {
      transpose[move] = (move % size) * size + Math.floor(move / size)
    }
    const incomingMoves = cellCount < 256 ? new Uint8Array(nodes.length) : new Uint16Array(nodes.length)
    incomingMoves[0] = cellCount
    const moves = cellCount <= 256 ? new Uint8Array(candidateCount) : new Uint16Array(candidateCount)
    let state = (wire[inputOffset] | wire[inputOffset + 1] << 8 | wire[inputOffset + 2] << 16 | wire[inputOffset + 3] << 24) >>> 0
    inputOffset += 4
    let nextPly = 1
    let nextChild = 1
    let candidateIndex = 0
    let edge = 0
    let isOddPly = false
    // An implicit edge discovers the next breadth-first node. Its move,
    // transposed for the next player, supplies that node's model context.
    for (let index = 0; index < nodes.length; index += 1) {
      if (index === nextPly) {
        nextPly = nextChild
        isOddPly = !isOddPly
      }
      const word = nodes[index]
      const count = word & NODE_COUNT_MASK
      if (!count) continue
      const hasChildren = Boolean(word & NODE_HAS_CHILDREN)
      const firstSlot = incomingMoves[index] * MOVE_MODEL_TOTAL
      for (let candidate = 0; candidate < count; candidate += 1) {
        const slot = state & MOVE_MODEL_MASK
        const code = slots[firstSlot + slot]
        const frequency = (code >>> (2 * MOVE_MODEL_BITS)) + 1
        const start = (code >>> MOVE_MODEL_BITS) & MOVE_MODEL_MASK
        state = (state >>> MOVE_MODEL_BITS) * frequency + slot - start
        // A ten-bit symbol needs at most two bytes to restore the state floor.
        if (state < MOVE_STATE_FLOOR) {
          state = (state << 8) | wire[inputOffset++]
          if (state < MOVE_STATE_FLOOR) state = (state << 8) | wire[inputOffset++]
        }
        const move = code & MOVE_MODEL_MASK
        moves[candidateIndex++] = isOddPly ? transpose[move] : move
        if (hasChildren && !isRedirect(redirectBitmap, edge++)) {
          incomingMoves[nextChild++] = transpose[move]
        }
      }
    }
    return moves
  }

  function residualCode(bytes, offset, index, escape) {
    return escape === LEAD_ESCAPE ? packedNibble(bytes, offset, index) : bytes[offset + index]
  }

  function indexResiduals(bytes, offset, length, escape) {
    const blockCount = Math.ceil(length / RANK_INTERVAL)
    const checkpoints = new Uint32Array(blockCount)
    let exceptionCount = 0
    for (let block = 0; block < blockCount; block += 1) {
      const start = block * RANK_INTERVAL
      const end = Math.min(start + RANK_INTERVAL, length)
      checkpoints[block] = exceptionCount
      for (let index = start; index < end; index += 1) {
        exceptionCount += residualCode(bytes, offset, index, escape) === escape
      }
    }
    const exceptionOffset = offset + (escape === LEAD_ESCAPE ? Math.ceil(length / 2) : length)
    return { offset, checkpoints, escape, exceptionOffset, end: exceptionOffset + Math.ceil(exceptionCount * 10 / 8) }
  }

  function exceptionRankBefore(bytes, stream, index) {
    const { offset, escape, checkpoints } = stream
    const block = Math.floor(index / RANK_INTERVAL)
    let rank = checkpoints[block]
    for (let cursor = block * RANK_INTERVAL; cursor < index; cursor += 1) {
      rank += residualCode(bytes, offset, cursor, escape) === escape
    }
    return rank
  }

  function makeCounts(nodeCount) {
    return { small: new Uint8Array(nodeCount), large: new Map() }
  }

  function readCount(counts, nodeIndex) {
    const value = counts.small[nodeIndex]
    return value < LARGE_COUNT ? value : counts.large.get(nodeIndex)
  }

  function writeCount(counts, nodeIndex, value) {
    if (value < LARGE_COUNT) {
      counts.small[nodeIndex] = value
    } else {
      counts.small[nodeIndex] = LARGE_COUNT
      counts.large.set(nodeIndex, value)
    }
  }

  function indexGraph(nodes, candidateCount, firstPriorCount, branchCount, redirects) {
    const nodeCount = nodes.length
    const checkpointCount = Math.ceil(nodeCount / NODE_INTERVAL)
    const candidateCheckpoints = new Uint32Array(checkpointCount)
    const firstPriorCheckpoints = new Uint32Array(checkpointCount)
    const branchCheckpoints = new Uint32Array(checkpointCount)
    const edgeCheckpoints = new Uint32Array(checkpointCount)
    const continuationCounts = makeCounts(nodeCount)
    const subtreeCore = makeCounts(nodeCount)
    const subtreeLeaves = makeCounts(nodeCount)
    let candidateStart = candidateCount
    let firstPriorStart = firstPriorCount
    let branchStart = branchCount
    let redirectIndex = redirects.targets.length
    let nextTreeChild = nodeCount - 1
    let edgeStart = nextTreeChild + redirectIndex

    // Breadth-first IDs put all child counts ahead of their parents. Walk the
    // materialized edges backward so implicit children and redirects are both sequential.
    for (let nodeIndex = nodeCount - 1; nodeIndex >= 0; nodeIndex -= 1) {
      const word = nodes[nodeIndex]
      const count = word & NODE_COUNT_MASK
      candidateStart -= count
      firstPriorStart -= count > 0
      branchStart -= count > 1
      let continuations = count
      let core = nodeIndex > 0 && (word & NODE_IS_CORE) ? 1 : 0
      let leaves = Math.max(1, count)
      if (word & NODE_HAS_CHILDREN) {
        leaves = 0
        for (let end = edgeStart - count; edgeStart > end;) {
          const target = isRedirect(redirects.bitmap, --edgeStart)
            ? redirects.targets[--redirectIndex]
            : nextTreeChild--
          if (target <= nodeIndex || target >= nodeCount) { throw new Error("Opening redirect topology is invalid") }
          continuations += readCount(continuationCounts, target)
          core += readCount(subtreeCore, target)
          leaves += readCount(subtreeLeaves, target)
        }
      }
      if (Math.max(continuations, core, leaves) > 0xffffffff) {
        throw new Error("Opening continuation count is too large")
      }
      writeCount(continuationCounts, nodeIndex, continuations)
      writeCount(subtreeCore, nodeIndex, core)
      writeCount(subtreeLeaves, nodeIndex, leaves)
      if (nodeIndex % NODE_INTERVAL === 0) {
        const block = nodeIndex / NODE_INTERVAL
        candidateCheckpoints[block] = candidateStart
        firstPriorCheckpoints[block] = firstPriorStart
        branchCheckpoints[block] = branchStart
        edgeCheckpoints[block] = edgeStart
      }
    }
    if (nodeCount === 0 || nextTreeChild !== 0 || candidateStart || firstPriorStart || branchStart || edgeStart) {
      throw new Error("Opening node topology is incomplete")
    }
    return {
      candidateCheckpoints, firstPriorCheckpoints, branchCheckpoints, edgeCheckpoints,
      continuationCounts, subtreeCore, subtreeLeaves,
    }
  }

  function normalizeLoadedData(rawBuffer) {
    if (!(rawBuffer instanceof ArrayBuffer) || rawBuffer.byteLength < HEADER_SIZE) {
      throw new Error("Unsupported opening data format")
    }
    const view = new DataView(rawBuffer)
    if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) { throw new Error("Unsupported opening data format") }
    const boardSize = view.getUint16(3, true)
    const nodeCount = view.getUint32(5, true)
    const candidateCount = view.getUint32(9, true)
    const firstPriorCount = view.getUint32(13, true)
    const branchCount = view.getUint32(17, true)
    const redirectCount = view.getUint32(21, true)
    const largeCount = view.getUint32(25, true)
    const lastParentLimit = view.getUint32(29, true)
    const flagCount = view.getUint32(33, true)
    if (boardSize === 0) { throw new Error("Unsupported opening board size") }
    const wire = new Uint8Array(rawBuffer)
    const { nodes, offset: moveOffset } = decodeNodeRows(wire, nodeCount, largeCount, lastParentLimit, flagCount)
    const scoreOffset = moveOffset + 4 + view.getUint32(moveOffset, true)
    // Keep only scores and redirects from the wire buffer after decoding the moves.
    const bytes = wire.slice(scoreOffset)
    const priorHighOffset = Math.ceil(candidateCount / 4)
    const lead = indexResiduals(bytes, priorHighOffset + candidateCount, branchCount, LEAD_ESCAPE)
    const other = indexResiduals(bytes, lead.end, candidateCount - firstPriorCount, OTHER_ESCAPE)
    const redirects = decodeRedirects(bytes, other.end, nodeCount + redirectCount - 1, redirectCount)
    if (bytes.length !== redirects.offset) { throw new Error("Opening bundle size mismatch") }
    const frequencyCount = view.getUint32(moveOffset + 4, true)
    const moves = decodeOpeningMoves(wire, moveOffset + 8, frequencyCount, boardSize, nodes, candidateCount, redirects.bitmap)
    const indexes = indexGraph(nodes, candidateCount, firstPriorCount, branchCount, redirects)
    return {
      board_size: boardSize, bytes, nodes, nodeCount, candidateCount, redirectCount,
      moves, priorHighOffset, firstPriorCount,
      lead, other, redirects, ...indexes,
      moveNames: Array.from({ length: boardSize * boardSize }, (_, index) => cellIdToMove(index, boardSize)),
      nodeByLine: new Map(),
    }
  }

  function nodeCursors(data, nodeIndex) {
    const block = Math.floor(nodeIndex / NODE_INTERVAL)
    let candidate = data.candidateCheckpoints[block]
    let firstPrior = data.firstPriorCheckpoints[block]
    let branch = data.branchCheckpoints[block]
    let edge = data.edgeCheckpoints[block]
    for (let index = block * NODE_INTERVAL; index < nodeIndex; index += 1) {
      const word = data.nodes[index]
      const count = word & NODE_COUNT_MASK
      candidate += count
      firstPrior += count > 0
      branch += count > 1
      if (word & NODE_HAS_CHILDREN) { edge += count }
    }
    return { candidate, firstPrior, branch, edge }
  }

  function openingChildIndices(data, nodeIndex, edgeStart = nodeCursors(data, nodeIndex).edge) {
    const word = data.nodes[nodeIndex]
    const count = word & NODE_COUNT_MASK
    if (!(word & NODE_HAS_CHILDREN)) { return Array(count).fill(null) }
    const { bitmap, checkpoints, targets } = data.redirects
    let redirectIndex = bitmapRankBefore(bitmap, checkpoints, edgeStart, RANK_INTERVAL)
    let nextTreeChild = 1 + edgeStart - redirectIndex
    const children = new Array(count)
    for (let index = 0; index < count; index += 1) {
      children[index] = isRedirect(bitmap, edgeStart + index) ? targets[redirectIndex++] : nextTreeChild++
    }
    return children
  }

  function openingDescendantCount(data, nodeIndex) {
    if (!data || !Number.isInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= data.nodeCount) { return 0 }
    return readCount(data.continuationCounts, nodeIndex)
  }

  function openingMove(data, candidateIndex) { return data.moveNames[data.moves[candidateIndex]] }

  function readPrior(data, index) {
    const value = (data.bytes[index >>> 2] >>> ((index & 3) * 2)) & 3
    return (data.bytes[data.priorHighOffset + index] << 2) | value
  }

  function makeCandidate(move, line, childIndex, prior, redMetric, ply) {
    const moverMetric = redMetric === null || redMetric === OPTIONAL_NULL
      ? OPTIONAL_NULL
      : (ply & 1) ? 1000 - redMetric : redMetric
    return {
      move, childLine: line + move, childIndex, redMetric,
      prior: decodeOptionalThousandths(prior, OPTIONAL_NULL),
      tree_mover_winrate: decodeOptionalThousandths(moverMetric, OPTIONAL_NULL),
      elo_loss: null,
      retained: true,
    }
  }

  function applyResidual(prediction, code, bias) {
    if (prediction === null || prediction === OPTIONAL_NULL) { throw new Error("Unsupported opening candidate metric") }
    return prediction + code - bias
  }

  function decodeOpeningNode(data, nodeIndex, line, parentEdgeRedMetric, ply) {
    const cached = data.nodeByLine.get(line)
    if (cached) { return cached }
    const word = data.nodes[nodeIndex]
    const count = word & NODE_COUNT_MASK
    const candidates = new Array(count)
    if (count) {
      const { bytes, lead, other } = data
      const { candidate, firstPrior, branch, edge } = nodeCursors(data, nodeIndex)
      const children = openingChildIndices(data, nodeIndex, edge)
      const leadPrior = readPrior(data, firstPrior)
      let leadMetric = parentEdgeRedMetric
      if (count > 1) {
        const otherIndex = candidate - firstPrior
        let exception = -1
        let correction = 0
        let canPredict = leadPrior !== 0 && leadPrior !== OPTIONAL_NULL &&
          parentEdgeRedMetric !== null && parentEdgeRedMetric !== OPTIONAL_NULL
        for (let index = 1; index < count; index += 1) {
          const streamIndex = otherIndex + index - 1
          const prior = readPrior(data, data.firstPriorCount + streamIndex)
          const code = bytes[other.offset + streamIndex]
          let redMetric
          if (code === OTHER_ESCAPE) {
            if (exception < 0) { exception = exceptionRankBefore(bytes, other, streamIndex) }
            redMetric = packedU10(bytes, other.exceptionOffset, exception++)
          } else {
            redMetric = applyResidual(parentEdgeRedMetric, code, 127)
          }
          if (canPredict && prior !== OPTIONAL_NULL) {
            if (redMetric === OPTIONAL_NULL) { canPredict = false }
            else { correction += prior * (parentEdgeRedMetric - redMetric) }
          }
          candidates[index] = makeCandidate(openingMove(data, candidate + index), line, children[index], prior, redMetric, ply)
        }
        const code = packedNibble(bytes, lead.offset, branch)
        if (code === LEAD_ESCAPE) {
          leadMetric = packedU10(bytes, lead.exceptionOffset, exceptionRankBefore(bytes, lead, branch))
        } else {
          if (canPredict) { leadMetric += Math.floor((correction + Math.floor(leadPrior / 2)) / leadPrior) }
          leadMetric = applyResidual(leadMetric, code, 7)
        }
      }
      candidates[0] = makeCandidate(openingMove(data, candidate), line, children[0], leadPrior, leadMetric, ply)
    }
    const node = { index: nodeIndex, line, is_core: Boolean(word & NODE_IS_CORE), candidates }
    if (data.nodeByLine.size >= LINE_CACHE_CAPACITY) {
      data.nodeByLine.delete(data.nodeByLine.keys().next().value)
    }
    data.nodeByLine.set(line, node)
    return node
  }

  function openingRandomLineCount(data, kind) {
    return readCount(kind === "core" ? data.subtreeCore : data.subtreeLeaves, 0)
  }

  function openingLineForRandomRank(data, kind, rank) {
    const core = kind === "core"
    const weights = core ? data.subtreeCore : data.subtreeLeaves
    let nodeIndex = 0
    let line = ""
    while (true) {
      const word = data.nodes[nodeIndex]
      if (core && nodeIndex > 0 && (word & NODE_IS_CORE)) {
        if (rank === 0) { return line }
        rank -= 1
      }
      const { candidate, edge } = nodeCursors(data, nodeIndex)
      if (!(word & NODE_HAS_CHILDREN)) {
        return (word & NODE_COUNT_MASK) === 0 ? line : line + openingMove(data, candidate + rank)
      }
      const children = openingChildIndices(data, nodeIndex, edge)
      let selected = false
      for (let index = 0; index < children.length; index += 1) {
        const child = children[index]
        const weight = readCount(weights, child)
        if (rank < weight) {
          nodeIndex = child
          line += openingMove(data, candidate + index)
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
    openingRandomLineCount,
    openingLineForRandomRank,
  }
})()
