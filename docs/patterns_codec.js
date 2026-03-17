window.HexPatternCodec = (() => {
  const { bitmapRankBefore, bitmapRankCheckpoints, readAsciiMagic, readBitplaneValue, readUvarintAt } =
    window.HexStudyUI

  const BUNDLE_MAGIC = "HPB"
  const HEADER_SIZE = 17
  const PAIR_COUNT_SIZE = 2
  const PAIR_ROW_SIZE = 2
  const PACKED_STONE_FRACTION_BITS = 10
  const PACKED_OPTIONAL_NULL = 1023
  const PACKED_DEAD_CELL = 1001
  const PACKED_CAPTURED_CELL = 1002
  const FRACTION_MODE_AFFINE_ROW_COORD_RICE = 3
  const ROW_MEAN_PREDICTOR_SIZE = 5
  const PACKED_KEY_COORD_MIN = -8

  function pointKey(q, r) { return `${q},${r}` }

  function pointsEqual(a, b) { return comparePoints(a, b) === 0 }

  function comparePoints(a, b) {
    if (a[0] !== b[0]) { return a[0] - b[0] }
    return a[1] - b[1]
  }

  function sortPoints(points) { return [...points].sort(comparePoints) }

  function comparePointLists(a, b) {
    const limit = Math.min(a.length, b.length)
    for (let i = 0; i < limit; i += 1) {
      const cmp = comparePoints(a[i], b[i])
      if (cmp !== 0) { return cmp }
    }
    return a.length - b.length
  }

  function comparePatterns(aPlus, aMinus, bPlus, bMinus) {
    const plusCmp = comparePointLists(aPlus, bPlus)
    if (plusCmp !== 0) { return plusCmp }
    return comparePointLists(aMinus, bMinus)
  }

  function oppositeColor(color) { return color === "red" ? "blue" : "red" }

  function parseInteger(text) {
    if (!/^-?[0-9]+$/.test(text)) { throw new Error(`Bad integer '${text}'`) }
    return Number.parseInt(text, 10)
  }

  function parsePointBlock(body) {
    if (body === "") { return [] }
    const points = []
    for (const token of body.split(":")) {
      if (!token.includes(",")) { throw new Error(`Bad point '${token}'`) }
      const [qText, rText] = token.split(",", 2)
      const point = [parseInteger(qText), parseInteger(rText)]
      if (points.some((other) => comparePoints(other, point) === 0)) { throw new Error("Duplicate point in block") }
      points.push(point)
    }
    return sortPoints(points)
  }

  function readDelimitedBlock(text, startIndex, openChar, closeChar) {
    if (text[startIndex] !== openChar) { throw new Error(`Expected '${openChar}' at position ${startIndex}`) }
    const closeIndex = text.indexOf(closeChar, startIndex + 1)
    if (closeIndex < 0) { throw new Error(`Missing closing '${closeChar}'`) }
    return { points: parsePointBlock(text.slice(startIndex + 1, closeIndex)), nextIndex: closeIndex + 1 }
  }

  function assertNoCrossBlockOverlap(plusPoints, minusPoints) {
    const seen = new Set()
    for (const point of [...plusPoints, ...minusPoints]) {
      const key = pointKey(point[0], point[1])
      if (seen.has(key)) { throw new Error("Cross-block overlap is not allowed") }
      seen.add(key)
    }
  }

  function parsePattern(rawText) {
    const text = String(rawText || "").replace(/\s+/g, "")
    if (!text) { throw new Error("Empty pattern") }
    if (!text.startsWith("+")) { throw new Error("Pattern must use labeled notation and start with '+'") }
    const plusBlock = readDelimitedBlock(text, 1, "[", "]")
    if (text[plusBlock.nextIndex] !== "-") { throw new Error("Expected '-' after plus block") }
    const minusBlock = readDelimitedBlock(text, plusBlock.nextIndex + 1, "[", "]")
    if (minusBlock.nextIndex !== text.length) { throw new Error("Trailing characters after pattern") }
    assertNoCrossBlockOverlap(plusBlock.points, minusBlock.points)
    return { plus: plusBlock.points, minus: minusBlock.points }
  }

  function parseHashPattern(rawHash) {
    const text = String(rawHash || "").replace(/\s+/g, "")
    const hashText = text.startsWith("#") ? text.slice(1) : text
    if (!hashText) { throw new Error("Empty hash pattern") }
    const plusBlock = readDelimitedBlock(hashText, 0, "(", ")")
    const minusBlock = readDelimitedBlock(hashText, plusBlock.nextIndex, "(", ")")
    if (minusBlock.nextIndex !== hashText.length) { throw new Error("Trailing characters after hash pattern") }
    assertNoCrossBlockOverlap(plusBlock.points, minusBlock.points)
    return { plus: plusBlock.points, minus: minusBlock.points }
  }

  function moveCountForPattern(patternText) {
    const parsed = parsePattern(patternText)
    return moveCountForStoneCounts(parsed.plus.length, parsed.minus.length)
  }

  function moveCountForStoneCounts(plusCount, minusCount) {
    const diff = labeledFamilyDiff(plusCount, minusCount)
    if (diff === null) {
      throw new Error("Pattern is not a supported labeled family under red-first play with at most one tenuki")
    }
    return plusCount + minusCount + (diff === -1 || diff === 2 ? 1 : 0)
  }

  const PATTERN_TRANSFORMS = [
    [1, 0, 0, 1],
    [0, -1, 1, 1],
    [-1, -1, 1, 0],
    [-1, 0, 0, -1],
    [0, 1, -1, -1],
    [1, 1, -1, 0],
    [1, 0, -1, -1],
    [0, -1, -1, 0],
    [-1, -1, 0, 1],
    [-1, 0, 1, 1],
    [0, 1, 1, 0],
    [1, 1, 0, -1],
  ]

  function applyTransformAx(point, transformId) {
    if (!(transformId >= 0 && transformId < 12)) { throw new Error(`Bad transform id: ${transformId}`) }
    const [a, b, c, d] = PATTERN_TRANSFORMS[transformId]
    return [a * point[0] + b * point[1], c * point[0] + d * point[1]]
  }

  function inverseTransformId(transformId) {
    if (transformId >= 0 && transformId < 6) { return (6 - transformId) % 6 }
    if (transformId >= 6 && transformId < 12) { return transformId }
    throw new Error(`Bad transform id: ${transformId}`)
  }

  function normalizeLabeledPoints(plus, minus) {
    const all = [...plus, ...minus]
    if (all.length === 0) { throw new Error("Pattern has no stones") }
    const anchor = all.reduce((best, point) => (comparePoints(point, best) < 0 ? point : best))
    const shift = (point) => [point[0] - anchor[0], point[1] - anchor[1]]
    return { plus: sortPoints(plus.map(shift)), minus: sortPoints(minus.map(shift)), anchor: [anchor[0], anchor[1]] }
  }

  function formatPointList(points) { return points.map((point) => `${point[0]},${point[1]}`).join(":") }

  function formatPattern(plus, minus) { return `+[${formatPointList(plus)}]-[${formatPointList(minus)}]` }

  function formatHashPattern(plus, minus) { return `(${formatPointList(plus)})(${formatPointList(minus)})` }

  function canonicalizeLabeledPattern(plus, minus) {
    let best = null
    for (let transformId = 0; transformId < 12; transformId += 1) {
      const plusT = plus.map((point) => applyTransformAx(point, transformId))
      const minusT = minus.map((point) => applyTransformAx(point, transformId))
      const normalized = normalizeLabeledPoints(plusT, minusT)
      if (best === null || comparePatterns(normalized.plus, normalized.minus, best.plus, best.minus) < 0) {
        best = {
          pattern: formatPattern(normalized.plus, normalized.minus),
          plus: normalized.plus,
          minus: normalized.minus,
          transformId,
          anchor: normalized.anchor,
        }
      }
    }
    return best
  }

  function shiftPoints(points, dq, dr) { return points.map((point) => [point[0] + dq, point[1] + dr]) }

  function labeledFamilyDiff(plusCount, minusCount) {
    const diff = minusCount - plusCount
    return diff >= -1 && diff <= 2 ? diff : null
  }

  function inferBoardFromLabeledPattern(plus, minus) {
    const diff = labeledFamilyDiff(plus.length, minus.length)
    if (diff !== null) {
      return {
        toPlay: diff <= 0 ? "red" : "blue",
        red: sortPoints(diff <= 0 ? plus : minus),
        blue: sortPoints(diff <= 0 ? minus : plus),
        tenukiOwner: diff === -1 || diff === 2 ? "blue" : null,
      }
    }
    throw new Error("Pattern is not a supported labeled family under red-first play with at most one tenuki")
  }

  function assertPatternBundleSize(condition) {
    if (!condition) { throw new Error("Pattern bundle size mismatch") }
  }

  function compactKeyCoordText(word) {
    const q = (word & 0x0f) + PACKED_KEY_COORD_MIN
    const r = ((word >> 4) & 0x0f) + PACKED_KEY_COORD_MIN
    return `${q},${r}`
  }

  function decodeCompactPatternKey(keyBlob, keyOffset, keyLength) {
    if (keyLength < 1) { throw new Error("Unsupported pattern key slice") }
    const countByte = keyBlob[keyOffset]
    const plusCount = countByte & 0x0f
    const minusCount = (countByte >> 4) & 0x0f
    if (keyLength !== 1 + plusCount + minusCount) { throw new Error("Unsupported pattern key slice") }
    const plus = []
    const minus = []
    let offset = keyOffset + 1
    for (let idx = 0; idx < plusCount; idx += 1) {
      plus.push(compactKeyCoordText(keyBlob[offset]))
      offset += 1
    }
    for (let idx = 0; idx < minusCount; idx += 1) {
      minus.push(compactKeyCoordText(keyBlob[offset]))
      offset += 1
    }
    return `+[${plus.join(":")}]-[${minus.join(":")}]`
  }

  function compactPatternKeyToPlay(keyBytes) {
    if (!keyBytes || keyBytes.length < 1) { throw new Error("Unsupported pattern key slice") }
    const plusCount = keyBytes[0] & 0x0f
    const minusCount = (keyBytes[0] >> 4) & 0x0f
    return minusCount - plusCount <= 0 ? "red" : "blue"
  }

  function concatPatternKeyPrefix(prevKey, prefixLength, suffix) {
    if (prefixLength < 0 || prefixLength > prevKey.length) { throw new Error("Unsupported pattern key slice") }
    const key = new Uint8Array(prefixLength + suffix.length)
    key.set(prevKey.slice(0, prefixLength), 0)
    key.set(suffix, prefixLength)
    return key
  }

  function unzigzag(value) { return value % 2 === 0 ? value / 2 : -((value + 1) / 2) }

  function roundRatioHalfEven(numerator, denominator) {
    assertPatternBundleSize(Number.isInteger(numerator) && Number.isInteger(denominator) && denominator > 0)
    let quotient = Math.floor(numerator / denominator)
    const doubledRemainder = (numerator - quotient * denominator) * 2
    if (doubledRemainder > denominator || (doubledRemainder === denominator && quotient % 2 === 1)) { quotient += 1 }
    return quotient
  }

  // Rank checkpoints bound bitmap scans; Rice checkpoints bound unary scans.
  const PATTERN_RANK_INTERVAL = 256
  const RICE_SEEK_INTERVAL = 32
  const RICE_ZERO_COUNTS = new Uint8Array(256)
  const RICE_ZERO_BITS = new Uint8Array(256 * 8)
  for (let byte = 0; byte < 256; byte += 1) {
    let count = 0
    for (let bit = 0; bit < 8; bit += 1) {
      if (!(byte & (1 << bit))) { RICE_ZERO_BITS[(byte << 3) + count++] = bit }
    }
    RICE_ZERO_COUNTS[byte] = count
  }

  function createGeometryCandidates(pairTable, distances) {
    const wordCount = Math.ceil(pairTable.length / 32)
    const neighborhoods = distances.map(() => new Uint32Array(256 * wordCount))
    const occupied = new Int16Array(256).fill(-1)
    for (let code = 0; code < 256; code += 1) {
      const q = (code & 15) - 8
      const r = (code >> 4) - 8
      for (let pairIndex = 0; pairIndex < pairTable.length; pairIndex += 1) {
        const dq = q - pairTable[pairIndex][0]
        const dr = r - pairTable[pairIndex][1]
        const distance = dq * dq + dq * dr + dr * dr
        const offset = code * wordCount + (pairIndex >>> 5)
        for (let profile = 0; profile < distances.length; profile += 1) {
          if (distance <= distances[profile]) { neighborhoods[profile][offset] |= 1 << (pairIndex & 31) }
        }
        if (distance === 0) occupied[code] = pairIndex
      }
    }
    return (key, stabilizerMask) => {
      const words = new Uint32Array(wordCount)
      let minQ = Infinity
      let minR = Infinity
      let minS = Infinity
      let maxQ = -Infinity
      let maxR = -Infinity
      let maxS = -Infinity
      for (let stone = 1; stone < key.length; stone += 1) {
        const q = (key[stone] & 15) - 8
        const r = (key[stone] >> 4) - 8
        minQ = Math.min(minQ, q)
        maxQ = Math.max(maxQ, q)
        minR = Math.min(minR, r)
        maxR = Math.max(maxR, r)
        minS = Math.min(minS, q + r)
        maxS = Math.max(maxS, q + r)
      }
      // Wide patterns use a tighter neighborhood; singleton patterns use the widest.
      const span = Math.max(maxQ - minQ, maxR - minR, maxS - minS)
      const neighborhood = neighborhoods[key.length === 2 ? 0 : span === 4 ? 2 : 1]
      for (let stone = 1; stone < key.length; stone += 1) {
        const offset = key[stone] * wordCount
        for (let word = 0; word < wordCount; word += 1) { words[word] |= neighborhood[offset + word] }
      }
      for (let stone = 1; stone < key.length; stone += 1) {
        const pairIndex = occupied[key[stone]]
        if (pairIndex >= 0) words[pairIndex >>> 5] &= ~(1 << (pairIndex & 31))
      }
      // Keep only the smallest coordinate in each orbit of a labeled symmetry.
      while (stabilizerMask) {
        const transform = PATTERN_TRANSFORMS[31 - Math.clz32(stabilizerMask & -stabilizerMask)]
        stabilizerMask &= stabilizerMask - 1
        let anchorQ = Infinity
        let anchorR = Infinity
        for (let stone = 1; stone < key.length; stone += 1) {
          const q = (key[stone] & 15) - 8
          const r = (key[stone] >> 4) - 8
          const x = transform[0] * q + transform[1] * r
          const y = transform[2] * q + transform[3] * r
          if (x < anchorQ || (x === anchorQ && y < anchorR)) {
            anchorQ = x
            anchorR = y
          }
        }
        for (let wordIndex = 0; wordIndex < wordCount; wordIndex += 1) {
          let word = words[wordIndex]
          while (word) {
            const bit = 31 - Math.clz32(word & -word)
            word &= word - 1
            const [q, r] = pairTable[(wordIndex << 5) + bit]
            const x = transform[0] * q + transform[1] * r - anchorQ
            const y = transform[2] * q + transform[3] * r - anchorR
            if (x < q || (x === q && y < r)) words[wordIndex] &= ~(1 << bit)
          }
        }
      }
      return words
    }
  }

  function readRiceValue(view, fractionData, cellCount, index) {
    const start = fractionData.riceDirectory[Math.floor(index / RICE_SEEK_INTERVAL)]
    let remaining = index % RICE_SEEK_INTERVAL
    let byteIndex = start >>> 3
    let byte = fractionData.unary[byteIndex] | ((1 << (start & 7)) - 1)
    let count = RICE_ZERO_COUNTS[byte]
    while (remaining >= count) {
      remaining -= count
      byte = fractionData.unary[++byteIndex]
      count = RICE_ZERO_COUNTS[byte]
    }
    const zero = byteIndex * 8 + RICE_ZERO_BITS[(byte << 3) + remaining]
    let quotient = 0
    for (let bit = zero - 1; bit >= 0 && fractionData.unary[bit >>> 3] & (1 << (bit & 7)); bit -= 1) { quotient += 1 }
    return (
      (quotient << fractionData.riceBits) |
      readBitplaneValue(view, fractionData.remainderOffset, cellCount, index, fractionData.riceBits)
    )
  }

  function normalizeLoadedData(rawBuffer) {
    if (!(rawBuffer instanceof ArrayBuffer) || rawBuffer.byteLength < HEADER_SIZE) {
      throw new Error("Unsupported pattern data format")
    }
    const view = new DataView(rawBuffer)
    if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) { throw new Error("Unsupported pattern data format") }
    const fractionMode = view.getUint16(3, true)
    if (fractionMode !== FRACTION_MODE_AFFINE_ROW_COORD_RICE) { throw new Error("Unsupported pattern fraction stream") }
    const patternCount = view.getUint32(5, true)
    const cellCount = view.getUint32(9, true)
    const keyStreamSize = view.getUint32(13, true)
    const tenukiOffset = HEADER_SIZE
    const pairCountOffset = tenukiOffset + Math.ceil((patternCount * PACKED_STONE_FRACTION_BITS) / 8)
    assertPatternBundleSize(rawBuffer.byteLength >= pairCountOffset + PAIR_COUNT_SIZE)
    const pairCount = view.getUint16(pairCountOffset, true)
    if (cellCount > 0 && pairCount === 0) { throw new Error("Unsupported pattern pair table") }
    const pairTableOffset = pairCountOffset + PAIR_COUNT_SIZE
    const geometrySizeOffset = pairTableOffset + pairCount * PAIR_ROW_SIZE
    assertPatternBundleSize(rawBuffer.byteLength >= geometrySizeOffset + 4)
    const geometryByteLength = view.getUint32(geometrySizeOffset, true)
    const geometryOffset = geometrySizeOffset + 4
    const fractionOffset = geometryOffset + geometryByteLength
    const keyStreamOffset = rawBuffer.byteLength - keyStreamSize
    assertPatternBundleSize(keyStreamOffset >= fractionOffset)
    const pairTable = []
    for (let idx = 0; idx < pairCount; idx += 1) {
      const offset = pairTableOffset + idx * PAIR_ROW_SIZE
      pairTable.push([view.getInt8(offset), view.getInt8(offset + 1)])
    }
    const geometryBytes = new Uint8Array(rawBuffer, geometryOffset, geometryByteLength)
    const geometryView = new DataView(rawBuffer, geometryOffset, geometryByteLength)
    const candidatesFor = createGeometryCandidates(pairTable, Array.from(geometryBytes.subarray(0, 3)))
    const stabilizers = new Uint16Array(patternCount)
    let geometryCursor = 5
    let stabilizerRow = -1
    for (let index = 0; index < geometryView.getUint16(3, true); index += 1) {
      const gap = readUvarintAt(geometryBytes, geometryCursor)
      stabilizerRow += gap.value
      stabilizers[stabilizerRow] = geometryView.getUint16(gap.offset, true)
      geometryCursor = gap.offset + 2
    }
    const coordRows = pairTable.map((pair) => ({
      pair,
      bitmap: new Uint8Array(Math.ceil(patternCount / 8)),
      fractionStart: 0,
    }))
    const fractionData = parseFractionData(view, fractionOffset, keyStreamOffset, patternCount, pairCount, cellCount)
    const keyStream = new Uint8Array(rawBuffer, keyStreamOffset, keyStreamSize)
    const patterns = Object.create(null)
    const randomPatterns = { le4: [], "5": [], "6": [] }
    const entries = []
    const entryCache = new Map()
    let keyOffset = 0
    let prevKey = new Uint8Array()
    for (let idx = 0; idx < patternCount; idx += 1) {
      const tenukiStoneFraction = readBitplaneValue(view, tenukiOffset, patternCount, idx, PACKED_STONE_FRACTION_BITS)
      const prefixLength = keyStream[keyOffset++]
      const countByte = prefixLength ? prevKey[0] : keyStream[keyOffset]
      const suffixLength = 1 + (countByte & 15) + (countByte >> 4) - prefixLength
      const suffix = keyStream.subarray(keyOffset, keyOffset + suffixLength)
      keyOffset += suffixLength
      const keyBytes = concatPatternKeyPrefix(prevKey, prefixLength, suffix)
      const words = candidatesFor(keyBytes, stabilizers[idx])
      for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
        let word = words[wordIndex]
        while (word) {
          const bit = 31 - Math.clz32(word & -word)
          word &= word - 1
          coordRows[(wordIndex << 5) + bit].bitmap[idx >>> 3] |= 1 << (idx & 7)
        }
      }
      const pattern = decodeCompactPatternKey(keyBytes, 0, keyBytes.length)
      const moveCount = moveCountForStoneCounts(keyBytes[0] & 0x0f, (keyBytes[0] >> 4) & 0x0f)
      const bucket = moveCount <= 4 ? "le4" : String(moveCount)
      randomPatterns[bucket].push(pattern)
      entries.push({ toPlay: compactPatternKeyToPlay(keyBytes), tenukiStoneFraction })
      patterns[pattern] = idx
      prevKey = keyBytes
    }
    let fractionStart = 0
    for (const row of coordRows) {
      row.fractionStart = fractionStart
      row.checkpoints = bitmapRankCheckpoints(row.bitmap, patternCount, PATTERN_RANK_INTERVAL)
      fractionStart += row.checkpoints[row.checkpoints.length - 1]
    }
    assertPatternBundleSize(fractionStart === cellCount)
    assertPatternBundleSize(keyOffset === keyStream.length)
    return {
      pattern_count: patternCount,
      rawBuffer,
      coordRows,
      fractionData,
      cellCount,
      entries,
      entryCache,
      patterns,
      randomPatterns,
    }
  }

  function parseFractionData(view, fractionOffset, keyStreamOffset, patternCount, pairCount, cellCount) {
    assertPatternBundleSize(fractionOffset + ROW_MEAN_PREDICTOR_SIZE <= keyStreamOffset)
    const rowMeanNumerator = view.getUint8(fractionOffset)
    const rowMeanDenominator = view.getUint8(fractionOffset + 1)
    const rowMeanIntercept = view.getInt16(fractionOffset + 2, true)
    const rowMeanResidualMagnitudeBits = view.getUint8(fractionOffset + 4)
    assertPatternBundleSize(rowMeanDenominator > 0 && rowMeanResidualMagnitudeBits > 0)
    const rowMeanResidualOffset = fractionOffset + ROW_MEAN_PREDICTOR_SIZE
    let offset = rowMeanResidualOffset + Math.ceil((patternCount * (rowMeanResidualMagnitudeBits + 1)) / 8)
    const pairMeanOffset = offset
    offset += Math.ceil((pairCount * PACKED_STONE_FRACTION_BITS) / 8)
    assertPatternBundleSize(offset + 3 <= keyStreamOffset)
    const globalMean = view.getUint16(offset, true)
    offset += 2
    const riceBits = view.getUint8(offset++)
    const remainderOffset = offset
    offset += Math.ceil((cellCount * riceBits) / 8)
    const unary = new Uint8Array(view.buffer, view.byteOffset + offset, keyStreamOffset - offset)
    const riceDirectory = new Uint32Array(Math.ceil(cellCount / RICE_SEEK_INTERVAL))
    let symbol = 0
    let previousZero = -1
    for (let byte = 0; byte < unary.length && symbol < cellCount; byte += 1) {
      for (let bit = 0; bit < 8 && symbol < cellCount; bit += 1) {
        if (!(unary[byte] & (1 << bit))) {
          if (symbol % RICE_SEEK_INTERVAL === 0) riceDirectory[symbol / RICE_SEEK_INTERVAL] = previousZero + 1
          previousZero = byte * 8 + bit
          symbol += 1
        }
      }
    }
    return {
      rowMeanNumerator,
      rowMeanDenominator,
      rowMeanIntercept,
      rowMeanResidualMagnitudeBits,
      rowMeanResidualOffset,
      pairMeanOffset,
      globalMean,
      riceBits,
      remainderOffset,
      unary,
      riceDirectory,
    }
  }

  function patternEntryForLookupInData(data, pattern) {
    if (!data || !data.patterns || !Object.prototype.hasOwnProperty.call(data.patterns, pattern)) { return null }
    const entryIndex = data.patterns[pattern]
    if (data.entryCache.has(entryIndex)) { return data.entryCache.get(entryIndex) }
    const rawEntry = data.entries[entryIndex]
    if (!rawEntry) { throw new Error("Unsupported pattern entry") }
    const view = new DataView(data.rawBuffer)
    const cells = []
    const rowMeanResidualCode = readBitplaneValue(
      view,
      data.fractionData.rowMeanResidualOffset,
      data.pattern_count,
      entryIndex,
      data.fractionData.rowMeanResidualMagnitudeBits + 1,
    )
    const rowMeanSignBit = 2 ** data.fractionData.rowMeanResidualMagnitudeBits
    const rowMeanMagnitude = rowMeanResidualCode % rowMeanSignBit
    const rowMeanResidual = rowMeanResidualCode >= rowMeanSignBit ? -rowMeanMagnitude : rowMeanMagnitude
    const rowMean =
      roundRatioHalfEven(
        data.fractionData.rowMeanNumerator * rawEntry.tenukiStoneFraction,
        data.fractionData.rowMeanDenominator,
      ) +
      data.fractionData.rowMeanIntercept +
      rowMeanResidual
    assertPatternBundleSize(rowMean >= 0 && rowMean < PACKED_OPTIONAL_NULL)
    for (let rowIdx = 0; rowIdx < data.coordRows.length; rowIdx += 1) {
      const row = data.coordRows[rowIdx]
      if (!(row.bitmap[entryIndex >>> 3] & (1 << (entryIndex & 7)))) { continue }
      const fractionIndex =
        row.fractionStart + bitmapRankBefore(row.bitmap, row.checkpoints, entryIndex, PATTERN_RANK_INTERVAL)
      assertPatternBundleSize(fractionIndex >= 0 && fractionIndex < data.cellCount)
      const pairMean = readBitplaneValue(
        view,
        data.fractionData.pairMeanOffset,
        data.coordRows.length,
        rowIdx,
        PACKED_STONE_FRACTION_BITS,
      )
      const residualCode = readRiceValue(view, data.fractionData, data.cellCount, fractionIndex)
      const stoneFraction =
        residualCode < 2
          ? PACKED_DEAD_CELL + residualCode
          : rowMean + pairMean - data.fractionData.globalMean + unzigzag(residualCode - 2)
      assertPatternBundleSize(stoneFraction >= 0 && stoneFraction < PACKED_OPTIONAL_NULL)
      const pair = row.pair
      cells.push([pair[0], pair[1], stoneFraction])
    }
    const entry = { p: rawEntry.toPlay, c: cells }
    if (rawEntry.tenukiStoneFraction !== PACKED_OPTIONAL_NULL) { entry.t = rawEntry.tenukiStoneFraction }
    data.entryCache.set(entryIndex, entry)
    return entry
  }

  return {
    PACKED_DEAD_CELL,
    PACKED_CAPTURED_CELL,
    pointKey,
    pointsEqual,
    comparePoints,
    sortPoints,
    oppositeColor,
    parsePattern,
    parseHashPattern,
    moveCountForPattern,
    applyTransformAx,
    inverseTransformId,
    formatPattern,
    formatHashPattern,
    canonicalizeLabeledPattern,
    shiftPoints,
    labeledFamilyDiff,
    inferBoardFromLabeledPattern,
    normalizeLoadedData,
    patternEntryForLookupInData,
  }
})()
