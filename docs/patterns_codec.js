window.HexPatternCodec = (() => {
  const { readAsciiMagic, readBitplaneValue, readUvarintAt } = window.HexStudyUI

  const BUNDLE_MAGIC = "HPB"
  const HEADER_SIZE = 19
  const PAIR_COUNT_SIZE = 2
  const PAIR_ROW_SIZE = 2
  const PACKED_STONE_FRACTION_BITS = 10
  const PACKED_OPTIONAL_NULL = 1023
  const PACKED_DEAD_CELL = 1001
  const PACKED_CAPTURED_CELL = 1002
  const ROW_MEAN_PREDICTOR_SIZE = 5
  const PACKED_KEY_COORD_MIN = -8
  const KEY_STONES = 6
  const KEY_BASE = 128
  // Six seven-bit coordinates and the count byte fit exactly in one Number.
  const KEY_COUNT_PLACE = KEY_BASE ** KEY_STONES
  const DISTANCE_CUTOFFS = [1, 3, 4, 7, 9, 12]
  const DISTANCE_CLASSES = DISTANCE_CUTOFFS.length + 1
  const DISTANCE_CONTEXTS = DISTANCE_CLASSES ** 2
  const FARTHEST_BINS = 0x11111111 * (DISTANCE_CLASSES - 1)
  const BIAS_SCALE = 6
  const RANK_WORDS = 4

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
      const coordinates = token.split(",")
      if (coordinates.length !== 2) { throw new Error(`Bad point '${token}'`) }
      const [qText, rText] = coordinates
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

  function parseHashPattern(rawText) {
    const text = String(rawText || "").replace(/\s+/g, "")
    if (!text) { throw new Error("Empty hash pattern") }
    const plusBlock = readDelimitedBlock(text, 0, "(", ")")
    const minusBlock = readDelimitedBlock(text, plusBlock.nextIndex, "(", ")")
    if (minusBlock.nextIndex !== text.length) { throw new Error("Trailing characters after hash pattern") }
    assertNoCrossBlockOverlap(plusBlock.points, minusBlock.points)
    return { plus: plusBlock.points, minus: minusBlock.points }
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
          plus: normalized.plus,
          minus: normalized.minus,
          transformId,
          anchor: normalized.anchor,
        }
      }
    }
    return { ...best, pattern: formatPattern(best.plus, best.minus) }
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

  function keyDigit(place, q, r) {
    return place * KEY_BASE + ((r - PACKED_KEY_COORD_MIN) << 3) + q
  }

  function patternIndexForLookup(keys, { plus, minus }) {
    const count = plus.length + minus.length
    if (count > KEY_STONES) { return -1 }
    let key = plus.length | (minus.length << 4)
    for (const points of [plus, minus]) {
      for (const [q, r] of points) {
        if (q < 0 || q >= 8 || r < PACKED_KEY_COORD_MIN || r >= 8) { return -1 }
        key = keyDigit(key, q, r)
      }
    }
    key *= KEY_BASE ** (KEY_STONES - count)
    let lo = 0
    let hi = keys.length
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      const value = keys[mid]
      if (value === key) { return mid }
      if (value < key) lo = mid + 1
      else hi = mid
    }
    return -1
  }

  function patternForIndex(data, index) {
    const key = data.keys[index]
    const countByte = Math.floor(key / KEY_COUNT_PLACE)
    const plus = new Array(countByte & 15)
    const minus = new Array(countByte >> 4)
    let place = KEY_COUNT_PLACE
    for (const points of [plus, minus]) {
      for (let i = 0; i < points.length; i += 1) {
        place /= KEY_BASE
        const digit = Math.floor(key / place) & (KEY_BASE - 1)
        points[i] = `${digit & 7},${(digit >> 3) + PACKED_KEY_COORD_MIN}`
      }
    }
    return `+[${plus.join(":")}]-[${minus.join(":")}]`
  }

  function randomPatternForBucket(data, bucket, current) {
    const ranges = data?.randomRanges[bucket]
    const count = data?.randomCounts[bucket] || 0
    if (!count) { return null }
    function patternAtRank(rank) {
      for (const [start, length] of ranges) {
        if (rank < length) { return patternForIndex(data, start + rank) }
        rank -= length
      }
    }
    const rank = Math.floor(Math.random() * count)
    const pattern = patternAtRank(rank)
    if (count === 1 || pattern !== current) { return pattern }
    return patternAtRank((rank + 1 + Math.floor(Math.random() * (count - 1))) % count)
  }

  function unzigzag(value) { return value % 2 === 0 ? value / 2 : -((value + 1) / 2) }

  function roundRatioHalfEven(numerator, denominator) {
    assertPatternBundleSize(Number.isInteger(numerator) && Number.isInteger(denominator) && denominator > 0)
    let quotient = Math.floor(numerator / denominator)
    const doubledRemainder = (numerator - quotient * denominator) * 2
    if (doubledRemainder > denominator || (doubledRemainder === denominator && quotient % 2 === 1)) { quotient += 1 }
    return quotient
  }

  function popcount(word) {
    word -= (word >>> 1) & 0x55555555
    word = (word & 0x33333333) + ((word >>> 2) & 0x33333333)
    return Math.imul((word + (word >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24
  }

  function transposeCandidateBlock(block, pairCount, stride, output, wordIndex) {
    for (let start = 0; start < pairCount; start += 32) {
      let mask = 0x0000ffff
      // Swap row and column index bits for a 32-by-32 occupancy tile.
      for (let shift = 16; shift > 0; shift >>>= 1, mask ^= mask << shift) {
        for (let row = 0; row < 32; row = (row + shift + 1) & ~shift) {
          const left = start + row
          const right = left + shift
          const swap = ((block[left] >>> shift) ^ block[right]) & mask
          block[left] ^= swap << shift
          block[right] ^= swap
        }
      }
      const end = Math.min(start + 32, pairCount)
      for (let i = start; i < end; i += 1) { output[i * stride + wordIndex] = block[i] }
    }
    block.fill(0)
  }

  function createPatternGeometry(pairTable, distances) {
    const wordCount = Math.ceil(pairTable.length / 32)
    const neighborhoods = Array.from(distances, () => new Uint32Array(KEY_BASE * wordCount))
    const occupied = new Int16Array(KEY_BASE).fill(-1)
    // One nibble per coordinate leaves a guard bit for parallel minima.
    const distanceStride = Math.ceil(pairTable.length / 8)
    const distanceBins = new Uint32Array(distanceStride * KEY_BASE)
    for (let code = 0; code < KEY_BASE; code += 1) {
      const q = code & 7
      const r = (code >> 3) + PACKED_KEY_COORD_MIN
      for (let pairIndex = 0; pairIndex < pairTable.length; pairIndex += 1) {
        const dq = q - pairTable[pairIndex][0]
        const dr = r - pairTable[pairIndex][1]
        const distance = dq * dq + dq * dr + dr * dr
        const offset = code * wordCount + (pairIndex >>> 5)
        for (let profile = 0; profile < distances.length; profile += 1) {
          if (distance <= distances[profile]) { neighborhoods[profile][offset] |= 1 << (pairIndex & 31) }
        }
        let bin = 0
        while (bin < DISTANCE_CUTOFFS.length && distance > DISTANCE_CUTOFFS[bin]) { bin += 1 }
        distanceBins[code * distanceStride + (pairIndex >>> 3)] |= bin << ((pairIndex & 7) * 4)
        if (distance === 0) occupied[code] = pairIndex
      }
    }
    const words = new Uint32Array(wordCount)
    function candidatesFor(key, keyLength, stabilizerMask) {
      words.fill(0)
      let minQ = Infinity
      let minR = Infinity
      let minS = Infinity
      let maxQ = -Infinity
      let maxR = -Infinity
      let maxS = -Infinity
      for (let stone = 1; stone < keyLength; stone += 1) {
        const q = key[stone] & 7
        const r = (key[stone] >> 3) + PACKED_KEY_COORD_MIN
        minQ = Math.min(minQ, q)
        maxQ = Math.max(maxQ, q)
        minR = Math.min(minR, r)
        maxR = Math.max(maxR, r)
        minS = Math.min(minS, q + r)
        maxS = Math.max(maxS, q + r)
      }
      // Wide patterns use a tighter neighborhood; singleton patterns use the widest.
      const span = Math.max(maxQ - minQ, maxR - minR, maxS - minS)
      const neighborhood = neighborhoods[keyLength === 2 ? 0 : span === 4 ? 2 : 1]
      for (let stone = 1; stone < keyLength; stone += 1) {
        const offset = key[stone] * wordCount
        for (let word = 0; word < wordCount; word += 1) { words[word] |= neighborhood[offset + word] }
      }
      for (let stone = 1; stone < keyLength; stone += 1) {
        const pairIndex = occupied[key[stone]]
        if (pairIndex >= 0) words[pairIndex >>> 5] &= ~(1 << (pairIndex & 31))
      }
      // Keep only the smallest coordinate in each orbit of a labeled symmetry.
      while (stabilizerMask) {
        const transform = PATTERN_TRANSFORMS[31 - Math.clz32(stabilizerMask & -stabilizerMask)]
        stabilizerMask &= stabilizerMask - 1
        let anchorQ = Infinity
        let anchorR = Infinity
        for (let stone = 1; stone < keyLength; stone += 1) {
          const q = key[stone] & 7
          const r = (key[stone] >> 3) + PACKED_KEY_COORD_MIN
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
    return { candidatesFor, distanceBins, distanceStride }
  }

  function decodeStoneFraction(code, prediction) {
    if (code < 2) { return PACKED_DEAD_CELL + code }
    if (code < 4) { return 1002 - code }
    prediction = Math.max(0, Math.min(998, prediction))
    const radius = Math.min(prediction, 998 - prediction)
    const delta = code - 4
    if (delta <= 2 * radius) { return prediction + unzigzag(delta) }
    return prediction + (prediction < 499 ? delta - radius : radius - delta)
  }

  function normalizeLoadedData(rawBuffer) {
    if (!(rawBuffer instanceof ArrayBuffer) || rawBuffer.byteLength < HEADER_SIZE) {
      throw new Error("Unsupported pattern data format")
    }
    const view = new DataView(rawBuffer)
    if (readAsciiMagic(view, 0, 3) !== BUNDLE_MAGIC) { throw new Error("Unsupported pattern data format") }
    const patternCount = view.getUint32(3, true)
    const corpusCount = view.getUint32(7, true)
    const cellCount = view.getUint32(11, true)
    const keyStreamSize = view.getUint32(15, true)
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
    const { candidatesFor, distanceBins, distanceStride } = createPatternGeometry(pairTable, geometryBytes.subarray(0, 3))
    const stabilizers = new Uint16Array(patternCount)
    let geometryCursor = 5
    let stabilizerRow = -1
    for (let index = 0; index < geometryView.getUint16(3, true); index += 1) {
      const gap = readUvarintAt(geometryBytes, geometryCursor)
      stabilizerRow += gap.value
      stabilizers[stabilizerRow] = geometryView.getUint16(gap.offset, true)
      geometryCursor = gap.offset + 2
    }
    const bitmapStride = Math.ceil(patternCount / 32)
    const rankStride = Math.ceil(bitmapStride / RANK_WORDS)
    const presence = new Uint32Array(pairCount * bitmapStride)
    const RankArray = patternCount <= 0xffff ? Uint16Array : Uint32Array
    const ranks = new RankArray(pairCount * rankStride)
    const fractionStarts = new Uint32Array(pairCount)
    const fractionData = parseFractionData(view, fractionOffset, keyStreamOffset, patternCount, pairCount, cellCount)
    const keyStream = new Uint8Array(rawBuffer, keyStreamOffset, keyStreamSize)
    const keys = new Float64Array(patternCount)
    const randomRanges = { le4: [], "5": [], "6": [] }
    const randomCounts = { le4: 0, "5": 0, "6": 0 }
    let keyOffset = 0
    const keyBytes = new Uint8Array(KEY_STONES + 1)
    let bucket = null
    let rangeStart = 0
    const candidateBlock = new Uint32Array(Math.ceil(pairCount / 32) * 32)
    for (let idx = 0; idx < patternCount; idx += 1) {
      const prefixLength = keyStream[keyOffset++]
      const countByte = prefixLength ? keyBytes[0] : keyStream[keyOffset]
      const keyLength = 1 + (countByte & 15) + (countByte >> 4)
      assertPatternBundleSize(keyLength <= keyBytes.length)
      for (let i = prefixLength; i < keyLength; i += 1) { keyBytes[i] = keyStream[keyOffset++] }
      let key = countByte
      for (let i = 1; i < keyLength; i += 1) {
        key = key * KEY_BASE + keyBytes[i]
      }
      keys[idx] = key * KEY_BASE ** (KEY_STONES + 1 - keyLength)
      if (prefixLength === 0) {
        const moveCount = moveCountForStoneCounts(countByte & 15, countByte >> 4)
        const nextBucket = moveCount <= 4 ? "le4" : String(moveCount)
        if (bucket !== nextBucket) {
          if (bucket !== null) {
            randomRanges[bucket].push([rangeStart, idx - rangeStart])
            randomCounts[bucket] += idx - rangeStart
          }
          bucket = nextBucket
          rangeStart = idx
        }
      }
      const words = candidatesFor(keyBytes, keyLength, stabilizers[idx])
      const lane = idx & 31
      for (let i = 0; i < words.length; i += 1) { candidateBlock[(i << 5) + lane] = words[i] }
      if (lane === 31 || idx === patternCount - 1) {
        transposeCandidateBlock(candidateBlock, pairCount, bitmapStride, presence, idx >>> 5)
      }
    }
    if (bucket !== null) {
      randomRanges[bucket].push([rangeStart, patternCount - rangeStart])
      randomCounts[bucket] += patternCount - rangeStart
    }
    let fractionStart = 0
    for (let pair = 0; pair < pairCount; pair += 1) {
      fractionStarts[pair] = fractionStart
      let rank = 0
      for (let word = 0; word < bitmapStride; word += 1) {
        if (word % RANK_WORDS === 0) { ranks[pair * rankStride + word / RANK_WORDS] = rank }
        rank += popcount(presence[pair * bitmapStride + word])
      }
      fractionStart += rank
    }
    assertPatternBundleSize(fractionStart === cellCount)
    assertPatternBundleSize(keyOffset === keyStream.length)
    return {
      pattern_count: patternCount,
      corpus_count: corpusCount,
      view,
      keys,
      digits: new Uint8Array(KEY_STONES),
      randomRanges,
      randomCounts,
      pairTable,
      distanceBins,
      plusBins: new Uint32Array(distanceStride),
      minusBins: new Uint32Array(distanceStride),
      plusSecondBins: new Uint32Array(distanceStride),
      minusSecondBins: new Uint32Array(distanceStride),
      presence,
      ranks,
      bitmapStride,
      rankStride,
      fractionStarts,
      fractionData,
      cellCount,
      lastIndex: -1,
      lastEntry: null,
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
    const biasOffset = rowMeanResidualOffset + Math.ceil((patternCount * (rowMeanResidualMagnitudeBits + 1)) / 8)
    const secondBiasOffset = biasOffset + pairCount * DISTANCE_CONTEXTS
    const residualOffset = secondBiasOffset + DISTANCE_CONTEXTS * DISTANCE_CONTEXTS
    assertPatternBundleSize(residualOffset + Math.ceil(cellCount * PACKED_STONE_FRACTION_BITS / 8) === keyStreamOffset)
    return {
      rowMeanNumerator,
      rowMeanDenominator,
      rowMeanIntercept,
      rowMeanResidualMagnitudeBits,
      rowMeanResidualOffset,
      biasOffset,
      secondBiasOffset,
      residualOffset,
    }
  }

  function nearestBins(distances, digits, start, end, bins, secondBins) {
    if (start === end) {
      bins.fill(FARTHEST_BINS)
      secondBins.fill(FARTHEST_BINS)
      return
    }
    for (let word = 0; word < bins.length; word += 1) {
      let packed = distances[digits[start] * bins.length + word]
      let second = FARTHEST_BINS
      for (let i = start + 1; i < end; i += 1) {
        const candidate = distances[digits[i] * bins.length + word]
        // Differences stay within their lanes. Expand successful comparisons to nibble masks.
        const greater = (((packed | 0x88888888) - candidate) & 0x88888888) >>> 3
        const swap = (packed ^ candidate) & (greater * 15)
        packed ^= swap
        const farther = candidate ^ swap
        const secondGreater = (((second | 0x88888888) - farther) & 0x88888888) >>> 3
        second ^= (second ^ farther) & (secondGreater * 15)
      }
      bins[word] = packed
      secondBins[word] = second
    }
  }

  function patternEntryForLookupInData(data, pattern) {
    if (!data) { return null }
    const entryIndex = patternIndexForLookup(data.keys, pattern)
    if (entryIndex < 0) { return null }
    if (data.lastIndex === entryIndex) { return data.lastEntry }
    const view = data.view
    const key = data.keys[entryIndex]
    const count = Math.floor(key / KEY_COUNT_PLACE)
    const plusCount = count & 15
    const minusCount = count >> 4
    const stoneCount = plusCount + minusCount
    const digits = data.digits
    let place = KEY_COUNT_PLACE
    for (let i = 0; i < stoneCount; i += 1) {
      place /= KEY_BASE
      digits[i] = Math.floor(key / place) & (KEY_BASE - 1)
    }
    const tenuki = readBitplaneValue(view, HEADER_SIZE, data.pattern_count, entryIndex, PACKED_STONE_FRACTION_BITS)
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
        data.fractionData.rowMeanNumerator * tenuki,
        data.fractionData.rowMeanDenominator,
      ) +
      data.fractionData.rowMeanIntercept +
      rowMeanResidual
    assertPatternBundleSize(rowMean >= 0 && rowMean < PACKED_OPTIONAL_NULL)
    nearestBins(data.distanceBins, digits, 0, plusCount, data.plusBins, data.plusSecondBins)
    nearestBins(data.distanceBins, digits, plusCount, stoneCount, data.minusBins, data.minusSecondBins)
    const wordIndex = entryIndex >>> 5
    const wordBit = 1 << (entryIndex & 31)
    const rankIndex = Math.floor(wordIndex / RANK_WORDS)
    const rankWordIndex = rankIndex * RANK_WORDS
    for (let pairIndex = 0; pairIndex < data.pairTable.length; pairIndex += 1) {
      const offset = pairIndex * data.bitmapStride
      const word = data.presence[offset + wordIndex]
      if (!(word & wordBit)) { continue }
      let rank = data.ranks[pairIndex * data.rankStride + rankIndex]
      for (let i = rankWordIndex; i < wordIndex; i += 1) { rank += popcount(data.presence[offset + i]) }
      const fractionIndex = data.fractionStarts[pairIndex] + rank + popcount(word & (wordBit - 1))
      assertPatternBundleSize(fractionIndex >= 0 && fractionIndex < data.cellCount)
      const shift = (pairIndex & 7) * 4
      const binWord = pairIndex >>> 3
      const plusBin = (data.plusBins[binWord] >>> shift) & 15
      const minusBin = (data.minusBins[binWord] >>> shift) & 15
      const distanceContext = plusBin * DISTANCE_CLASSES + minusBin
      const plusSecond = (data.plusSecondBins[binWord] >>> shift) & 15
      const minusSecond = (data.minusSecondBins[binWord] >>> shift) & 15
      const secondContext = distanceContext * DISTANCE_CONTEXTS + plusSecond * DISTANCE_CLASSES + minusSecond
      const bias = view.getInt8(data.fractionData.biasOffset + pairIndex * DISTANCE_CONTEXTS + distanceContext)
      const correction = view.getInt8(data.fractionData.secondBiasOffset + secondContext)
      const prediction = rowMean + (bias + correction) * BIAS_SCALE
      const code = readBitplaneValue(view, data.fractionData.residualOffset, data.cellCount, fractionIndex, PACKED_STONE_FRACTION_BITS)
      const stoneFraction = decodeStoneFraction(code, prediction)
      assertPatternBundleSize(stoneFraction >= 0 && stoneFraction < PACKED_OPTIONAL_NULL)
      const pair = data.pairTable[pairIndex]
      cells.push([pair[0], pair[1], stoneFraction])
    }
    const entry = { p: minusCount <= plusCount ? "red" : "blue", c: cells }
    if (tenuki !== PACKED_OPTIONAL_NULL) { entry.t = tenuki }
    data.lastIndex = entryIndex
    data.lastEntry = entry
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
    moveCountForStoneCounts,
    applyTransformAx,
    inverseTransformId,
    formatPattern,
    formatHashPattern,
    canonicalizeLabeledPattern,
    shiftPoints,
    labeledFamilyDiff,
    inferBoardFromLabeledPattern,
    normalizeLoadedData,
    patternForIndex,
    patternEntryForLookupInData,
    randomPatternForBucket,
  }
})()
