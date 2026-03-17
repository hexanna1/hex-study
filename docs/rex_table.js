(() => {
  async function decode(data) {
    let cursor = 0

    function varint() {
      let value = 0
      let shift = 0
      let byte
      do {
        byte = data[cursor++]
        value |= (byte & 127) << shift
        shift += 7
      } while (byte & 128)
      return value >>> 0
    }

    const levelCount = varint()
    const root = varint()
    const count = varint()
    const size = Math.sqrt(levelCount)
    // Edges use 26 bits: the low bit complements the result, and 0/1 are terminals.
    // The high six bits of columns 0/1 hold the source/transposed cell respectively.
    const children = new Uint32Array(count * 3)
    // Copy to avoid retaining the inflated payload.
    const order = data.slice(cursor, cursor + levelCount)
    cursor += levelCount
    const choices = data.subarray(cursor, cursor + levelCount)
    cursor += levelCount
    let bit = 0
    let next = 1
    let sliceStarted = performance.now()

    async function pause() {
      await new Promise((resolve) => setTimeout(resolve, 0))
      sliceStarted = performance.now()
    }

    function peek(count) {
      const p = bit >>> 3
      const offset = bit & 7
      const word = (data[p] << 24) | (data[p + 1] << 16) | (data[p + 2] << 8) | data[p + 3]
      return (word >>> (32 - offset - count)) & ((1 << count) - 1)
    }

    function read(count) {
      if (count > 24) return read(16) * 2 ** (count - 16) + read(count - 16)
      const result = peek(count)
      bit += count
      return result
    }

    function table() {
      const sizes = data.subarray(cursor, cursor + 88)
      cursor += 88
      const counts = new Uint32Array(32)
      const first = new Uint32Array(32)
      const base = new Uint32Array(32)
      for (const size of sizes) {
        if (size) { counts[size]++ }
      }
      let code = 0
      let position = 0
      for (let size = 1; size < 32; size++) {
        code = (code + counts[size - 1]) << 1
        first[size] = code
        base[size] = position
        position += counts[size]
      }
      const symbols = new Uint8Array(position)
      const offsets = base.slice()
      const fast = new Uint32Array(1024)
      for (let symbol = 0; symbol < 88; symbol++) {
        const size = sizes[symbol]
        if (!size) { continue }
        const index = offsets[size]++
        symbols[index] = symbol
        if (size <= 10) {
          const start = (first[size] + index - base[size]) << (10 - size)
          fast.fill((symbol << 5) | size, start, start + (1 << (10 - size)))
        }
      }
      return { counts, first, base, symbols, fast }
    }

    function integer(book) {
      const entry = book.fast[peek(10)]
      let symbol
      if (entry) {
        bit += entry & 31
        symbol = entry >>> 5
      } else {
        let code = 0
        let size = 0
        do {
          code = (code << 1) | read(1)
          size++
        } while (code - book.first[size] >= book.counts[size])
        symbol = book.symbols[book.base[size] + code - book.first[size]]
      }
      if (symbol < 64) { return symbol }
      const remainder = symbol - 64 + 6
      return (1 << remainder) | read(remainder)
    }

    async function decodePairs() {
      const count = varint()
      const books = Array.from({ length: 3 }, table)
      const rows = new Uint32Array(2 * count)
      bit = cursor * 8
      let previous = 0
      for (let i = 0; i < count; i++) {
        rows[2 * i + 1] = previous += integer(books[0])
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      previous = 0
      for (let i = 0; i < count; i++) {
        const same = rows[2 * i + 1] === (i ? rows[2 * i - 1] : 0)
        const x = integer(books[same ? 1 : 2])
        rows[2 * i] = previous += same ? x : (x >>> 1) ^ -(x & 1)
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      cursor = (bit + 7) >>> 3
      return rows
    }

    async function decodeTriples(permutation) {
      const count = varint()
      const books = Array.from({ length: 5 }, table)
      bit = cursor * 8
      let previous = 0
      for (let i = 0; i < count; i++) {
        const p = 3 * (next + i)
        previous += integer(books[0])
        children[p + 2] = previous
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      previous = 0
      for (let i = 0; i < count; i++) {
        const p = 3 * (next + i)
        const same = children[p + 2] === (i ? children[p - 1] : 0)
        const x = integer(books[same ? 1 : 2])
        previous += same ? x : (x >>> 1) ^ -(x & 1)
        children[p] = previous * 2
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      previous = 0
      for (let i = 0; i < count; i++) {
        const p = 3 * (next + i)
        const same = children[p + 2] === (i ? children[p - 1] : 0) && children[p] === (i ? children[p - 3] : 0)
        const x = integer(books[same ? 3 : 4])
        previous += same ? x : (x >>> 1) ^ -(x & 1)
        children[p + 1] = previous
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      for (let i = 0; i < count; i++) {
        const p = 3 * (next + i)
        const a = children[p]
        const b = children[p + 1]
        const c = children[p + 2]
        children[p + permutation[0]] = a
        children[p + permutation[1]] = b
        children[p + permutation[2]] = c
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      cursor = (bit + 7) >>> 3
      return count
    }

    for (let level = levelCount - 1; level >= 0; level--) {
      const choice = choices[level]
      const dictionary = choice < 12
      const other = dictionary ? choice >> 2 : (choice - 12) >> 1
      const pair = [0, 1, 2].filter((digit) => digit !== other)
      if (dictionary ? choice & 2 : choice & 1) { pair.reverse() }
      const start = next
      if (dictionary) {
        const pairs = await decodePairs()
        const rows = await decodePairs()
        const reverse = choice & 1
        for (let i = 0; i < rows.length / 2; i++) {
          const p = 3 * next++
          const id = 2 * rows[2 * i + reverse]
          children[p + pair[0]] = pairs[id] * 2
          children[p + pair[1]] = pairs[id + 1]
          children[p + other] = rows[2 * i + (reverse ^ 1)]
          if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
        }
      }
      next += await decodeTriples([...pair, other])
      const cell = order[level]
      const dual = (cell % size) * size + Math.floor(cell / size)
      for (let i = start; i < next; i++) {
        children[3 * i] |= cell << 26
        children[3 * i + 1] |= dual << 26
        if ((i & 8191) === 8191 && performance.now() - sliceStarted >= 8) await pause()
      }
      if (performance.now() - sliceStarted >= 8) await pause()
    }
    return { size, root, children, order }
  }

  function makeLookup({ size, root, children, order }, { prefix = 0 } = {}) {
    const heads = new Uint32Array(prefix ? 3 ** prefix : 0)
    if (prefix) {
      heads[0] = root
      for (let level = 0, count = 1; level < prefix; level++, count *= 3) {
        const cell = order[level]
        for (let key = count - 1; key >= 0; key--) {
          const value = heads[key]
          const p = 3 * (value >>> 1)
          const target = 3 * key
          if (value >= 2 && children[p] >>> 26 === cell) {
            const phase = value & 1
            heads[target] = (children[p] & 0x03ffffff) ^ phase
            heads[target + 1] = (children[p + 1] & 0x03ffffff) ^ phase
            heads[target + 2] = (children[p + 2] & 0x03ffffff) ^ phase
          } else { heads[target] = heads[target + 1] = heads[target + 2] = value }
        }
      }
    }
    const dualOrder = Array.from(order, (cell) => (cell % size) * size + Math.floor(cell / size))
    function direct(board) {
      let value = root
      if (prefix) {
        let key = 0
        for (let i = 0; i < prefix; i++) key = 3 * key + board[order[i]]
        value = heads[key]
      }
      while (value >= 2) {
        const p = 3 * (value >>> 1)
        const digit = board[children[p] >>> 26]
        value = (children[p + digit] & 0x03ffffff) ^ (value & 1)
      }
      return value
    }
    // The two-bit digit map 24 encodes [0, 2, 1] for color exchange.
    function dual(board) {
      let value = root
      if (prefix) {
        let key = 0
        for (let i = 0; i < prefix; i++) key = 3 * key + ((24 >>> (2 * board[dualOrder[i]])) & 3)
        value = heads[key]
      }
      while (value >= 2) {
        const p = 3 * (value >>> 1)
        const digit = board[children[p + 1] >>> 26]
        value = (children[p + ((24 >>> (2 * digit)) & 3)] & 0x03ffffff) ^ (value & 1)
      }
      return value
    }
    const lookup = (board, transposeColors) => (transposeColors ? dual(board) : direct(board))
    lookup.dispatch = { heads, order, dualOrder, prefix }
    return lookup
  }

  window.HexRexTable = { decode, makeLookup }
})()
