export async function loadBytes(url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error("Could not load analysis. Reload to retry.")
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

export function copyToWasm(kernel, bytes) {
  const address = kernel.allocate(bytes.length)
  new Uint8Array(kernel.memory.buffer, address, bytes.length).set(bytes)
  return address
}

export function rexPanel(kernel, cells) {
  const board = kernel.allocate(cells)
  const moves = kernel.allocate(2 * cells)
  return position => {
    new Uint8Array(kernel.memory.buffer, board, cells).set(position)
    const wins = kernel.point(board)
    kernel.placements(board, moves)
    return { wins, placements: new Uint8Array(kernel.memory.buffer, moves, 2 * cells).slice() }
  }
}
