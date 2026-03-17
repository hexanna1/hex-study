// Shared playable-cell geometry in zero-indexed tensor coordinates.
(() => {
  function isPlayable(game, x, y, size) {
    if (x < 0 || y < 0 || x >= size || y >= size) return false
    if (game === "y") return x + y < size
    if (game === "yavalath") {
      const radius = (size - 1) / 2
      return x + y >= radius && x + y <= 3 * radius
    }
    return true
  }
  function playablePoints(game, size) {
    const points = []
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (isPlayable(game, x, y, size)) points.push(y * size + x)
      }
    }
    return points
  }
  globalThis.BoardGeometry = { isPlayable, playablePoints }
})()
