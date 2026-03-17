(() => {
  const { lerpRgb, rgbText, THEME } = window.HexStudyUI

  for (const figure of document.querySelectorAll(".result-board")) {
    const table = figure.querySelector("table")
    const rows = [...table.tBodies[0].rows]
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
    svg.classList.add("result-hex-board")
    svg.setAttribute("aria-hidden", "true")
    const board = window.HexMoveTree.createBoardSvg(svg)
    const columns = rows[0].cells.length - 1
    board.setupViewBox(rows.length)
    const squareWidth = Number(svg.getAttribute("viewBox").split(" ")[2])
    board.setupViewBox(columns, null, rows.length)
    const widthRatio = Number(svg.getAttribute("viewBox").split(" ")[2]) / squareWidth
    svg.style.width = `${100 * widthRatio}%`
    svg.style.maxWidth = `${380 * widthRatio}px`

    for (let row = 0; row < rows.length; row++) {
      for (let col = 0; col < columns; col++) {
        const value = rows[row].cells[col + 1].textContent.trim()
        const percent = Number(value)
        const fill = rgbText(lerpRgb(THEME.OFF_WHITE_RGB, THEME.CANDIDATE_HIGH, percent / 100))
        rows[row].cells[col + 1].style.backgroundColor = fill
        const cell = board.appendHex(col + 1, row + 1, { fill, title: `${String.fromCharCode(97 + col)}${row + 1}` })
        board.appendText(cell.cx, cell.cy, percent.toFixed(1))
      }
    }

    board.renderFrame(columns, rows.length)

    const details = document.createElement("details")
    const summary = document.createElement("summary")
    summary.textContent = "Values to four decimal places"
    details.append(summary, table)
    figure.append(svg, details)
  }
})()
