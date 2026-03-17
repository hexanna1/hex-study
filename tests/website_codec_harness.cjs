const fs = require("node:fs")
const path = require("node:path")

// Load the same codec scripts as the pages, without starting page controllers.
function loadCodec(directory, name) {
  const window = {}
  if (name === "openings") require("../docs/position.js")
  for (const file of ["study_ui.js", `${name}_codec.js`]) {
    Function("window", fs.readFileSync(path.join(directory, file), "utf8"))(window)
  }
  if (name === "joseki") return window.createJosekiCodec(["A", "O"])
  return window[name === "patterns" ? "HexPatternCodec" : "HexOpeningCodec"]
}

function arrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

module.exports = { loadCodec, arrayBuffer }
