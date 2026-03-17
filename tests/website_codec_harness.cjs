const fs = require("node:fs")
const path = require("node:path")

// Load the actual codec functions without starting page controllers or fetching data.
function loadCodec(directory, name) {
  const window = { HexMoveTree: {} }
  if (name === "openings") require("../docs/position.js")
  const document = { documentElement: { dataset: {
    openingBoardSizes: "11,12,13,14,17", josekiFamilies: "A:A,O:O",
  } } }
  Function("window", fs.readFileSync(path.join(directory, "study_ui.js"), "utf8"))(window)
  const source = fs.readFileSync(path.join(directory, `${name}.js`), "utf8")
  const prefix = source.slice(0, source.indexOf("const elements ="))
  const functions = [...source.matchAll(/^function \w+\([\s\S]*?^\}/gm)].map((match) => match[0])
  const extras = name === "patterns"
    ? source.slice(source.indexOf("const PATTERN_TRANSFORMS"), source.indexOf("function applyTransformAx"))
      + source.slice(source.indexOf("const PATTERN_RANK_INTERVAL"), source.indexOf("function createGeometryCandidates"))
    : ""
  return Function("window", "document", prefix + extras + functions.join("\n") + `
    return {
      normalizeLoadedData,
      ${name === "patterns" ? "patternEntryForLookupInData, applyTransformAx" : ""}
      ${name === "openings" ? "openingChildIndices, decodeOpeningNode" : ""}
      ${name === "joseki" ? "josekiChildIndices, decodeJosekiNode, ensureJosekiRandomIndex, josekiLineForRandomRank" : ""}
    }
  `)(window, document)
}

function arrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

module.exports = { loadCodec, arrayBuffer }
