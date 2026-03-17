import { copyToWasm, loadBytes, rexPanel } from "./rex_data.js"

async function loadOracle() {
  const binary = await loadBytes(new URL("./data/rex/6x6.wasm.gz", import.meta.url))
  const { instance, module } = await WebAssembly.instantiate(binary)
  const kernel = instance.exports
  const section = name => copyToWasm(kernel, new Uint8Array(WebAssembly.Module.customSections(module, name)[0]))
  kernel.skip_dominated_moves(1)
  kernel.install(section("tables"), 0, 0, 2, section("classifier"))
  kernel.apply_subsets(section("bounds"))
  kernel.build_band(2, 0)
  kernel.build_ranked()
  return rexPanel(kernel, 36)
}

const ready = loadOracle()
ready.catch(error => self.postMessage({ error: error.message }))
self.onmessage = async ({ data }) => {
  try {
    const analyze = await ready
    const panel = analyze(data.board)
    self.postMessage({ id: data.id, panel }, [panel.placements.buffer])
  } catch (error) { self.postMessage({ error: error.message }) }
}
