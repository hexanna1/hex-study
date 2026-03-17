// SPDX-License-Identifier: MIT
// Winograd transforms follow the KataGo OpenCL kernels. The low-error filter coefficients
// are adapted from transform3x3_6 in cpp/neuralnet/openclbackend.cpp. See LICENSE.
import { unpackSignedBytes } from "./compact_winograd.js"
import { kernel, READ, WRITE } from "./kernel.js"
import { WINOGRAD_OUTPUT_TILE } from "./winograd_kernels.js"

const INPUT_TILE = WINOGRAD_OUTPUT_TILE + 2
const ROOT_TWO = Math.SQRT2
const INNER_FILTER = [-2 / 3, -ROOT_TWO / 3, -1 / 3]
const OUTER_FILTER = [1 / 6, ROOT_TWO / 6, 1 / 3]

export function transformWinogradWeights(layer, scratch, write) {
  const count = layer.input * layer.output
  const stride = 3 * count
  const rowWeights = scratch("winogradRow", Float32Array, INPUT_TILE * count)
  const columns = scratch("winogradColumns", Float64Array, stride)
  const otherColumns = scratch("winogradOtherColumns", Float64Array, stride)
  const [innerFirst, innerMiddle, innerLast] = INNER_FILTER
  const [outerFirst, outerMiddle, outerLast] = OUTER_FILTER

  const writeRow = (row, input) => {
    for (let i = 0; i < count; i++) {
      const left = input[i], center = input[count + i], right = input[2 * count + i]
      const innerLeft = innerFirst * left, innerCenter = innerMiddle * center, innerRight = innerLast * right
      const outerLeft = outerFirst * left, outerCenter = outerMiddle * center, outerRight = outerLast * right
      rowWeights[i] = left
      rowWeights[count + i] = (innerLeft + innerCenter) + innerRight
      rowWeights[2 * count + i] = (innerLeft - innerCenter) + innerRight
      rowWeights[3 * count + i] = (outerLeft + outerCenter) + outerRight
      rowWeights[4 * count + i] = (outerLeft - outerCenter) + outerRight
      rowWeights[5 * count + i] = right
    }
    write(rowWeights, row * INPUT_TILE * count)
  }
  for (let i = 0; i < stride; i++) {
    columns[i] = layer.weights[i]
    otherColumns[i] = layer.weights[2 * stride + i]
  }
  writeRow(0, columns)
  writeRow(5, otherColumns)

  const writePair = (row, [first, middle, last]) => {
    for (let i = 0; i < stride; i++) {
      const left = first * layer.weights[i]
      const center = middle * layer.weights[stride + i]
      const right = last * layer.weights[2 * stride + i]
      columns[i] = (left + center) + right
      otherColumns[i] = (left - center) + right
    }
    writeRow(row, columns)
    writeRow(row + 1, otherColumns)
  }
  writePair(1, INNER_FILTER)
  writePair(3, OUTER_FILTER)
}

function filterTransform() {
  const lines = []
  const axis = (name, [a, b, c]) => {
    lines.push(`
    let ${name}Even = (-2.0/3.0) * ${a} + (-1.0/3.0) * ${c};
    let ${name}Odd = (-1.4142135623730951/3.0) * ${b};
    let ${name}Outer = (1.0/6.0) * ${a} + (1.0/3.0) * ${c};
    let ${name}Side = (1.4142135623730951/6.0) * ${b};`)
    return [a, `${name}Even + ${name}Odd`, `${name}Even - ${name}Odd`,
      `${name}Outer + ${name}Side`, `${name}Outer - ${name}Side`, c]
  }
  for (let tap = 0; tap < 9; tap++) {
    const word = `source[${tap}u * params.stride + index]`
    lines.push(`let tap${tap} = vec4<f32>(${unpackSignedBytes(word)}) * gain;`)
  }
  const columns = Array.from({ length: 3 }, (_, x) =>
    axis("col" + x, [`tap${x}`, `tap${x + 3}`, `tap${x + 6}`]))
  for (let y = 0; y < 6; y++) {
    const row = axis("row" + y, columns.map((column) => `(${column[y]})`))
    row.forEach((value, x) => lines.push(`store(${y * 6 + x}u, index, ${value});`))
  }
  return lines.join("\n")
}

export function createWinogradFilter(half) {
  const vector = half ? "vec4<f16>" : "vec4<f32>"
  return kernel(`
${half ? "enable f16;" : ""}
struct Params { stride: u32, outputVectors: u32 }
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> weights: array<${vector}>;
@group(0) @binding(2) var<uniform> params: Params;
fn store(plane: u32, index: u32, value: vec4<f32>) {
  weights[plane * params.stride + index] = ${vector}(value);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.stride) { return; }
  let c = 9u * params.stride + (index % params.outputVectors) * 4u;
  // Persistent FP32-path filters store the scale for the integer F(2,3) numerator.
  let gain = ${half ? "" : "4.0 * "}bitcast<vec4<f32>>(vec4<u32>(source[c], source[c + 1u], source[c + 2u], source[c + 3u]));
  ${filterTransform()}
}`, [READ, WRITE])
}
