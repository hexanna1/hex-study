// SPDX-License-Identifier: MIT
// Search weighting and exploration adapted from KataHex. See THIRD_PARTY_NOTICES.md.

const VALUE_CDF = Float64Array.from({ length: 2000 }, (_, i) => {
  if (i === 0) return 0
  if (i === 1999) return 1
  const z = -50 + i * 100 / 1999
  return 0.5 + (Math.atan(z / Math.sqrt(3)) + Math.sqrt(3) * z / (z * z + 3)) / Math.PI
})

function valueCdf(z) {
  const index = Math.max(0, Math.min(1999, 1999 * (z + 50) / 100))
  const lower = Math.min(1998, Math.floor(index))
  return VALUE_CDF[lower] + (index - lower) * (VALUE_CDF[lower + 1] - VALUE_CDF[lower])
}

export function childWeight(child, visits) {
  return child.weight * (visits / Math.max(1, child.visits))
}

export function recompute(current) {
  if (current.terminal !== null) {
    current.value = current.terminal
    current.valueSq = current.terminal ** 2
    current.weight = current.visits
    current.weightSq = current.visits
    return
  }
  const children = current.children
  const rows = []
  if (children) {
    // Internal children form a prior-sorted prefix; refreshed roots keep insertion order.
    const count = children.order?.length ?? children.nodes.length
    for (let j = 0; j < count; j++) {
      const i = children.order ? children.order[j] : j
      const child = children.nodes[i], visits = children.visits[i]
      if (visits && child.visits) rows.push({ child, prior: children.priors[i], weight: childWeight(child, visits), value: -child.value })
    }
  }
  let total = 0, utilitySum = 0, policySum = 0
  for (const row of rows) {
    const prior = Math.max(1e-30, row.prior)
    if (total > 0 && policySum > 0) {
      const gap = utilitySum / total - row.value
      const share = 2 * total * prior / policySum
      if (gap > 0 && row.weight > share) row.weight -= (row.weight - share) * (1 - Math.exp(-gap / 0.15))
    }
    total += row.weight
    utilitySum += row.weight * row.value
    policySum += prior
  }
  let adjusted = 0
  for (const row of rows) {
    const stdev = Math.sqrt(1e-8 + 1 / (1.5 * Math.sqrt(row.weight)))
    row.weight *= (valueCdf((row.value - utilitySum / total) / stdev) + 0.0001) ** 0.25
    adjusted += row.weight
  }
  let value = current.initialValue, valueSq = value * value, weightSq = 1
  for (const row of rows) {
    const weight = row.weight * total / adjusted
    value += weight * row.value
    valueSq += weight * row.child.valueSq
    weightSq += (weight / row.child.weight) ** 2 * row.child.weightSq
  }
  current.weightSq = weightSq
  current.weight = 1 + total
  current.value = value / current.weight
  current.valueSq = valueSq / current.weight
}

export function explorationScale(parent, childWeight) {
  const meanSq = parent.value ** 2
  const variance = parent.weight <= 1 ? 0.16 : Math.max(0,
    ((meanSq + 0.16) * 2 + Math.max(parent.valueSq, meanSq) * parent.weight) / (2 + parent.weight - 1) - meanSq)
  return (0.9 + 0.6 * Math.log((childWeight + 500) / 500)) * Math.sqrt(childWeight + 0.01)
    * (1 + 0.85 * (Math.sqrt(variance) / 0.4 - 1))
}

export function selectionWeights(parent, isRoot, useLcb) {
  const { nodes, priors, visits } = parent.children
  const count = Math.min(nodes.length, priors.length)
  const weights = new Float64Array(priors.length)
  for (let i = 0; i < count; i++) weights[i] = nodes[i] ? childWeight(nodes[i], visits[i]) : 0
  let best = -1, total = 0
  for (let i = 0; i < count; i++) {
    total += weights[i]
    if (weights[i] > 0 && (best < 0 || weights[i] > weights[best])) best = i
  }
  if (best < 0) return weights
  const mostWeight = weights[best]
  if (isRoot) {
    const exploration = explorationScale(parent, total)
    const bestScore = -nodes[best].value + exploration * priors[best] / (1 + weights[best])
    for (let i = 0; i < count; i++) {
      if (i === best || !weights[i]) continue
      const gap = bestScore + nodes[i].value
      const desired = gap <= 0 ? Infinity : Math.max(0, exploration * priors[i] / gap - 1)
      weights[i] = Math.ceil(Math.min(weights[i], desired))
    }
  }
  if (useLcb) {
    const bounds = nodes.map((child, i) => {
      if (!visits[i] || !child.weight) return { lcb: -10, radius: 10 }
      const fraction = visits[i] / Math.max(1, child.visits)
      let weight = child.weight * fraction, weightSq = child.weightSq * fraction
      let ess = weight * weight / weightSq
      const priorWeight = weight / ess ** 3
      let square = Math.max(child.valueSq, child.value ** 2 + 1e-8)
      square = (square * weight + (square + 1) * priorWeight) / (weight + priorWeight)
      weight += priorWeight
      weightSq += priorWeight ** 2
      ess = weight * weight / weightSq
      const radius = 5 * Math.sqrt((square - child.value ** 2) / ess)
      return { lcb: -child.value - radius, radius }
    })
    let bestLcb = -1
    for (let i = 0; i < count; i++) {
      if (weights[i] > 0 && weights[i] >= 0.15 * mostWeight
        && (bestLcb < 0 || bounds[i].lcb > bounds[bestLcb].lcb)) bestLcb = i
    }
    if (bestLcb >= 0) {
      let adjusted = weights[bestLcb]
      for (let i = 0; i < count; i++) {
        if (i === bestLcb || !bounds[i]) continue
        const excess = bounds[bestLcb].lcb - bounds[i].lcb
        if (excess < 0) continue
        const radius = bounds[i].radius
        const factor = (radius + excess) / (radius + 0.2 * excess)
        adjusted = Math.max(adjusted, factor * factor * weights[i])
      }
      weights[bestLcb] = adjusted
    }
  }
  const maximum = Math.max(...weights)
  const prune = Math.min(1, maximum / 64)
  for (let i = 0; i < count; i++) {
    weights[i] = weights[i] < prune ? 0 : weights[i] * Math.max(1, 1 / maximum)
  }
  return weights
}
