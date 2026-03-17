// SPDX-License-Identifier: MIT

// Views remain valid until the next use of the same workspace.
export function tensorScratch() {
  const arrays = new Map()
  return (name, Type, length) => {
    let array = arrays.get(name)
    if (!array || array.length < length) {
      array = new Type(length)
      arrays.set(name, array)
    }
    return array.subarray(0, length)
  }
}
