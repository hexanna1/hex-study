export const READ = "read-only-storage"
export const WRITE = "storage"

const LAYER = [READ, READ, WRITE]

export function kernel(code, buffers = LAYER) {
  return { code, buffers }
}
