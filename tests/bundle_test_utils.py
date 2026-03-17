def read_little_endian_bits(data: bytes, *, offset: int, bit_offset: int, bits: int, chunk_bytes: int) -> int:
    if bits == 0:
        return 0
    byte_offset = int(offset) + (int(bit_offset) // 8)
    shift = int(bit_offset) % 8
    chunk = 0
    for idx in range(int(chunk_bytes)):
        pos = byte_offset + idx
        if pos < len(data):
            chunk |= data[pos] << (8 * idx)
    return (chunk >> shift) & ((1 << int(bits)) - 1)
