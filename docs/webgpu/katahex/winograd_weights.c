// SPDX-License-Identifier: MIT
// Adapted from winograd_kernels.js. See LICENSE for the upstream notice.

#include <stdint.h>
#include <wasm_simd128.h>

static inline v128_t dequantize(const int8_t *codes, v128_t scales) {
  const v128_t bytes = wasm_v128_load32_zero(codes);
  const v128_t values = wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(bytes));
  return wasm_f32x4_mul(wasm_f32x4_convert_i32x4(values), scales);
}

static inline void transformAxis(v128_t left, v128_t center, v128_t right, v128_t output[6]) {
  // Exact f64 filter constants from the JS transform. Keep the explicit zero
  // terms and addition order to match its signed-zero and rounding behavior.
  const v128_t zero = wasm_f64x2_splat(0);
  const v128_t innerLeft = wasm_f64x2_mul(wasm_f64x2_splat(-0x1.5555555555555p-1), left);
  const v128_t innerCenter = wasm_f64x2_mul(wasm_f64x2_splat(-0x1.e2b7dddfefa67p-2), center);
  const v128_t innerRight = wasm_f64x2_mul(wasm_f64x2_splat(-0x1.5555555555555p-2), right);
  const v128_t outerLeft = wasm_f64x2_mul(wasm_f64x2_splat(0x1.5555555555555p-3), left);
  const v128_t outerCenter = wasm_f64x2_mul(wasm_f64x2_splat(0x1.e2b7dddfefa67p-3), center);
  const v128_t outerRight = wasm_f64x2_mul(wasm_f64x2_splat(0x1.5555555555555p-2), right);
  output[0] = wasm_f64x2_add(wasm_f64x2_add(left, wasm_f64x2_mul(zero, center)), wasm_f64x2_mul(zero, right));
  output[1] = wasm_f64x2_add(wasm_f64x2_add(innerLeft, innerCenter), innerRight);
  output[2] = wasm_f64x2_add(wasm_f64x2_sub(innerLeft, innerCenter), innerRight);
  output[3] = wasm_f64x2_add(wasm_f64x2_add(outerLeft, outerCenter), outerRight);
  output[4] = wasm_f64x2_add(wasm_f64x2_sub(outerLeft, outerCenter), outerRight);
  output[5] = wasm_f64x2_add(wasm_f64x2_add(wasm_f64x2_mul(zero, left), wasm_f64x2_mul(zero, center)), right);
}

static inline v128_t halfBits(v128_t values) {
  const v128_t magnitude = wasm_v128_and(values, wasm_i32x4_splat(0x7fffffff));
  const v128_t sign = wasm_v128_and(wasm_u32x4_shr(values, 16), wasm_i32x4_splat(0x8000));
  // Round to nearest even. Subnormal half values are multiples of 2^-24.
  const v128_t parity = wasm_v128_and(wasm_u32x4_shr(magnitude, 13), wasm_i32x4_splat(1));
  const v128_t rounded = wasm_i32x4_add(magnitude, wasm_i32x4_add(wasm_i32x4_splat(0xfff), parity));
  const v128_t normal = wasm_i32x4_sub(wasm_u32x4_shr(rounded, 13), wasm_i32x4_splat(0x1c000));
  const v128_t subnormal = wasm_u32x4_trunc_sat_f32x4(
    wasm_f32x4_nearest(wasm_f32x4_mul(magnitude, wasm_f32x4_splat(0x1p24f))));
  v128_t half = wasm_v128_bitselect(normal, subnormal, wasm_i32x4_ge(magnitude, wasm_i32x4_splat(0x38800000)));
  half = wasm_v128_bitselect(half, wasm_i32x4_splat(0x7c00), wasm_i32x4_lt(magnitude, wasm_i32x4_splat(0x47800000)));
  half = wasm_v128_bitselect(wasm_i32x4_splat(0x7e00), half, wasm_i32x4_gt(magnitude, wasm_i32x4_splat(0x7f800000)));
  return wasm_v128_or(half, sign);
}

static inline v128_t promoteHigh(v128_t values) {
  return wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(values, values, 2, 3, 0, 1));
}

// Both arrays are coefficient-major, with output channels contiguous in each plane.
// The caller selects layers whose input and output channel counts are multiples of four.
void transform(uint16_t *output, const int8_t *codes, const float *scales, unsigned planeSize, unsigned channels) {
  unsigned channel = 0;
  for (unsigned i = 0; i < planeSize; i += 4) {
    const v128_t scale = wasm_v128_load(scales + channel);
    v128_t low[3][6], high[3][6];
    #pragma clang loop unroll(full)
    for (unsigned column = 0; column < 3; column++) {
      const v128_t top = dequantize(codes + i + column * planeSize, scale);
      const v128_t middle = dequantize(codes + i + (column + 3) * planeSize, scale);
      const v128_t bottom = dequantize(codes + i + (column + 6) * planeSize, scale);
      transformAxis(wasm_f64x2_promote_low_f32x4(top), wasm_f64x2_promote_low_f32x4(middle),
                    wasm_f64x2_promote_low_f32x4(bottom), low[column]);
      transformAxis(promoteHigh(top), promoteHigh(middle), promoteHigh(bottom), high[column]);
    }
    #pragma clang loop unroll(full)
    for (unsigned row = 0; row < 6; row++) {
      v128_t lowRow[6], highRow[6];
      transformAxis(low[0][row], low[1][row], low[2][row], lowRow);
      transformAxis(high[0][row], high[1][row], high[2][row], highRow);
      #pragma clang loop unroll(full)
      for (unsigned column = 0; column < 6; column++) {
        const v128_t lowF32 = wasm_f32x4_demote_f64x2_zero(lowRow[column]);
        const v128_t highF32 = wasm_f32x4_demote_f64x2_zero(highRow[column]);
        const v128_t values = wasm_i32x4_shuffle(lowF32, highF32, 0, 1, 4, 5);
        const v128_t bits = halfBits(values);
        wasm_v128_store64_lane(output + (row * 6 + column) * planeSize + i, wasm_u16x8_narrow_i32x4(bits, bits), 0);
      }
    }
    channel += 4;
    if (channel == channels) channel = 0;
  }
}
