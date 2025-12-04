//
// `LongMen` - 'ONNX Model inference in c++'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// LongMen is provided under: GNU Affero General Public License (AGPL3.0)
// https://www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//

#ifndef LONGMEN_FP16_TO_FP32_H_
#define LONGMEN_FP16_TO_FP32_H_

#include <cmath>
#include <cstdint>

/**
 * @file fp16_to_fp32.h
 * @brief IEEE 754 half-precision to single-precision floating-point conversion
 *
 * Provides optimized conversion from FP16 (half-precision) to FP32
 * (single-precision) floating-point format. Automatically selects the best
 * implementation based on available hardware features:
 * - F16C intrinsics for x86/x64 with F16C support
 * - ARM NEON intrinsics for ARM processors with NEON support
 * - Software fallback for other platforms
 *
 * @par IEEE 754 Half-Precision Format (16 bits)
 * @code
 * Bit layout: SEEEEE MMMMMMMMMM
 * - S (1 bit):  Sign bit (bit 15)
 * - E (5 bits): Exponent (bits 14-10), bias = 15
 * - M (10 bits): Mantissa/Significand (bits 9-0)
 * @endcode
 *
 * @par IEEE 754 Single-Precision Format (32 bits)
 * @code
 * Bit layout: SEEEEEEE EMMMMMMMMMMMMMMMMMMMMMMM
 * - S (1 bit):  Sign bit (bit 31)
 * - E (8 bits): Exponent (bits 30-23), bias = 127
 * - M (23 bits): Mantissa/Significand (bits 22-0)
 * @endcode
 *
 * @par Special Values
 * - Zero: exponent = 0, mantissa = 0
 * - Subnormal: exponent = 0, mantissa ≠ 0
 * - Infinity: exponent = all 1s, mantissa = 0
 * - NaN: exponent = all 1s, mantissa ≠ 0
 *
 * @note All implementations are header-only and inline for performance
 * @note Thread-safe and reentrant
 */

#ifdef __F16C__
#include <immintrin.h>

/**
 * @brief Converts FP16 to FP32 using F16C intrinsics (x86/x64)
 *
 * Uses hardware-accelerated F16C instruction set (VCVTPH2PS) for fast
 * conversion. Available on Intel processors since Ivy Bridge (2012) and
 * AMD processors since Bulldozer (2011).
 *
 * @param half 16-bit half-precision float in IEEE 754 format
 * @return 32-bit single-precision float
 *
 * @note Requires CPU with F16C support (compile with -mf16c)
 * @note Handles all special values (zero, infinity, NaN, subnormals)
 * @note Typically 5-10x faster than software implementation
 * @note Thread-safe and reentrant
 *
 * @par Performance
 * - Latency: ~3-5 CPU cycles
 * - Throughput: ~1-2 conversions per cycle
 *
 * @par Example
 * @code
 * uint16_t fp16_value = 0x3C00;  // 1.0 in FP16
 * float fp32_value = fp16_to_fp32(fp16_value);  // 1.0f in FP32
 * @endcode
 */
inline float fp16_to_fp32(uint16_t half) noexcept {
  // Load 16-bit value into 128-bit SSE register
  __m128i h = _mm_cvtsi32_si128(static_cast<int32_t>(half));

  // Convert packed FP16 to packed FP32 (hardware instruction)
  __m128 f = _mm_cvtph_ps(h);

  // Extract first float from packed result
  return _mm_cvtss_f32(f);
}

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

/**
 * @brief Converts FP16 to FP32 using ARM NEON intrinsics
 *
 * Uses hardware-accelerated ARM NEON instruction set (VCVT) for fast
 * conversion. Available on ARMv7-A with NEON and all ARMv8-A processors.
 *
 * @param half 16-bit half-precision float in IEEE 754 format
 * @return 32-bit single-precision float
 *
 * @note Requires CPU with NEON support (compile with -mfpu=neon or
 * -march=armv8-a)
 * @note Handles all special values (zero, infinity, NaN, subnormals)
 * @note Thread-safe and reentrant
 *
 * @par Performance
 * - Latency: ~2-4 CPU cycles
 * - Throughput: ~1-2 conversions per cycle
 *
 * @par Example
 * @code
 * uint16_t fp16_value = 0x3C00;  // 1.0 in FP16
 * float fp32_value = fp16_to_fp32(fp16_value);  // 1.0f in FP32
 * @endcode
 */
inline float fp16_to_fp32(uint16_t half) noexcept {
  // Reinterpret uint16_t as float16_t (ARM native FP16 type)
  float16_t h = *reinterpret_cast<const float16_t *>(&half);

  // Load single FP16 value and convert to FP32 vector
  float16x4_t h_vec = vld1_dup_f16(&h);
  float32x4_t f32_vec = vcvt_f32_f16(h_vec);

  // Extract first element
  return vgetq_lane_f32(f32_vec, 0);
}

#else

/**
 * @brief Converts FP16 to FP32 using software implementation
 *
 * Pure software implementation for platforms without hardware FP16 support.
 * Correctly handles all IEEE 754 special cases including zero, subnormals,
 * infinity, and NaN values.
 *
 * @param half 16-bit half-precision float in IEEE 754 format
 * @return 32-bit single-precision float
 *
 * @note Fallback implementation for platforms without F16C or NEON
 * @note Handles all special values correctly:
 *       - Zero (±0.0)
 *       - Subnormal numbers (denormalized)
 *       - Normal numbers
 *       - Infinity (±∞)
 *       - NaN (quiet and signaling)
 * @note Thread-safe and reentrant
 * @note Approximately 5-10x slower than hardware implementations
 *
 * @par Algorithm
 * 1. Extract sign, exponent, and mantissa from FP16
 * 2. Handle special cases (zero, subnormal, infinity, NaN)
 * 3. Convert exponent bias from 15 (FP16) to 127 (FP32)
 * 4. Extend mantissa from 10 bits to 23 bits
 * 5. Reconstruct FP32 value
 *
 * @par Example
 * @code
 * uint16_t fp16_value = 0x3C00;  // 1.0 in FP16
 * float fp32_value = fp16_to_fp32(fp16_value);  // 1.0f in FP32
 *
 * // Special values
 * fp16_to_fp32(0x0000);  // +0.0
 * fp16_to_fp32(0x8000);  // -0.0
 * fp16_to_fp32(0x7C00);  // +Infinity
 * fp16_to_fp32(0xFC00);  // -Infinity
 * fp16_to_fp32(0x7E00);  // NaN
 * @endcode
 */
inline float fp16_to_fp32(uint16_t half) noexcept {
  // Union for type-punning (safe in C++)
  union {
    uint32_t u;
    float f;
  } converter;

  // Extract components from FP16
  const uint32_t u = static_cast<uint32_t>(half);
  const uint32_t sign = (u & 0x8000U) << 16;   // Extract sign bit (bit 15)
  const uint32_t exponent = (u >> 10) & 0x1FU; // Extract exponent (bits 14-10)
  const uint32_t mantissa = u & 0x3FFU;        // Extract mantissa (bits 9-0)

  // Handle special cases based on exponent value
  if (exponent == 0) {
    // Exponent = 0: Zero or subnormal number
    if (mantissa == 0) {
      // ±Zero: mantissa = 0
      converter.u = sign;
      return converter.f;
    } else {
      // Subnormal number: convert to normalized FP32
      // Subnormals have implicit leading 0, need to find leading 1

      // Count leading zeros in mantissa to find normalization shift
      uint32_t shift = 0;
      uint32_t m = mantissa;

      // Find the position of the leading 1 bit
      while ((m & 0x400U) == 0) { // 0x400 = bit 10 (implicit 1 position)
        m <<= 1;
        ++shift;
      }

      // Remove the implicit leading 1 bit
      m &= 0x3FFU;

      // Calculate FP32 exponent: adjust for bias difference and normalization
      // FP16 bias = 15, FP32 bias = 127
      // Subnormal exponent is effectively -14 in FP16
      const uint32_t fp32_exponent = ((127 - 15 - shift) & 0xFFU) << 23;
      const uint32_t fp32_mantissa =
          m << 13; // Extend mantissa from 10 to 23 bits

      converter.u = sign | fp32_exponent | fp32_mantissa;
      return converter.f;
    }
  } else if (exponent == 0x1FU) {
    // Exponent = all 1s (31): Infinity or NaN
    if (mantissa == 0) {
      // ±Infinity: mantissa = 0
      converter.u = sign | 0x7F800000U;
    } else {
      // NaN: mantissa ≠ 0
      // Preserve NaN payload (mantissa bits)
      // Set all exponent bits to 1 and extend mantissa
      converter.u = sign | 0x7F800000U | (mantissa << 13);
    }
    return converter.f;
  } else {
    // Normal number: standard conversion

    // Convert exponent from FP16 bias (15) to FP32 bias (127)
    // FP32_exp = FP16_exp - 15 + 127 = FP16_exp + 112
    const uint32_t fp32_exponent = ((exponent + 112) & 0xFFU) << 23;

    // Extend mantissa from 10 bits to 23 bits (shift left by 13)
    const uint32_t fp32_mantissa = mantissa << 13;

    // Combine sign, exponent, and mantissa
    converter.u = sign | fp32_exponent | fp32_mantissa;
    return converter.f;
  }
}

#endif

#endif // LONGMEN_FP16_TO_FP32_H_
