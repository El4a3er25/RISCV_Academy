#include "box_filter_vec.hpp"
#include "riscv_utils.hpp"
#include <riscv_vector.h>
#include <algorithm>


#ifdef __has_include
#if __has_include(<alloca.h>)
#include <alloca.h>
#endif
#endif

// Some bare-metal systems use __builtin_alloca
#ifndef alloca
#define alloca __builtin_alloca
#endif

namespace {
    template <typename T, FilterSize filter_size>
    void BoxFilterImpl(const Image<T>& input, Image<T>& output, BorderMode border_mode) {
        const int32_t width = input.Width();
        const int32_t height = input.Height();
        constexpr int32_t kSize = static_cast<int32_t>(filter_size);
        constexpr int32_t kRadius = kSize / 2;
        constexpr int32_t kArea = kSize * kSize;

        constexpr uint16_t kMagic = (filter_size == FilterSize::K3x3) ? 1821 :
                                    (filter_size == FilterSize::K5x5) ? 5243 :
                                    (filter_size == FilterSize::K7x7) ? 2675 : 1;
        constexpr size_t   kShift = (filter_size == FilterSize::K3x3) ? 14 :
                                    (filter_size == FilterSize::K5x5) ? 17 :
                                    (filter_size == FilterSize::K7x7) ? 17 : 0;

        // 1. Process interior pixels using Enhanced Separable RVV 
        int32_t x = kRadius;
        int32_t remaining_width = width - 2 * kRadius;
        
        // Outer Loop: Iterate horizontally in strips of size 'vl'
        while (remaining_width > 0) {
            size_t vl = __riscv_vsetvl_e8m1(remaining_width);
            
            // Allocate a ring buffer on the stack for this strip.
            uint16_t ring_buffer[11 * 256];
            
            // Initialize the vector accumulator for this strip
            vuint16m2_t accumulator = __riscv_vmv_v_x_u16m2(0, vl);
            
            // Pre-fill the ring buffer with the top (kSize - 1) rows
            for (int i = 0; i < kSize - 1; ++i) {
                int32_t current_y = i; // from 0 to kSize-2
                
                // Compute Horizontal Sum for the current row
                vuint16m2_t h_sum = __riscv_vmv_v_x_u16m2(0, vl);
                for (int kx = -kRadius; kx <= kRadius; ++kx) {
                    const uint8_t* ptr = input.GetPtr(x + kx, current_y);
                    vuint8m1_t p = __riscv_vle8_v_u8m1(ptr, vl);
                    h_sum = __riscv_vwaddu_wv_u16m2(h_sum, p, vl);
                }
                
                // Store the horizontal sum in the ring buffer
                // Use modulo index to match the inner loop's ring_idx convention
                int32_t ring_idx = current_y % kSize;
                __riscv_vse16_v_u16m2(&ring_buffer[ring_idx * vl], h_sum, vl);
                
                // Add to the initial accumulator
                accumulator = __riscv_vadd_vv_u16m2(accumulator, h_sum, vl);
            }
            
            // Inner Loop: Slide the window vertically down the strip
            for (int y = kRadius; y < height - kRadius; ++y) {
                int32_t incoming_y = y + kRadius;      // The new row entering the window from the bottom
                int32_t outgoing_y = y - kRadius;      // The old row leaving the window from the top
                int32_t ring_idx_in  = incoming_y % kSize;
                int32_t ring_idx_out = outgoing_y % kSize;
                
                // 1. Compute Horizontal Sum for the INCOMING row
                vuint16m2_t incoming_hsum = __riscv_vmv_v_x_u16m2(0, vl);
                for (int kx = -kRadius; kx <= kRadius; ++kx) {
                    const uint8_t* ptr = input.GetPtr(x + kx, incoming_y);
                    vuint8m1_t p = __riscv_vle8_v_u8m1(ptr, vl);
                    incoming_hsum = __riscv_vwaddu_wv_u16m2(incoming_hsum, p, vl);
                }
                
                // 2. Add incoming row to accumulator
                accumulator = __riscv_vadd_vv_u16m2(accumulator, incoming_hsum, vl);
                
                // 3. Store incoming row to the ring buffer (overwriting the oldest row)
                __riscv_vse16_v_u16m2(&ring_buffer[ring_idx_in * vl], incoming_hsum, vl);
                
                // 4. Calculate Average and Output (Pipelined Reciprocal Multiplication)
                vuint32m4_t prod = __riscv_vwmulu_vx_u32m4(accumulator, kMagic, vl);
                vuint16m2_t avg = __riscv_vnsrl_wx_u16m2(prod, kShift, vl);
                vuint8m1_t result = __riscv_vncvt_x_x_w_u8m1(avg, vl);
                __riscv_vse8_v_u8m1(output.GetPtr(x, y), result, vl);
                
                // 5. Subtract OUTGOING row from accumulator for the next iteration
                vuint16m2_t outgoing_hsum = __riscv_vle16_v_u16m2(&ring_buffer[ring_idx_out * vl], vl);
                accumulator = __riscv_vsub_vv_u16m2(accumulator, outgoing_hsum, vl);
            }
            
            // Advance to the next horizontal strip
            remaining_width -= vl;
            x += vl;
        }
        
        // 2. Handle borders (Top, Bottom, Left, Right) using a scalar fallback
        // Top border
        for (int y = 0; y < kRadius; ++y) {
            for (int x = 0; x < width; ++x) {
                uint32_t sum = 0;
                for (int ky = -kRadius; ky <= kRadius; ++ky) {
                    for (int kx = -kRadius; kx <= kRadius; ++kx) {
                        sum += GetPixel(input, x + kx, y + ky, border_mode);
                    }
                }
                output.SetPixel(x, y, static_cast<T>(sum / kArea));
            }
        }
        
        // Bottom border
        for (int y = height - kRadius; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint32_t sum = 0;
                for (int ky = -kRadius; ky <= kRadius; ++ky) {
                    for (int kx = -kRadius; kx <= kRadius; ++kx) {
                        sum += GetPixel(input, x + kx, y + ky, border_mode);
                    }
                }
                output.SetPixel(x, y, static_cast<T>(sum / kArea));
            }
        }
        
        // Left and Right borders for interior rows
        for (int y = kRadius; y < height - kRadius; ++y) {
            // Left border
            for (int x = 0; x < kRadius; ++x) {
                uint32_t sum = 0;
                for (int ky = -kRadius; ky <= kRadius; ++ky) {
                    for (int kx = -kRadius; kx <= kRadius; ++kx) {
                        sum += GetPixel(input, x + kx, y + ky, border_mode);
                    }
                }
                output.SetPixel(x, y, static_cast<T>(sum / kArea));
            }
            // Right border
            for (int x = width - kRadius; x < width; ++x) {
                uint32_t sum = 0;
                for (int ky = -kRadius; ky <= kRadius; ++ky) {
                    for (int kx = -kRadius; kx <= kRadius; ++kx) {
                        sum += GetPixel(input, x + kx, y + ky, border_mode);
                    }
                }
                output.SetPixel(x, y, static_cast<T>(sum / kArea));
            }
        }
    }
}

namespace vec {
    template <typename T>
    void BoxFilterNxN(const Image<T>& input, 
                      Image<T>& output, 
                      FilterSize filter_size,
                      BorderMode border_mode) 
    {
        switch (filter_size) {
            case FilterSize::K3x3:
                BoxFilterImpl<T, FilterSize::K3x3>(input, output, border_mode);
                break;
            case FilterSize::K5x5:
                BoxFilterImpl<T, FilterSize::K5x5>(input, output, border_mode);
                break;
            case FilterSize::K7x7:
                BoxFilterImpl<T, FilterSize::K7x7>(input, output, border_mode);
                break;
            default:
                break;
        }
    }

    template void BoxFilterNxN<uint8_t>(const Image<uint8_t>&, Image<uint8_t>&, FilterSize, BorderMode);
}
