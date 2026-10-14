#include "rgb2gray_vec.hpp"
#include <riscv_vector.h>

namespace vec {

    template <typename T>
    void RGB2Gray(const Image<T, ImageType::RGB>& input, 
                  Image<T, ImageType::GRAY>& output) 
    {
        const uint8_t* in_ptr  = input.GetPtr(0, 0);
        uint8_t*       out_ptr = output.GetPtr(0, 0);

        const size_t total_pixels = input.Width() * input.Height();
        const size_t vlmax = __riscv_vsetvlmax_e8m2();
        size_t i = 0;

        const size_t PREFETCH_DISTANCE = 512;

        // 2x Unrolled Main Loop (Proven Sweet Spot for XiangShan 1-Vector Issue Queue)
        for (; i + 2 * vlmax <= total_pixels; i += 2 * vlmax)
        {
            // Software Prefetching
            __builtin_prefetch(in_ptr + PREFETCH_DISTANCE, 0, 0);
            __builtin_prefetch(out_ptr + (PREFETCH_DISTANCE / 3), 1, 0);

            // Base + Offset Addressing
            const uint8_t* in_ptr_B = in_ptr + (vlmax * 3);
            uint8_t* out_ptr_B = out_ptr + vlmax;

            // Load
            const vuint8m2x3_t RGB_A = __riscv_vlseg3e8_v_u8m2x3(in_ptr, vlmax);
            const vuint8m2x3_t RGB_B = __riscv_vlseg3e8_v_u8m2x3(in_ptr_B, vlmax);

            const vuint8m2_t R_A = __riscv_vget_v_u8m2x3_u8m2(RGB_A, 0);
            const vuint8m2_t G_A = __riscv_vget_v_u8m2x3_u8m2(RGB_A, 1);
            const vuint8m2_t B_A = __riscv_vget_v_u8m2x3_u8m2(RGB_A, 2);

            const vuint8m2_t R_B = __riscv_vget_v_u8m2x3_u8m2(RGB_B, 0);
            const vuint8m2_t G_B = __riscv_vget_v_u8m2x3_u8m2(RGB_B, 1);
            const vuint8m2_t B_B = __riscv_vget_v_u8m2x3_u8m2(RGB_B, 2);

            // Interleaved Multiply-Accumulate
            vuint16m4_t acc_A = __riscv_vwmulu_vx_u16m4(R_A, 77, vlmax);
            vuint16m4_t acc_B = __riscv_vwmulu_vx_u16m4(R_B, 77, vlmax);

            acc_A = __riscv_vwmaccu_vx_u16m4(acc_A, 150, G_A, vlmax);
            acc_B = __riscv_vwmaccu_vx_u16m4(acc_B, 150, G_B, vlmax);

            acc_A = __riscv_vwmaccu_vx_u16m4(acc_A, 29, B_A, vlmax);
            acc_B = __riscv_vwmaccu_vx_u16m4(acc_B, 29, B_B, vlmax);

            // Fused Narrow, Shift, Round and Clip
            const vuint8m2_t gray_A = __riscv_vnclipu_wx_u8m2(acc_A, 8, __RISCV_VXRM_RNU, vlmax);
            const vuint8m2_t gray_B = __riscv_vnclipu_wx_u8m2(acc_B, 8, __RISCV_VXRM_RNU, vlmax);

            // Store
            __riscv_vse8_v_u8m2(out_ptr, gray_A, vlmax);
            __riscv_vse8_v_u8m2(out_ptr_B, gray_B, vlmax);

            in_ptr  += vlmax * 6;
            out_ptr += vlmax * 2;
        }

        // Tail Loop
        size_t vl = 0;
        for (; i < total_pixels; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(total_pixels - i);

            const vuint8m2x3_t RGB = __riscv_vlseg3e8_v_u8m2x3(in_ptr, vl);
            const vuint8m2_t R = __riscv_vget_v_u8m2x3_u8m2(RGB, 0);
            const vuint8m2_t G = __riscv_vget_v_u8m2x3_u8m2(RGB, 1);
            const vuint8m2_t B = __riscv_vget_v_u8m2x3_u8m2(RGB, 2);

            vuint16m4_t acc = __riscv_vwmulu_vx_u16m4(R, 77, vl);
            acc = __riscv_vwmaccu_vx_u16m4(acc, 150, G, vl);
            acc = __riscv_vwmaccu_vx_u16m4(acc, 29, B, vl);

            const vuint8m2_t gray = __riscv_vnclipu_wx_u8m2(acc, 8, __RISCV_VXRM_RNU, vl);
            
            __riscv_vse8_v_u8m2(out_ptr, gray, vl);

            in_ptr  += vl * 3;
            out_ptr += vl;
        }
    }

    template void RGB2Gray<uint8_t>(const Image<uint8_t, ImageType::RGB>&, Image<uint8_t, ImageType::GRAY>&);

}