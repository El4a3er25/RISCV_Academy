#include "box_filter_ref.hpp"
#include "riscv_utils.hpp"
#include <cassert>

namespace {
    template <typename T, FilterSize filter_size>
    void BoxFilterImpl(const Image<T>& input, Image<T>& output, const BorderMode& border_mode) {
        constexpr int32_t kSize = static_cast<int32_t>(filter_size);
        constexpr int32_t kRadius = kSize / 2;
        constexpr int32_t kArea = kSize * kSize;

        for (int32_t y = 0; y < input.Height(); ++y) {
            for (int32_t x = 0; x < input.Width(); ++x) {
                uint32_t sum = 0;
                for (int32_t ky = -kRadius; ky <= kRadius; ++ky) {
                    for (int32_t kx = -kRadius; kx <= kRadius; ++kx) {
                        sum += GetPixel(input, x + kx, y + ky, border_mode);
                    }
                }
                output.SetPixel(x, y, static_cast<T>(sum / kArea));
            }
        }
    }
}

namespace ref {
    template <typename T>
    void BoxFilterNxN(const Image<T>& input, Image<T>& output, FilterSize filter_size, const BorderMode border_mode) {
        assert(input.Width() == output.Width() && input.Height() == output.Height());
        
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
                assert(false && "Unsupported filter size");
                break;
        }
    }

    // Explicit Instantiation
    template void BoxFilterNxN<uint8_t>(
        const Image<uint8_t>&, Image<uint8_t>&, FilterSize, const BorderMode);
}
