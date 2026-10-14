#include "rgb2gray_ref.hpp"

namespace ref {

    template <typename T>
    void RGB2Gray(const Image<T, ImageType::RGB>& input, 
                  Image<T, ImageType::GRAY>& output) 
    {
        int width = input.Width();
        int height = input.Height();
        const T* in_ptr = input.GetPtr(0, 0);
        T* out_ptr = output.GetPtr(0, 0);
        int total_pixels = width * height;

        for (int i = 0; i < total_pixels; ++i) {
            T r = in_ptr[i * 3];
            T g = in_ptr[i * 3 + 1];
            T b = in_ptr[i * 3 + 2];
            
            // Y = 0.299*R + 0.587*G + 0.114*B
            // Fixed-point with rounding: Y = (R*77 + G*150 + B*29 + 128) >> 8
            uint16_t y = (r * 77 + g * 150 + b * 29 + 128) >> 8;
            out_ptr[i] = static_cast<T>(y);
        }
    }

    // Explicit instantiation
    template void RGB2Gray<uint8_t>(const Image<uint8_t, ImageType::RGB>&, Image<uint8_t, ImageType::GRAY>&);

}
