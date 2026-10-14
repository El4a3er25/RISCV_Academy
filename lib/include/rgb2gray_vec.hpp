#pragma once
#include "image.hpp"
#include "types.hpp"

namespace vec {
    template <typename T>
    void RGB2Gray(const Image<T, ImageType::RGB>& input, 
                  Image<T, ImageType::GRAY>& output);
}
