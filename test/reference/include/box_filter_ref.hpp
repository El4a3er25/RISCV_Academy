#pragma once
#include "image.hpp"
#include "box_filter_types.hpp"

namespace ref {
    template <typename T>
    void BoxFilterNxN(const Image<T>& input, 
                      Image<T>& output, 
                      FilterSize filter_size,
                      const BorderMode border_mode);
}
