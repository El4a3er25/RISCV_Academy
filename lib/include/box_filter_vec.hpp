#pragma once
#include "image.hpp"
#include "box_filter_types.hpp"

namespace vec {
    template <typename T>
    void BoxFilterNxN(const Image<T>& input, 
                      Image<T>& output, 
                      FilterSize filter_size,
                      BorderMode border_mode);
}
