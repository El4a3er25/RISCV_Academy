#include "gtest_lite.hpp"

#include <tuple>

#include "riscv_cv.hpp"
#include "rgb2gray_ref.hpp"
#include "rgb2gray_vec.hpp"
#include "test_utils.hpp"

class RGB2GrayTest : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(RGB2GrayTest, VectorizedMatchesReference)
{
    const auto [width, height] = GetParam();

    printf("[ %-7s %16d ]\n", "width:", width);
    printf("[ %-7s %16d ]\n", "height:", height);

    Image<uint8_t, ImageType::RGB> input(width, height);
    Image<uint8_t, ImageType::GRAY> output_reference(width, height);
    Image<uint8_t, ImageType::GRAY> output_vectorized(width, height);

    RandomInt<uint8_t> random(0, 255);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            input.SetPixel(x, y, random.get(), 0);
            input.SetPixel(x, y, random.get(), 1);
            input.SetPixel(x, y, random.get(), 2);
        }
    }

    ref::RGB2Gray(input, output_reference);
    vec::RGB2Gray(input, output_vectorized);

    ExpectImagesEqual(output_reference, output_vectorized);
}

// VLEN = 128 bit, SEW = 8 bit, LMUL = 2 -> vlmax = 32 elements.
// main loop unrolls 2-way: 2 * vlmax = 64 elements per iteration.
INSTANTIATE_TEST_SUITE_P(ImageSizes, RGB2GrayTest,
    ::testing::Combine(
        ::testing::Values(
            1, 7,        // tail test: width < VLEN 
            16, 31,      // tail test: below single-chunk boundary 
            32, 33,      // exact 1 chunk vs 1 chunk + 1 tail
            63, 64, 65,  // 2-way unroll boundary 
            127, 128     // multiple unrolled chunks
        ),
        ::testing::Values(
            1,           // single row 
            7,           // Small odd height
            16,          // even height
            23           // prime height
        )
    )
);
