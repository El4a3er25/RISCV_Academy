#include "gtest_lite.hpp"
#include <tuple>
#include "riscv_cv.hpp"
#include "box_filter_ref.hpp"
#include "box_filter_vec.hpp"
#include "test_utils.hpp"

class BoxFilterTest : public ::testing::TestWithParam<std::tuple<int, int, FilterSize, BorderPolicy>>
{
};

TEST_P(BoxFilterTest, VectorizedMatchesReference) // TEST_P is a macro
{
    const auto [width, height, filterSize, borderPolicy] = GetParam();

    BorderMode borderMode = {borderPolicy, 0};

    printf("[ %-7s %16d ]\n", "width:", width);
    printf("[ %-7s %16d ]\n", "height:", height);

    Image<uint8_t> input(width, height);
    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input);

    ref::BoxFilterNxN<uint8_t>(input, output_reference, filterSize, borderMode);
    vec::BoxFilterNxN<uint8_t>(input, output_vectorized, filterSize, borderMode);

    ExpectImagesEqual(output_reference, output_vectorized);
}

INSTANTIATE_TEST_SUITE_P(ImageSizesAndBorders, BoxFilterTest, // macro
    ::testing::Combine(
        ::testing::Values(16, 31, 47),
        ::testing::Values(16, 22),
        ::testing::Values(FilterSize::K3x3, FilterSize::K5x5, FilterSize::K7x7),
        ::testing::Values(BorderPolicy::CONSTANT, BorderPolicy::REPLICATE, BorderPolicy::MIRROR, BorderPolicy::NO_BORDER))
);
