#pragma once

// Shared utilities for image fill and correctness checking.

#include "image.hpp"
#include "riscv_port.hpp"
#include "box_filter_types.hpp"
#include <algorithm>

template <typename T>
T GetPixel(const Image<T>& input, int32_t x, int32_t y, const BorderMode& border_mode) {
    int32_t width = input.Width();
    int32_t height = input.Height();

    if (x >= 0 && x < width && y >= 0 && y < height) {
        return input.GetPixel(x, y);
    }

    switch (border_mode.policy) {
        case BorderPolicy::REPLICATE: {
            int32_t clamped_x = std::clamp(x, 0, width - 1);
            int32_t clamped_y = std::clamp(y, 0, height - 1);
            return input.GetPixel(clamped_x, clamped_y);
        }
        case BorderPolicy::CONSTANT: {
            return static_cast<T>(border_mode.value);
        }
        case BorderPolicy::MIRROR: {
            int32_t mirror_x = x;
            int32_t mirror_y = y;
            if (x < 0) mirror_x = -x;
            else if (x >= width) mirror_x = 2 * width - 2 - x;
            if (y < 0) mirror_y = -y;
            else if (y >= height) mirror_y = 2 * height - 2 - y;
            return input.GetPixel(mirror_x, mirror_y);
        }
        case BorderPolicy::NO_BORDER:
        default:
            return 0; 
    }
}


// Compare two images pixel-by-pixel, printing the first few mismatches.
template <typename T>
bool CheckCorrectness(const Image<T>& expected, const Image<T>& actual)
{
    if (expected.Width() != actual.Width() || expected.Height() != actual.Height())
    {
        return false;
    }

    bool is_correct = true;
    int mismatches = 0;

    for (int y = 0; y < expected.Height(); y++)
    {
        for (int x = 0; x < expected.Width(); x++)
        {
            if (expected.GetPixel(x, y) != actual.GetPixel(x, y))
            {
                is_correct = false;
                if (mismatches < 8) // cap the UART spam on large images
                {
                    printf("Mismatch at (%d, %d): expected = %d, actual = %d\n",
                           x, y,
                           static_cast<int>(expected.GetPixel(x, y)),
                           static_cast<int>(actual.GetPixel(x, y)));
                }
                ++mismatches;
            }
        }
    }

    if (mismatches > 0)
    {
        printf("Total mismatches: %d\n", mismatches);
    }
    return is_correct;
}

// Deterministic random fill using klib's rand() with a fixed seed
template <typename T>
class RandomInt
{
public:
    RandomInt(T min, T max, unsigned int seed = 0xC0FFEE)
        : min_(static_cast<int>(min)), span_(static_cast<int>(max) - static_cast<int>(min) + 1)
    {
        srand(seed);
    }

    T get()
    {
        return static_cast<T>(min_ + (rand() % span_));
    }

    void ImageRandomInitialize(Image<T>& image)
    {
        for (int y = 0; y < image.Height(); y++)
        {
            for (int x = 0; x < image.Width(); x++)
            {
                image.SetPixel(x, y, get());
            }
        }
    }

private:
    int min_;
    int span_;
};
