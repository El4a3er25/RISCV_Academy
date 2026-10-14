#include "image.hpp"
#include "rgb2gray_vec.hpp"
#include "rgb2gray_ref.hpp"
#include "riscv_port.hpp"
#include "riscv_cv.hpp"

// This Example is dedicated to QEMU purely for Functional Verification and debugging.
// It reads a real image from disk, processes it, and writes the output back to disk.

int main(int argc, char** argv)
{
#ifndef RISCV_QEMU
    printf("This example is meant to be run on QEMU to generate output images.\n");
    printf("Please use the benchmark target for gem5 performance measurement.\n");
    return 0;
#else
    Image<uint8_t, ImageType::RGB> input_img;

    if (argc < 4)
    {
        printf("Usage: %s <input_rgb.ppm> <output_scalar.pgm> <output_vectorized.pgm>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_scalar_path = argv[2];
    const char* output_vector_path = argv[3];

    printf("Loading RGB image from: %s\n", input_path);
    input_img.Read(input_path);

    int width = input_img.Width();
    int height = input_img.Height();

    if (width == 0 || height == 0)
    {
        printf("Failed to read input image or image is empty.\n");
        return 1;
    }

    printf("Image size: %dx%d\n", width, height);

    Image<uint8_t, ImageType::GRAY> output_scalar(width, height);
    Image<uint8_t, ImageType::GRAY> output_vectorized(width, height);

    printf("Running Scalar RGB to Gray conversion...\n");
    ref::RGB2Gray(input_img, output_scalar);
    
    printf("Saving Scalar Gray image to: %s\n", output_scalar_path);
    output_scalar.Write(output_scalar_path);

    printf("Running Vectorized RGB to Gray conversion...\n");
    vec::RGB2Gray(input_img, output_vectorized);

    printf("Saving Vectorized Gray image to: %s\n", output_vector_path);
    output_vectorized.Write(output_vector_path);
    
    printf("Done! Use verify_RGB2Gray.py to verify functional correctness.\n");

    return 0;
#endif
}
