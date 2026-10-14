#include "benchmark.hpp"
#include "rgb2gray_vec.hpp"
#include "rgb2gray_ref.hpp"
#include "sample.hpp"

// This Benchmark is dedicated to gem5 purely for Performance Measurement (Cycle accurate).
// It uses a real image compiled into a header file to avoid bare-metal file I/O issues.

int main()
{
#ifdef RISCV_QEMU
    printf("Benchmarks are meant to be run on gem5. Skipping QEMU run.\n");
    return 0;
#else
    // Enable the vector unit on bare-metal
    riscv_enable_vector();

    // We use a small 64x64 crop of the real image for the benchmark
    // to avoid overflowing the bare-metal heap memory limit (which caused the map::at crash).
    const int width = 64;
    const int height = 64;
    const int loop_count = 5;

    printf("RGB2Gray benchmark: %dx%d, %d iterations\n", width, height, loop_count);

    Image<uint8_t, ImageType::RGB> input;
    
    // Load a 64x64 chunk from the compiled header array to save heap space
    input.Read(width, height, sample_data_data);

    Image<uint8_t, ImageType::GRAY> output_scalar(width, height);
    Image<uint8_t, ImageType::GRAY> output_vectorized(width, height);

    uint64_t scalar_cycles = 0, vectorized_cycles = 0;
    uint64_t scalar_instrs = 0, vectorized_instrs = 0;

    for (int i = 0; i < loop_count; ++i)
    {
        Timer timer_scalar, timer_vectorized;

        timer_scalar.Start();
        ref::RGB2Gray(input, output_scalar);
        timer_scalar.Stop();

        timer_vectorized.Start();
        vec::RGB2Gray(input, output_vectorized);
        timer_vectorized.Stop();

        scalar_cycles += timer_scalar.ElapsedCycles();
        scalar_instrs += timer_scalar.ElapsedInstructions();

        vectorized_cycles += timer_vectorized.ElapsedCycles();
        vectorized_instrs += timer_vectorized.ElapsedInstructions();
    }

    // PrintTime is a helper in benchmark.hpp that prints a nicely formatted table
    PrintTime("RGB2Gray_Scalar", width * height,
              0, 0, // No reference to compare against here for the scalar line
              scalar_cycles / loop_count, scalar_instrs / loop_count);
              
    PrintTime("RGB2Gray_Vector", width * height,
              vectorized_cycles / loop_count, vectorized_instrs / loop_count,
              scalar_cycles / loop_count, scalar_instrs / loop_count);

    printf("Benchmark finished.\n");
    return 0;
#endif
}
