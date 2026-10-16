#include "benchmark.hpp"
#include "box_filter_ref.hpp"
#include "box_filter_vec.hpp"

int main()
{
#ifdef RISCV_BAREMETAL
    // Enable the vector unit before any auto-vectorized loop runs (nexus-am's
    // _start only enables the FPU).
    riscv_enable_vector();
#endif

    const int width = 128;
    const int height = 128;
    const int loop_count = 1;

    printf("BoxFilter benchmark: %dx%d, %d iterations\n", width, height, loop_count);

    Image<uint8_t> input(width, height);
    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    // Deterministic fill so runs are reproducible.
    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input);

    bool all_correct = true;

    // We can test different filter sizes
    const FilterSize filters[] = {FilterSize::K3x3, FilterSize::K5x5, FilterSize::K7x7};
    const char* filter_names[] = {"BoxFilter 3x3", "BoxFilter 5x5", "BoxFilter 7x7"};
    BorderMode border_mode = {BorderPolicy::CONSTANT, 0};

    for (int p = 0; p < 3; ++p)
    {
        const FilterSize filterSize = filters[p];

        uint64_t reference_cycles = 0, vectorized_cycles = 0;
        uint64_t reference_instrs = 0, vectorized_instrs = 0;

        for (int i = 0; i < loop_count; ++i)
        {
            Timer timer_scalar, timer_vectorized;

            timer_scalar.Start();
            ref::BoxFilterNxN<uint8_t>(input, output_reference, filterSize, border_mode);
            timer_scalar.Stop();

            timer_vectorized.Start();
            vec::BoxFilterNxN<uint8_t>(input, output_vectorized, filterSize, border_mode);
            timer_vectorized.Stop();

            reference_cycles += timer_scalar.ElapsedCycles();
            reference_instrs += timer_scalar.ElapsedInstructions();

            vectorized_cycles += timer_vectorized.ElapsedCycles();
            vectorized_instrs += timer_vectorized.ElapsedInstructions();

            if (!CheckCorrectness(output_reference, output_vectorized))
            {
                all_correct = false;
                break;
            }
        }

        PrintTime(filter_names[p], width * height,
                  vectorized_cycles / loop_count, vectorized_instrs / loop_count,
                  reference_cycles / loop_count, reference_instrs / loop_count);

        if (!all_correct)
        {
            break;
        }
    }

    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    else
    {
        printf("Output is wrong!\n");
        return 1;
    }
}
