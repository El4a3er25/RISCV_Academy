#include "riscv_cv.hpp"
#include "box_filter_ref.hpp"
#include "box_filter_vec.hpp"

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    // Load the input image
    Image<uint8_t> img;

#ifdef RISCV_QEMU
    img.Read("/home/RISCV_Academy/images/image0.pgm");
#else
    // Baremetal cannot read files
#endif

    if (img.Width() == 0 || img.Height() == 0)
    {
        printf("Error: Failed to load image!\n");
        return 1;
    }

    Image<uint8_t> out_ref(img.Width(), img.Height());
    Image<uint8_t> out_vec(img.Width(), img.Height());

    // Setup Border Mode
    BorderMode border_mode = {BorderPolicy::CONSTANT, 0};

    // Run Scalar (Reference) Implementation
    printf("Running Scalar (Reference) Implementation...\n");
    ref::BoxFilterNxN<uint8_t>(img, out_ref, FilterSize::K3x3, border_mode);

    // Run Vector Implementation
    printf("Running Vector Implementation...\n");
    vec::BoxFilterNxN<uint8_t>(img, out_vec, FilterSize::K3x3, border_mode);

#ifdef RISCV_QEMU
    out_ref.Write("/home/RISCV_Academy/images/output_ref.pgm");
    out_vec.Write("/home/RISCV_Academy/images/output_vec.pgm");
    printf("Wrote output_ref.pgm and output_vec.pgm (%dx%d)\n", out_ref.Width(), out_ref.Height());
#endif

    printf("Example passed.\n");
    return 0;
}
