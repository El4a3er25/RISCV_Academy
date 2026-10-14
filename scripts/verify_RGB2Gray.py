import cv2
import sys
import numpy as np

def main():
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} <input_rgb.ppm> <output_scalar.pgm> <output_vectorized.pgm>")
        sys.exit(1)

    input_path = sys.argv[1]
    scalar_path = sys.argv[2]
    vectorized_path = sys.argv[3]

    # Read input RGB image
    img_rgb = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_rgb is None:
        print(f"Error: Could not read {input_path}")
        sys.exit(1)
        
    # Read our outputs
    scalar_gray = cv2.imread(scalar_path, cv2.IMREAD_GRAYSCALE)
    if scalar_gray is None:
        print(f"Error: Could not read {scalar_path}")
        sys.exit(1)
        
    vector_gray = cv2.imread(vectorized_path, cv2.IMREAD_GRAYSCALE)
    if vector_gray is None:
        print(f"Error: Could not read {vectorized_path}")
        sys.exit(1)

    # Convert RGB to Gray using OpenCV
    cv2_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    print("--- SCALAR VERIFICATION ---")
    diff_scalar = cv2.absdiff(cv2_gray, scalar_gray)
    max_diff_s = np.max(diff_scalar)
    print(f"Max absolute difference: {max_diff_s}")
    if max_diff_s <= 2:
        print("SUCCESS: Scalar output matches OpenCV.")
    else:
        print("WARNING: Large difference found in Scalar output!")

    print("\n--- VECTORIZED VERIFICATION ---")
    diff_vector = cv2.absdiff(cv2_gray, vector_gray)
    max_diff_v = np.max(diff_vector)
    print(f"Max absolute difference: {max_diff_v}")
    if max_diff_v <= 2:
        print("SUCCESS: Vectorized output matches OpenCV.")
    else:
        print("WARNING: Large difference found in Vectorized output!")

    print("\n--- SCALAR VS VECTORIZED EXACT MATCH ---")
    diff_sv = cv2.absdiff(scalar_gray, vector_gray)
    max_diff_sv = np.max(diff_sv)
    if max_diff_sv == 0:
        print("SUCCESS: Vectorized output is an EXACT BIT-FOR-BIT match with Scalar output!")
    else:
        print(f"WARNING: Vectorized differs from Scalar by max {max_diff_sv}")

if __name__ == "__main__":
    main()
