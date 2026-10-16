import cv2
import numpy as np
import sys

# 1. Load the original image
img = cv2.imread("../images/image0.pgm", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Could not load original image.")
    sys.exit(1)

# 2. Run OpenCV Box Filter (Golden Reference)
# normalize=True averages the pixels. BORDER_CONSTANT applies zero padding.
opencv_out = cv2.boxFilter(img, -1, (3, 3), normalize=True, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite("output_opencv.pgm", opencv_out) # Save to see it visually

# 3. Load our C++ output
our_out = cv2.imread("output_scalar.pgm", cv2.IMREAD_GRAYSCALE)
if our_out is None:
    print("Error: Could not load output_scalar.pgm.")
    sys.exit(1)

# 4. Compare them pixel by pixel
diff = cv2.absdiff(opencv_out, our_out)
non_zero = np.count_nonzero(diff > 1)

if non_zero == 0:
    print("SUCCESS: Your boxfilter Scalar implementation matches OpenCV!")
else:
    print(f"FAILED: Found {non_zero} differing pixels. Check border logic or rounding.")
