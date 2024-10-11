#include <iostream>
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

using namespace std;
using namespace cv;

// Function to estimate illumination and enhance the image using NEON
void neon_enhance_image(const Mat &input, Mat &output, float gamma) {
    int rows = input.rows;
    int cols = input.cols;

    // Ensure the output matrix has the same size and type as the input
    output = Mat(rows, cols, input.type());

    // Iterate over the image in blocks of 16 pixels (as NEON works with 128-bit vectors)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j += 16) {
            // Load 16 pixels for each channel (R, G, B)
            uint8x16_t r = vld1q_u8(input.ptr(i, j));      // Load R channel
            uint8x16_t g = vld1q_u8(input.ptr(i, j + 1));  // Load G channel
            uint8x16_t b = vld1q_u8(input.ptr(i, j + 2));  // Load B channel

            // Calculate the maximum of the three channels to estimate illumination
            uint8x16_t max_rgb = vmaxq_u8(vmaxq_u8(r, g), b);

            // Perform gamma correction (simplified, with NEON, for each pixel)
            float32x4_t max_rgb_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(max_rgb)))));
            max_rgb_f32 = powq_f32(max_rgb_f32, gamma);

            // Store the result in the output matrix
            vst1q_u8(output.ptr(i, j), vmaxq_u8(max_rgb, max_rgb_f32));  // Apply back to output
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: ./neon_enhance <input_image> <output_image>" << endl;
        return -1;
    }

    // Read the input image
    Mat input_image = imread(argv[1], IMREAD_COLOR);
    if (input_image.empty()) {
        cout << "Error: Cannot load image " << argv[1] << endl;
        return -1;
    }

    Mat enhanced_image;

    // Parameters for enhancement (can be fine-tuned)
    float gamma = 0.6;  // Gamma correction value

    // Enhance the image using NEON
    neon_enhance_image(input_image, enhanced_image, gamma);

    // Save the enhanced image
    imwrite(argv[2], enhanced_image);

    cout << "Enhanced image saved as " << argv[2] << endl;

    return 0;
}