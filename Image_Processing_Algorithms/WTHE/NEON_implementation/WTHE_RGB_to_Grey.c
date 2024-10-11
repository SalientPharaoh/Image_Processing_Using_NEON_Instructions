#include <arm_neon.h>
#include <stdio.h>

#define MAX_GRAY_LEVEL 256

// Function to clamp values to 255 using 8-bit operations
static inline uint8x16_t clamp_to_255(uint8x16_t vec) {
    uint8x16_t max_val = vdupq_n_u8(255);  // Vector with all elements set to 255
    return vminq_u8(vec, max_val);         // Clamp values to 255
}

// Function to convert grayscale image back to RGB using 8-bit operations
void grayscale_to_RGB_neon(uint8_t *gray_image, uint8_t *rgb_image, int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; i += 16) {
        // Load 16 grayscale pixel values (8-bit)
        uint8x16_t gray_neon = vld1q_u8(&gray_image[i]);

        // Set R, G, B channels to the grayscale value (8-bit)
        uint8x16x3_t rgb_neon;
        rgb_neon.val[0] = gray_neon;  // R channel
        rgb_neon.val[1] = gray_neon;  // G channel
        rgb_neon.val[2] = gray_neon;  // B channel

        // Store the RGB values back to the image
        vst3q_u8(&rgb_image[i * 3], rgb_neon);
    }
}

// Function for Weighted Threshold Histogram Equalization (WTHE) using 8-bit operations
void WTHE_neon(uint8_t *gray_image, int width, int height, float r) {
    int size = width * height;
    uint32_t histogram[MAX_GRAY_LEVEL] = {0};
    float PDF[MAX_GRAY_LEVEL] = {0.0f};
    float CDF[MAX_GRAY_LEVEL] = {0.0f};
    uint8_t output_image[size];
    float total_pixels = (float)size;

    // 1. Compute the Histogram
    for (int i = 0; i < size; i++) {
        histogram[gray_image[i]]++;
    }

    // 2. Modify the PDF with weighting and thresholding
    float threshold = 0.01;  // Set a suitable threshold for clamping
    for (int i = 0; i < MAX_GRAY_LEVEL; i++) {
        PDF[i] = (float)histogram[i] / total_pixels;
        PDF[i] = (PDF[i] > threshold) ? threshold : PDF[i];  // Apply threshold
        PDF[i] = powf(PDF[i], r);  // Apply weighting
    }

    // 3. Compute the CDF
    CDF[0] = PDF[0];
    for (int i = 1; i < MAX_GRAY_LEVEL; i++) {
        CDF[i] = CDF[i - 1] + PDF[i];
    }

    // 4. Map input pixel intensities using the CDF with 8-bit operations and clamping
    for (int i = 0; i < size; i += 16) {
        // Load 16 grayscale pixels (8-bit)
        uint8x16_t pixels_neon = vld1q_u8(&gray_image[i]);
        uint8x16_t result_neon;

        // Map each pixel using CDF and threshold to ensure values don't exceed 255
        for (int j = 0; j < 16; j++) {
            uint8_t pixel_value = vgetq_lane_u8(pixels_neon, j);
            float new_pixel_value = CDF[pixel_value] * (MAX_GRAY_LEVEL - 1);

            // Ensure the new value is within 0-255
            new_pixel_value = (new_pixel_value > 255.0f) ? 255.0f : new_pixel_value;

            result_neon = vsetq_lane_u8((uint8_t)new_pixel_value, result_neon, j);
        }

        // Clamp the result values between 0 and 255
        result_neon = clamp_to_255(result_neon);

        // Store the result back to the output image
        vst1q_u8(&output_image[i], result_neon);
    }

    // Optionally adjust brightness (W_out, M_adj) if required
    float W_out = 1.0f;  // Scaling factor for brightness
    float M_adj = 0.0f;  // Brightness adjustment
    for (int i = 0; i < size; i++) {
        output_image[i] = (uint8_t)(W_out * output_image[i] + M_adj);
        output_image[i] = (output_image[i] > 255) ? 255 : output_image[i];  // Ensure it stays within 0-255
    }

    // Return the output_image (grayscale) for further processing
}

// Function to convert RGB image to Grayscale using 8-bit operations
void RGB_to_grayscale_neon(uint8_t *rgb_image, uint8_t *gray_image, int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; i += 16) {
        // Load 16 RGB pixels (each pixel is 3 bytes, so 48 bytes total)
        uint8x16x3_t rgb_neon = vld3q_u8(&rgb_image[i * 3]);

        // Convert each RGB triplet to grayscale using 8-bit operations and clamp to 0-255
        uint8x16_t r_neon = rgb_neon.val[0];
        uint8x16_t g_neon = rgb_neon.val[1];
        uint8x16_t b_neon = rgb_neon.val[2];

        // Grayscale formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
        uint16x8_t gray_neon_l = vmlal_n_u8(vmull_n_u8(vget_low_u8(r_neon), 77), vget_low_u8(g_neon), 150);
        gray_neon_l = vmlal_n_u8(gray_neon_l, vget_low_u8(b_neon), 29);

        uint16x8_t gray_neon_h = vmlal_n_u8(vmull_n_u8(vget_high_u8(r_neon), 77), vget_high_u8(g_neon), 150);
        gray_neon_h = vmlal_n_u8(gray_neon_h, vget_high_u8(b_neon), 29);

        // Shift right by 8 bits to normalize the result to 8-bit and combine the two halves
        uint8x16_t gray_neon = vcombine_u8(vrshrn_n_u16(gray_neon_l, 8), vrshrn_n_u16(gray_neon_h, 8));

        // Store the result into the grayscale image
        vst1q_u8(&gray_image[i], gray_neon);
    }
}

int main() {
    // Sample usage with a 256x256 RGB image
    int width = 256;
    int height = 256;
    uint8_t rgb_image[256 * 256 * 3];  // Initialize with your RGB image data
    uint8_t gray_image[256 * 256];     // Grayscale image
    uint8_t output_rgb_image[256 * 256 * 3];  // Output RGB image

    // Convert RGB image to grayscale
    RGB_to_grayscale_neon(rgb_image, gray_image, width, height);

    // Apply WTHE to the grayscale image
    float r = 0.5f;  // Degree of enhancement
    WTHE_neon(gray_image, width, height, r);

    // Convert enhanced grayscale image back to RGB
    grayscale_to_RGB_neon(gray_image, output_rgb_image, width, height);

    // Now output_rgb_image contains the final enhanced RGB image
    return 0;
}