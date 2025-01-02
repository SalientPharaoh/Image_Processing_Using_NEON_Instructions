/**
 * @file norm-denorm.c
 * @brief NEON-optimized image normalization and denormalization implementation
 * 
 * This file implements fast image normalization and denormalization using ARM NEON
 * SIMD instructions. It provides functions to convert between uint8 [0, 255] and
 * normalized float [0.0, 1.0] representations of image data.
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/**
 * @brief Normalize an 8-bit image to floating-point values using NEON
 * 
 * This function converts uint8 values [0, 255] to normalized float values [0.0, 1.0]
 * using NEON SIMD instructions for improved performance. The function processes
 * 16 pixels at a time.
 * 
 * Algorithm steps:
 * 1. Load 16 uint8 values
 * 2. Convert to float32 through progressive widening
 * 3. Multiply by 1/255 to normalize
 * 
 * @param input_image Input uint8 image array
 * @param output_image Output float image array (normalized values)
 * @param num_pixels Number of pixels to process
 */
void normalize_image_neon(uint8_t* input_image, float* output_image, int num_pixels) {
    // Initialize reciprocal of 255 for normalization
    float32x4_t v_recip_255 = vdupq_n_f32(1.0f / 255.0f);

    // Process 16 pixels at a time
    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 uint8 pixels
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        // Widen to uint16 (8 pixels each)
        uint16x8_t v_pixels_low = vmovl_u8(vget_low_u8(v_pixels));    // Lower 8 pixels
        uint16x8_t v_pixels_high = vmovl_u8(vget_high_u8(v_pixels));  // Upper 8 pixels

        // Convert to float32 (4 pixels each)
        float32x4_t v_float_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_low)));   // First 4
        float32x4_t v_float_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_low)));  // Next 4
        float32x4_t v_float_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_high)));  // Next 4
        float32x4_t v_float_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_high))); // Last 4

        // Normalize to [0.0, 1.0] by multiplying with 1/255
        v_float_0 = vmulq_f32(v_float_0, v_recip_255);
        v_float_1 = vmulq_f32(v_float_1, v_recip_255);
        v_float_2 = vmulq_f32(v_float_2, v_recip_255);
        v_float_3 = vmulq_f32(v_float_3, v_recip_255);

        // Store normalized results
        vst1q_f32(output_image + i, v_float_0);
        vst1q_f32(output_image + i + 4, v_float_1);
        vst1q_f32(output_image + i + 8, v_float_2);
        vst1q_f32(output_image + i + 12, v_float_3);
    }
}

/**
 * @brief Denormalize a floating-point image to 8-bit values using NEON
 * 
 * This function converts normalized float values [0.0, 1.0] back to uint8 values
 * [0, 255] using NEON SIMD instructions. The function processes 16 pixels at a time
 * and includes proper rounding.
 * 
 * Algorithm steps:
 * 1. Load 16 float values
 * 2. Multiply by 255 and add 0.5 for rounding
 * 3. Convert to integers with saturation
 * 4. Narrow to uint8 with saturation
 * 
 * @param input_image Input float image array (normalized values)
 * @param output_image Output uint8 image array
 * @param num_pixels Number of pixels to process
 */
void denormalize_image_neon(float* input_image, uint8_t* output_image, int num_pixels) {
    // Initialize constants for denormalization
    float32x4_t v_255 = vdupq_n_f32(255.0f);  // For scaling back to [0, 255]
    float32x4_t v_0_5 = vdupq_n_f32(0.5f);    // For rounding

    // Process 16 pixels at a time
    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 normalized float values (4 at a time)
        float32x4_t v_float_0 = vld1q_f32(input_image + i);
        float32x4_t v_float_1 = vld1q_f32(input_image + i + 4);
        float32x4_t v_float_2 = vld1q_f32(input_image + i + 8);
        float32x4_t v_float_3 = vld1q_f32(input_image + i + 12);

        // Scale to [0, 255] and add 0.5 for rounding
        // Uses fused multiply-add for better performance
        v_float_0 = vmlaq_f32(v_0_5, v_float_0, v_255);
        v_float_1 = vmlaq_f32(v_0_5, v_float_1, v_255);
        v_float_2 = vmlaq_f32(v_0_5, v_float_2, v_255);
        v_float_3 = vmlaq_f32(v_0_5, v_float_3, v_255);

        // Convert to integers with saturation
        uint16x4_t v_u16_0 = vqmovun_s32(vcvtq_s32_f32(v_float_0));
        uint16x4_t v_u16_1 = vqmovun_s32(vcvtq_s32_f32(v_float_1));
        uint16x4_t v_u16_2 = vqmovun_s32(vcvtq_s32_f32(v_float_2));
        uint16x4_t v_u16_3 = vqmovun_s32(vcvtq_s32_f32(v_float_3));

        // Combine and narrow to uint8 with saturation
        uint8x8_t v_u8_0 = vqmovn_u16(vcombine_u16(v_u16_0, v_u16_1));
        uint8x8_t v_u8_1 = vqmovn_u16(vcombine_u16(v_u16_2, v_u16_3));

        // Store final results
        vst1q_u8(output_image + i, vcombine_u8(v_u8_0, v_u8_1));
    }
}

/**
 * @brief Helper function to print a sample of image data
 * 
 * Prints the first 16 pixels of an image array, handling both float and uint8 formats.
 * Values are printed in rows of 8 for better readability.
 * 
 * @param label Description of the data being printed
 * @param image Pointer to image data (float* or uint8_t*)
 * @param num_pixels Total number of pixels in the image
 * @param is_float Flag indicating if the data is floating-point (1) or uint8 (0)
 */
void print_image_sample(const char* label, void* image, int num_pixels, int is_float) {
    printf("%s (first 16 pixels):\n", label);
    for (int i = 0; i < 16 && i < num_pixels; i++) {
        if (is_float) {
            printf("%.6f ", ((float*)image)[i]);
        } else {
            printf("%3d ", ((uint8_t*)image)[i]);
        }
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
}

/**
 * @brief Main function demonstrating normalization and denormalization
 * 
 * Creates a test image, applies normalization to convert to float values,
 * then denormalizes back to uint8. Prints samples at each stage to verify
 * the conversion process.
 */
int main() {
    // Test image dimensions
    int width = 640;
    int height = 480;
    int num_pixels = width * height;

    // Allocate memory for all image buffers
    uint8_t* original_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    float* normalized_image = (float*)malloc(num_pixels * sizeof(float));
    uint8_t* denormalized_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (!original_image || !normalized_image || !denormalized_image) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize original image with test pattern
    for (int i = 0; i < num_pixels; i++) {
        original_image[i] = (uint8_t)(i % 256);
    }

    // Display original values
    print_image_sample("Original Image", original_image, num_pixels, 0);

    // Convert to normalized float values
    normalize_image_neon(original_image, normalized_image, num_pixels);
    print_image_sample("Normalized Image", normalized_image, num_pixels, 1);

    // Convert back to uint8 values
    denormalize_image_neon(normalized_image, denormalized_image, num_pixels);
    print_image_sample("Denormalized Image", denormalized_image, num_pixels, 0);

    // Compare original and denormalized images
    int max_diff = 0;
    for (int i = 0; i < num_pixels; i++) {
        int diff = abs(original_image[i] - denormalized_image[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between original and denormalized: %d\n", max_diff);

    // Clean up
    free(original_image);
    free(normalized_image);
    free(denormalized_image);

    return 0;
}