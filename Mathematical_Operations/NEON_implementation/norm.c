/**
 * @file norm.c
 * @brief NEON-optimized image normalization implementation
 * 
 * This file implements fast image normalization using ARM NEON SIMD instructions.
 * It converts 8-bit unsigned integers [0, 255] to normalized floating-point values [0.0, 1.0].
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * @brief Normalize an 8-bit image to floating-point values using NEON
 * 
 * This function converts 8-bit unsigned integers in range [0, 255] to
 * normalized floating-point values in range [0.0, 1.0] using NEON SIMD
 * instructions for improved performance. The function processes 16 pixels at a time.
 * 
 * Algorithm steps:
 * 1. Load 16 uint8 values using NEON registers
 * 2. Convert uint8 to float32 through progressive widening
 * 3. Multiply by 1/255 to normalize to [0.0, 1.0] range
 * 
 * @param input_image Input uint8 image array
 * @param output_image Output float image array (normalized values)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void normalize_image_neon(uint8_t* input_image, float* output_image, int width, int height) {
    int num_pixels = width * height;
    // Create vector with reciprocal of 255 for normalization
    float32x4_t v_recip_255 = vdupq_n_f32(1.0f / 255.0f);

    // Process 16 pixels at a time using NEON
    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 uint8 pixels into a NEON register
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        // Convert first 8 pixels from uint8 to float32
        // This requires multiple steps due to progressive widening:
        // uint8 -> uint16 -> uint32 -> float32
        uint16x8_t v_pixels_low = vmovl_u8(vget_low_u8(v_pixels));  // Widen lower 8 pixels to uint16
        float32x4_t v_float_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_low)));   // Convert first 4 to float
        float32x4_t v_float_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_low)));  // Convert next 4 to float

        // Convert second 8 pixels from uint8 to float32
        uint16x8_t v_pixels_high = vmovl_u8(vget_high_u8(v_pixels));  // Widen upper 8 pixels to uint16
        float32x4_t v_float_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_high)));  // Convert next 4 to float
        float32x4_t v_float_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_high))); // Convert last 4 to float

        // Normalize to [0.0, 1.0] range by multiplying with 1/255
        v_float_0 = vmulq_f32(v_float_0, v_recip_255);
        v_float_1 = vmulq_f32(v_float_1, v_recip_255);
        v_float_2 = vmulq_f32(v_float_2, v_recip_255);
        v_float_3 = vmulq_f32(v_float_3, v_recip_255);

        // Store normalized results back to memory
        vst1q_f32(output_image + i, v_float_0);
        vst1q_f32(output_image + i + 4, v_float_1);
        vst1q_f32(output_image + i + 8, v_float_2);
        vst1q_f32(output_image + i + 12, v_float_3);
    }
}

/**
 * @brief Main function demonstrating the usage of normalize_image_neon
 * 
 * Creates a test image with uint8 values, applies the normalization,
 * and prints the first 16 pixels before and after conversion.
 */
int main() {
    // Test image dimensions
    int width = 640;
    int height = 480;
    int num_pixels = width * height;

    // Allocate memory for input and output images
    uint8_t* input_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    float* output_image = (float*)malloc(num_pixels * sizeof(float));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize input with test pattern (0-255 values)
    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    // Apply normalization
    normalize_image_neon(input_image, output_image, width, height);

    // Print results for verification
    printf("First 16 pixels after normalization:\n");
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %d, Output = %.6f\n", i, input_image[i], output_image[i]);
    }

    // Clean up
    free(input_image);
    free(output_image);

    return 0;
}