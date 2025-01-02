/**
 * @file denorm.c
 * @brief NEON-optimized image denormalization implementation
 * 
 * This file implements fast image denormalization using ARM NEON SIMD instructions.
 * It converts normalized floating-point values [0.0, 1.0] to 8-bit unsigned integers [0, 255].
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * @brief Denormalize a floating-point image to 8-bit unsigned integers using NEON
 * 
 * This function converts normalized floating-point values in range [0.0, 1.0] to
 * 8-bit unsigned integers in range [0, 255] using NEON SIMD instructions for
 * improved performance. The function processes 16 pixels at a time.
 * 
 * Algorithm steps:
 * 1. Load 16 float values using NEON registers
 * 2. Multiply by 255 and add 0.5 for proper rounding
 * 3. Convert to integers with saturation to prevent overflow
 * 4. Narrow the values to uint8 format
 * 
 * @param input_image Input floating-point image array (normalized values)
 * @param output_image Output uint8 image array
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void denormalize_image_neon(float* input_image, uint8_t* output_image, int width, int height) {
    int num_pixels = width * height;
    // Create vectors with constant values for multiplication and rounding
    float32x4_t v_255 = vdupq_n_f32(255.0f);  // Vector of 255.0
    float32x4_t v_0_5 = vdupq_n_f32(0.5f);    // Vector of 0.5 for rounding

    // Process 16 pixels at a time using NEON
    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 float values (4 vectors of 4 floats each)
        float32x4_t v_float_0 = vld1q_f32(input_image + i);
        float32x4_t v_float_1 = vld1q_f32(input_image + i + 4);
        float32x4_t v_float_2 = vld1q_f32(input_image + i + 8);
        float32x4_t v_float_3 = vld1q_f32(input_image + i + 12);

        // Multiply by 255 and add 0.5 for rounding
        // Uses fused multiply-add for better performance
        v_float_0 = vmlaq_f32(v_0_5, v_float_0, v_255);
        v_float_1 = vmlaq_f32(v_0_5, v_float_1, v_255);
        v_float_2 = vmlaq_f32(v_0_5, v_float_2, v_255);
        v_float_3 = vmlaq_f32(v_0_5, v_float_3, v_255);

        // Convert to integers with saturation
        // First convert to signed 32-bit integers, then narrow to unsigned 16-bit
        uint16x4_t v_u16_0 = vqmovun_s32(vcvtq_s32_f32(v_float_0));
        uint16x4_t v_u16_1 = vqmovun_s32(vcvtq_s32_f32(v_float_1));
        uint16x4_t v_u16_2 = vqmovun_s32(vcvtq_s32_f32(v_float_2));
        uint16x4_t v_u16_3 = vqmovun_s32(vcvtq_s32_f32(v_float_3));

        // Combine pairs of vectors and narrow to uint8
        uint8x8_t v_u8_0 = vqmovn_u16(vcombine_u16(v_u16_0, v_u16_1));
        uint8x8_t v_u8_1 = vqmovn_u16(vcombine_u16(v_u16_2, v_u16_3));

        // Store the final result as 16 uint8 values
        vst1q_u8(output_image + i, vcombine_u8(v_u8_0, v_u8_1));
    }
}

/**
 * @brief Main function demonstrating the usage of denormalize_image_neon
 * 
 * Creates a test image with normalized values, applies the denormalization,
 * and prints the first 16 pixels before and after conversion.
 */
int main() {
    // Test image dimensions
    int width = 640;
    int height = 480;
    int num_pixels = width * height;

    // Allocate memory for input and output images
    float* input_image = (float*)malloc(num_pixels * sizeof(float));
    uint8_t* output_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize input with test pattern (normalized values)
    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (float)(i % 256) / 255.0f;
    }

    // Apply denormalization
    denormalize_image_neon(input_image, output_image, width, height);

    // Print results for verification
    printf("First 16 pixels after denormalization:\n");
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %.6f, Output = %d\n", i, input_image[i], output_image[i]);
    }

    // Clean up
    free(input_image);
    free(output_image);

    return 0;
}