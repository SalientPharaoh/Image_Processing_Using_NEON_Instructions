/**
 * @file division.c
 * @brief NEON-optimized image division implementation using fixed-point arithmetic
 * 
 * This file implements fast image division using ARM NEON SIMD instructions and
 * fixed-point arithmetic. Instead of using floating-point division, which is slow,
 * it uses multiplication by a fixed-point reciprocal for better performance.
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * @brief Divide an 8-bit image by a constant using NEON and fixed-point arithmetic
 * 
 * This function divides each pixel value by a constant divisor using NEON SIMD
 * instructions and fixed-point arithmetic for improved performance. The function
 * processes 16 pixels at a time.
 * 
 * Algorithm steps:
 * 1. Convert division to multiplication by reciprocal using fixed-point arithmetic
 * 2. Load 16 uint8 values using NEON registers
 * 3. Progressively widen values for multiplication
 * 4. Multiply by fixed-point reciprocal
 * 5. Shift right to get final result
 * 6. Narrow results back to uint8 with saturation
 * 
 * @param input_image Input uint8 image array
 * @param output_image Output uint8 image array
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param divisor Floating-point divisor value
 */
void divide_image_neon(uint8_t* input_image, uint8_t* output_image, int width, int height, float divisor) {
    int num_pixels = width * height;
    
    // Convert division to multiplication by computing fixed-point reciprocal
    // Use 24-bit fixed-point format for better precision
    // fixed_point_multiplier = (1 << 24) / divisor, rounded to nearest
    uint32_t fixed_point_multiplier = (uint32_t)((1 << 24) / divisor + 0.5f);
    uint32x4_t v_multiplier = vdupq_n_u32(fixed_point_multiplier);

    // Process 16 pixels at a time using NEON
    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 uint8 pixels into a NEON register
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        // Widen pixels from uint8 to uint16 for intermediate calculations
        uint16x8_t v_pixels_low = vmovl_u8(vget_low_u8(v_pixels));    // Lower 8 pixels
        uint16x8_t v_pixels_high = vmovl_u8(vget_high_u8(v_pixels));  // Upper 8 pixels

        // Further widen to uint32 for multiplication
        uint32x4_t v_pixels_0 = vmovl_u16(vget_low_u16(v_pixels_low));   // First 4 pixels
        uint32x4_t v_pixels_1 = vmovl_u16(vget_high_u16(v_pixels_low));  // Next 4 pixels
        uint32x4_t v_pixels_2 = vmovl_u16(vget_low_u16(v_pixels_high));  // Next 4 pixels
        uint32x4_t v_pixels_3 = vmovl_u16(vget_high_u16(v_pixels_high)); // Last 4 pixels

        // Multiply by fixed-point reciprocal
        // This effectively divides by the original divisor
        v_pixels_0 = vmulq_u32(v_pixels_0, v_multiplier);
        v_pixels_1 = vmulq_u32(v_pixels_1, v_multiplier);
        v_pixels_2 = vmulq_u32(v_pixels_2, v_multiplier);
        v_pixels_3 = vmulq_u32(v_pixels_3, v_multiplier);

        // Shift right by 24 to convert fixed-point result back to integer
        v_pixels_0 = vshrq_n_u32(v_pixels_0, 24);
        v_pixels_1 = vshrq_n_u32(v_pixels_1, 24);
        v_pixels_2 = vshrq_n_u32(v_pixels_2, 24);
        v_pixels_3 = vshrq_n_u32(v_pixels_3, 24);

        // Narrow results back to uint16
        uint16x8_t v_result_low = vcombine_u16(vmovn_u32(v_pixels_0), vmovn_u32(v_pixels_1));
        uint16x8_t v_result_high = vcombine_u16(vmovn_u32(v_pixels_2), vmovn_u32(v_pixels_3));

        // Final narrow to uint8 with saturation to prevent overflow
        uint8x16_t v_result = vcombine_u8(vqmovn_u16(v_result_low), vqmovn_u16(v_result_high));

        // Store final results back to memory
        vst1q_u8(output_image + i, v_result);
    }
}

/**
 * @brief Main function demonstrating the usage of divide_image_neon
 * 
 * Creates a test image with uint8 values, divides all pixels by 2.0,
 * and prints the first 16 pixels before and after division.
 */
int main() {
    // Test image dimensions
    int width = 640;
    int height = 480;
    int num_pixels = width * height;
    float divisor = 2.0f;  // Divide all pixels by 2

    // Allocate memory for input and output images
    uint8_t* input_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    uint8_t* output_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize input with test pattern (0-255 values)
    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    // Apply division
    divide_image_neon(input_image, output_image, width, height, divisor);

    // Print results for verification
    printf("First 16 pixels after division:\n");
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %d, Output = %d\n", i, input_image[i], output_image[i]);
    }

    // Clean up
    free(input_image);
    free(output_image);

    return 0;
}