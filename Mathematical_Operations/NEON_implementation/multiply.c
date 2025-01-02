/**
 * @file multiply.c
 * @brief NEON-optimized image multiplication implementation
 * 
 * This file implements fast image multiplication using ARM NEON SIMD instructions.
 * It multiplies each pixel value by a constant factor, with proper clamping to
 * ensure results stay within the valid uint8 range [0, 255].
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>

/**
 * @brief Multiply an 8-bit image by a constant factor using NEON
 * 
 * This function multiplies each pixel value by a constant factor using NEON SIMD
 * instructions for improved performance. The function processes 16 pixels at a time
 * and includes proper clamping to ensure results stay within [0, 255].
 * 
 * Algorithm steps:
 * 1. Load 16 uint8 values using NEON registers
 * 2. Convert to float32 for multiplication
 * 3. Multiply by factor
 * 4. Clamp results to [0, 255]
 * 5. Convert back to uint8 with proper rounding
 * 
 * @param input_image Input uint8 image array
 * @param output_image Output uint8 image array
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param factor Floating-point multiplication factor
 */
void multiply_image_neon(uint8_t* input_image, uint8_t* output_image, int width, int height, float factor) {
    int num_pixels = width * height;
    
    // Initialize constant vectors for multiplication and clamping
    float32x4_t v_factor = vdupq_n_f32(factor);   // Vector of multiplication factor
    float32x4_t v_zero = vdupq_n_f32(0.0f);       // Vector of zeros for clamping
    float32x4_t v_255 = vdupq_n_f32(255.0f);      // Vector of 255 for clamping

    // Process 16 pixels at a time using NEON
    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 uint8 pixels into a NEON register
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        // Widen pixels from uint8 to uint16 for intermediate calculations
        uint16x8_t v_pixels_lo = vmovl_u8(vget_low_u8(v_pixels));   // Lower 8 pixels
        uint16x8_t v_pixels_hi = vmovl_u8(vget_high_u8(v_pixels));  // Upper 8 pixels
        
        // Convert to float32 for multiplication
        // This requires two steps: uint16->uint32->float32
        float32x4_t v_pixels_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_lo)));   // First 4 pixels
        float32x4_t v_pixels_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_lo)));  // Next 4 pixels
        float32x4_t v_pixels_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_hi)));   // Next 4 pixels
        float32x4_t v_pixels_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_hi)));  // Last 4 pixels

        // Multiply each vector by the factor
        v_pixels_f32_0 = vmulq_f32(v_pixels_f32_0, v_factor);
        v_pixels_f32_1 = vmulq_f32(v_pixels_f32_1, v_factor);
        v_pixels_f32_2 = vmulq_f32(v_pixels_f32_2, v_factor);
        v_pixels_f32_3 = vmulq_f32(v_pixels_f32_3, v_factor);

        // Clamp values between 0 and 255 using min/max operations
        v_pixels_f32_0 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_0));
        v_pixels_f32_1 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_1));
        v_pixels_f32_2 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_2));
        v_pixels_f32_3 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_3));

        // Convert back to uint32
        uint32x4_t v_pixels_u32_0 = vcvtq_u32_f32(v_pixels_f32_0);
        uint32x4_t v_pixels_u32_1 = vcvtq_u32_f32(v_pixels_f32_1);
        uint32x4_t v_pixels_u32_2 = vcvtq_u32_f32(v_pixels_f32_2);
        uint32x4_t v_pixels_u32_3 = vcvtq_u32_f32(v_pixels_f32_3);

        // Narrow from uint32 to uint16
        uint16x4_t v_pixels_u16_0 = vmovn_u32(v_pixels_u32_0);
        uint16x4_t v_pixels_u16_1 = vmovn_u32(v_pixels_u32_1);
        uint16x4_t v_pixels_u16_2 = vmovn_u32(v_pixels_u32_2);
        uint16x4_t v_pixels_u16_3 = vmovn_u32(v_pixels_u32_3);

        // Combine uint16 vectors
        uint16x8_t v_pixels_u16_lo = vcombine_u16(v_pixels_u16_0, v_pixels_u16_1);
        uint16x8_t v_pixels_u16_hi = vcombine_u16(v_pixels_u16_2, v_pixels_u16_3);

        // Final narrow to uint8 and combine results
        uint8x16_t v_result = vcombine_u8(vmovn_u16(v_pixels_u16_lo), vmovn_u16(v_pixels_u16_hi));

        // Store final results back to memory
        vst1q_u8(output_image + i, v_result);
    }
}

/**
 * @brief Main function demonstrating the usage of multiply_image_neon
 * 
 * Creates a test image with uint8 values, multiplies all pixels by 2.0,
 * and prints the first 16 pixels before and after multiplication.
 */
int main() {
    // Test image dimensions
    int width = 640;
    int height = 480;
    int num_pixels = width * height;
    float factor = 2.0f;  // Multiply all pixels by 2

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

    // Apply multiplication
    multiply_image_neon(input_image, output_image, width, height, factor);

    // Print results for verification
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %d, Output = %d\n", i, input_image[i], output_image[i]);
    }

    // Clean up
    free(input_image);
    free(output_image);

    return 0;
}