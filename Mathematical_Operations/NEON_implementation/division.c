#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void divide_image_neon(uint8_t* input_image, uint8_t* output_image, int width, int height, float divisor) {
    int num_pixels = width * height;
    uint32_t fixed_point_multiplier = (uint32_t)((1 << 24) / divisor + 0.5f);
    uint32x4_t v_multiplier = vdupq_n_u32(fixed_point_multiplier);

    for (int i = 0; i < num_pixels; i += 16) {
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        // Convert to 16-bit
        uint16x8_t v_pixels_low = vmovl_u8(vget_low_u8(v_pixels));
        uint16x8_t v_pixels_high = vmovl_u8(vget_high_u8(v_pixels));

        // Convert to 32-bit
        uint32x4_t v_pixels_0 = vmovl_u16(vget_low_u16(v_pixels_low));
        uint32x4_t v_pixels_1 = vmovl_u16(vget_high_u16(v_pixels_low));
        uint32x4_t v_pixels_2 = vmovl_u16(vget_low_u16(v_pixels_high));
        uint32x4_t v_pixels_3 = vmovl_u16(vget_high_u16(v_pixels_high));

        // Multiply by fixed-point multiplier
        v_pixels_0 = vmulq_u32(v_pixels_0, v_multiplier);
        v_pixels_1 = vmulq_u32(v_pixels_1, v_multiplier);
        v_pixels_2 = vmulq_u32(v_pixels_2, v_multiplier);
        v_pixels_3 = vmulq_u32(v_pixels_3, v_multiplier);

        // Shift right by 24 to get the result
        v_pixels_0 = vshrq_n_u32(v_pixels_0, 24);
        v_pixels_1 = vshrq_n_u32(v_pixels_1, 24);
        v_pixels_2 = vshrq_n_u32(v_pixels_2, 24);
        v_pixels_3 = vshrq_n_u32(v_pixels_3, 24);

        // Narrow to 16-bit
        uint16x8_t v_result_low = vcombine_u16(vmovn_u32(v_pixels_0), vmovn_u32(v_pixels_1));
        uint16x8_t v_result_high = vcombine_u16(vmovn_u32(v_pixels_2), vmovn_u32(v_pixels_3));

        // Narrow to 8-bit (with saturation to handle potential overflow)
        uint8x16_t v_result = vcombine_u8(vqmovn_u16(v_result_low), vqmovn_u16(v_result_high));

        // Store the result
        vst1q_u8(output_image + i, v_result);
    }
}

int main() {
    int width = 640;
    int height = 480;
    int num_pixels = width * height;
    float divisor = 2.0f;

    uint8_t* input_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    uint8_t* output_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    divide_image_neon(input_image, output_image, width, height, divisor);

    printf("First 16 pixels after division:\n");
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %d, Output = %d\n", i, input_image[i], output_image[i]);
    }

    free(input_image);
    free(output_image);

    return 0;
}