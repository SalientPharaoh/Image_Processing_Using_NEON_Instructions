#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>

void multiply_image_neon(uint8_t* input_image, uint8_t* output_image, int width, int height, float factor) {
    int num_pixels = width * height;
    float32x4_t v_factor = vdupq_n_f32(factor);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_255 = vdupq_n_f32(255.0f);

    for (int i = 0; i < num_pixels; i += 16) {
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        uint16x8_t v_pixels_lo = vmovl_u8(vget_low_u8(v_pixels));
        uint16x8_t v_pixels_hi = vmovl_u8(vget_high_u8(v_pixels));
        
        float32x4_t v_pixels_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_lo)));
        float32x4_t v_pixels_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_lo)));
        float32x4_t v_pixels_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_hi)));
        float32x4_t v_pixels_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_hi)));

        v_pixels_f32_0 = vmulq_f32(v_pixels_f32_0, v_factor);
        v_pixels_f32_1 = vmulq_f32(v_pixels_f32_1, v_factor);
        v_pixels_f32_2 = vmulq_f32(v_pixels_f32_2, v_factor);
        v_pixels_f32_3 = vmulq_f32(v_pixels_f32_3, v_factor);

        // Clamp values between 0 and 255
        v_pixels_f32_0 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_0));
        v_pixels_f32_1 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_1));
        v_pixels_f32_2 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_2));
        v_pixels_f32_3 = vmaxq_f32(v_zero, vminq_f32(v_255, v_pixels_f32_3));

        // Convert back to uint8_t
        uint32x4_t v_pixels_u32_0 = vcvtq_u32_f32(v_pixels_f32_0);
        uint32x4_t v_pixels_u32_1 = vcvtq_u32_f32(v_pixels_f32_1);
        uint32x4_t v_pixels_u32_2 = vcvtq_u32_f32(v_pixels_f32_2);
        uint32x4_t v_pixels_u32_3 = vcvtq_u32_f32(v_pixels_f32_3);

        uint16x4_t v_pixels_u16_0 = vmovn_u32(v_pixels_u32_0);
        uint16x4_t v_pixels_u16_1 = vmovn_u32(v_pixels_u32_1);
        uint16x4_t v_pixels_u16_2 = vmovn_u32(v_pixels_u32_2);
        uint16x4_t v_pixels_u16_3 = vmovn_u32(v_pixels_u32_3);

        uint16x8_t v_pixels_u16_lo = vcombine_u16(v_pixels_u16_0, v_pixels_u16_1);
        uint16x8_t v_pixels_u16_hi = vcombine_u16(v_pixels_u16_2, v_pixels_u16_3);

        uint8x16_t v_result = vcombine_u8(vmovn_u16(v_pixels_u16_lo), vmovn_u16(v_pixels_u16_hi));

        vst1q_u8(output_image + i, v_result);
    }
}

int main() {
    int width = 640;
    int height = 480;
    int num_pixels = width * height;
    float factor = 2.0f;

    uint8_t* input_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    uint8_t* output_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    multiply_image_neon(input_image, output_image, width, height, factor);

    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %d, Output = %d\n", i, input_image[i], output_image[i]);
    }

    free(input_image);
    free(output_image);

    return 0;
}