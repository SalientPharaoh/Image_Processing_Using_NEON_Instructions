#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void normalize_image_neon(uint8_t* input_image, float* output_image, int width, int height) {
    int num_pixels = width * height;
    float32x4_t v_recip_255 = vdupq_n_f32(1.0f / 255.0f);

    for (int i = 0; i < num_pixels; i += 16) {
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        // Convert low 8 pixels to float32
        uint16x8_t v_pixels_low = vmovl_u8(vget_low_u8(v_pixels));
        float32x4_t v_float_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_low)));
        float32x4_t v_float_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_low)));

        // Convert high 8 pixels to float32
        uint16x8_t v_pixels_high = vmovl_u8(vget_high_u8(v_pixels));
        float32x4_t v_float_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_high)));
        float32x4_t v_float_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_high)));

        // Normalize by multiplying with 1/255
        v_float_0 = vmulq_f32(v_float_0, v_recip_255);
        v_float_1 = vmulq_f32(v_float_1, v_recip_255);
        v_float_2 = vmulq_f32(v_float_2, v_recip_255);
        v_float_3 = vmulq_f32(v_float_3, v_recip_255);

        // Store results
        vst1q_f32(output_image + i, v_float_0);
        vst1q_f32(output_image + i + 4, v_float_1);
        vst1q_f32(output_image + i + 8, v_float_2);
        vst1q_f32(output_image + i + 12, v_float_3);
    }
}

int main() {
    int width = 640;
    int height = 480;
    int num_pixels = width * height;

    uint8_t* input_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    float* output_image = (float*)malloc(num_pixels * sizeof(float));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    normalize_image_neon(input_image, output_image, width, height);

    printf("First 16 pixels after normalization:\n");
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %d, Output = %.6f\n", i, input_image[i], output_image[i]);
    }

    free(input_image);
    free(output_image);

    return 0;
}