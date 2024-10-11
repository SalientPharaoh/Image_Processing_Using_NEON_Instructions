#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void denormalize_image_neon(float* input_image, uint8_t* output_image, int width, int height) {
    int num_pixels = width * height;
    float32x4_t v_255 = vdupq_n_f32(255.0f);
    float32x4_t v_0_5 = vdupq_n_f32(0.5f); // For rounding

    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 float values
        float32x4_t v_float_0 = vld1q_f32(input_image + i);
        float32x4_t v_float_1 = vld1q_f32(input_image + i + 4);
        float32x4_t v_float_2 = vld1q_f32(input_image + i + 8);
        float32x4_t v_float_3 = vld1q_f32(input_image + i + 12);

        // Multiply by 255 and add 0.5 for rounding
        v_float_0 = vmlaq_f32(v_0_5, v_float_0, v_255);
        v_float_1 = vmlaq_f32(v_0_5, v_float_1, v_255);
        v_float_2 = vmlaq_f32(v_0_5, v_float_2, v_255);
        v_float_3 = vmlaq_f32(v_0_5, v_float_3, v_255);

        // Convert to integers with saturation
        uint16x4_t v_u16_0 = vqmovun_s32(vcvtq_s32_f32(v_float_0));
        uint16x4_t v_u16_1 = vqmovun_s32(vcvtq_s32_f32(v_float_1));
        uint16x4_t v_u16_2 = vqmovun_s32(vcvtq_s32_f32(v_float_2));
        uint16x4_t v_u16_3 = vqmovun_s32(vcvtq_s32_f32(v_float_3));

        // Combine and narrow to uint8
        uint8x8_t v_u8_0 = vqmovn_u16(vcombine_u16(v_u16_0, v_u16_1));
        uint8x8_t v_u8_1 = vqmovn_u16(vcombine_u16(v_u16_2, v_u16_3));

        // Store the result
        vst1q_u8(output_image + i, vcombine_u8(v_u8_0, v_u8_1));
    }
}

int main() {
    int width = 640;
    int height = 480;
    int num_pixels = width * height;

    float* input_image = (float*)malloc(num_pixels * sizeof(float));
    uint8_t* output_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (input_image == NULL || output_image == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize input with normalized values
    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (float)(i % 256) / 255.0f;
    }

    denormalize_image_neon(input_image, output_image, width, height);

    printf("First 16 pixels after denormalization:\n");
    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: Input = %.6f, Output = %d\n", i, input_image[i], output_image[i]);
    }

    free(input_image);
    free(output_image);

    return 0;
}