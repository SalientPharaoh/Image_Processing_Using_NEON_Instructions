#include <arm_neon.h>
#include <stdio.h>

void denormalize_image_neon(float* input_image, uint8_t* output_image, int width, int height) {
    int num_pixels = width * height;
    float32x4_t v_max_val = vdupq_n_f32(255.0f);

    for (int i = 0; i < num_pixels; i += 16) {
        float32x4_t v_pixels_f32_0 = vld1q_f32(input_image + i);
        float32x4_t v_pixels_f32_1 = vld1q_f32(input_image + i + 4);
        float32x4_t v_pixels_f32_2 = vld1q_f32(input_image + i + 8);
        float32x4_t v_pixels_f32_3 = vld1q_f32(input_image + i + 12);

        v_pixels_f32_0 = vmulq_f32(v_pixels_f32_0, v_max_val);
        v_pixels_f32_1 = vmulq_f32(v_pixels_f32_1, v_max_val);
        v_pixels_f32_2 = vmulq_f32(v_pixels_f32_2, v_max_val);
        v_pixels_f32_3 = vmulq_f32(v_pixels_f32_3, v_max_val);

        uint16x4_t v_pixels_u16_0 = vqmovun_s32(vcvtq_s32_f32(v_pixels_f32_0));
        uint16x4_t v_pixels_u16_1 = vqmovun_s32(vcvtq_s32_f32(v_pixels_f32_1));
        uint16x4_t v_pixels_u16_2 = vqmovun_s32(vcvtq_s32_f32(v_pixels_f32_2));
        uint16x4_t v_pixels_u16_3 = vqmovun_s32(vcvtq_s32_f32(v_pixels_f32_3));

        uint8x8_t v_pixels_u8_0 = vqmovun_s16(vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_pixels_f32_0)), vmovn_s32(vcvtq_s32_f32(v_pixels_f32_1))));
        uint8x8_t v_pixels_u8_1 = vqmovun_s16(vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_pixels_f32_2)), vmovn_s32(vcvtq_s32_f32(v_pixels_f32_3))));

        vst1q_u8(output_image + i, vcombine_u8(v_pixels_u8_0, v_pixels_u8_1));
    }
}

int main() {
    int width = 640; 
    int height = 480; 
    int num_pixels = width * height;

    uint8_t input_image[num_pixels];
    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    float output_image[num_pixels];
    denormalize_image_neon(input_image, output_image, width, height);

    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: %f\n", i, output_image[i]);
    }

    return 0;
}
