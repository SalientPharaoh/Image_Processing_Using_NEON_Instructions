#include <arm_neon.h>
#include <stdio.h>

void divide_image_neon(uint8_t* input_image, float* output_image, int width, int height, float divisor) {
    int num_pixels = width * height;
    float32x4_t v_recip_divisor = vdupq_n_f32(1.0f / divisor);

    for (int i = 0; i < num_pixels; i += 16) {
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        uint16x8_t v_pixels_lo = vmovl_u8(vget_low_u8(v_pixels));
        uint16x8_t v_pixels_hi = vmovl_u8(vget_high_u8(v_pixels));
        float32x4_t v_pixels_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_lo)));
        float32x4_t v_pixels_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_lo)));
        float32x4_t v_pixels_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_hi)));
        float32x4_t v_pixels_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_hi)));

        v_pixels_f32_0 = vmulq_f32(v_pixels_f32_0, v_recip_divisor);
        v_pixels_f32_1 = vmulq_f32(v_pixels_f32_1, v_recip_divisor);
        v_pixels_f32_2 = vmulq_f32(v_pixels_f32_2, v_recip_divisor);
        v_pixels_f32_3 = vmulq_f32(v_pixels_f32_3, v_recip_divisor);

        vst1q_f32(output_image + i, v_pixels_f32_0);
        vst1q_f32(output_image + i + 4, v_pixels_f32_1);
        vst1q_f32(output_image + i + 8, v_pixels_f32_2);
        vst1q_f32(output_image + i + 12, v_pixels_f32_3);
    }
}

int main() {
    int width = 640; 
    int height = 480; 
    int num_pixels = width * height;
    int factor = 2;

    uint8_t input_image[num_pixels];
    for (int i = 0; i < num_pixels; i++) {
        input_image[i] = (uint8_t)(i % 256);
    }

    float output_image[num_pixels];
    divide_image_neon(input_image, output_image, width, height, factor);

    for (int i = 0; i < 16; i++) {
        printf("Pixel %d: %f\n", i, output_image[i]);
    }

    return 0;
}
