#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void normalize_image_neon(uint8_t* input_image, float* output_image, int num_pixels) {
    float32x4_t v_recip_255 = vdupq_n_f32(1.0f / 255.0f);

    for (int i = 0; i < num_pixels; i += 16) {
        uint8x16_t v_pixels = vld1q_u8(input_image + i);

        uint16x8_t v_pixels_low = vmovl_u8(vget_low_u8(v_pixels));
        uint16x8_t v_pixels_high = vmovl_u8(vget_high_u8(v_pixels));

        float32x4_t v_float_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_low)));
        float32x4_t v_float_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_low)));
        float32x4_t v_float_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_pixels_high)));
        float32x4_t v_float_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_pixels_high)));

        v_float_0 = vmulq_f32(v_float_0, v_recip_255);
        v_float_1 = vmulq_f32(v_float_1, v_recip_255);
        v_float_2 = vmulq_f32(v_float_2, v_recip_255);
        v_float_3 = vmulq_f32(v_float_3, v_recip_255);

        vst1q_f32(output_image + i, v_float_0);
        vst1q_f32(output_image + i + 4, v_float_1);
        vst1q_f32(output_image + i + 8, v_float_2);
        vst1q_f32(output_image + i + 12, v_float_3);
    }
}

void denormalize_image_neon(float* input_image, uint8_t* output_image, int num_pixels) {
    float32x4_t v_255 = vdupq_n_f32(255.0f);
    float32x4_t v_0_5 = vdupq_n_f32(0.5f);

    for (int i = 0; i < num_pixels; i += 16) {
        float32x4_t v_float_0 = vld1q_f32(input_image + i);
        float32x4_t v_float_1 = vld1q_f32(input_image + i + 4);
        float32x4_t v_float_2 = vld1q_f32(input_image + i + 8);
        float32x4_t v_float_3 = vld1q_f32(input_image + i + 12);

        v_float_0 = vmlaq_f32(v_0_5, v_float_0, v_255);
        v_float_1 = vmlaq_f32(v_0_5, v_float_1, v_255);
        v_float_2 = vmlaq_f32(v_0_5, v_float_2, v_255);
        v_float_3 = vmlaq_f32(v_0_5, v_float_3, v_255);

        uint16x4_t v_u16_0 = vqmovun_s32(vcvtq_s32_f32(v_float_0));
        uint16x4_t v_u16_1 = vqmovun_s32(vcvtq_s32_f32(v_float_1));
        uint16x4_t v_u16_2 = vqmovun_s32(vcvtq_s32_f32(v_float_2));
        uint16x4_t v_u16_3 = vqmovun_s32(vcvtq_s32_f32(v_float_3));

        uint8x8_t v_u8_0 = vqmovn_u16(vcombine_u16(v_u16_0, v_u16_1));
        uint8x8_t v_u8_1 = vqmovn_u16(vcombine_u16(v_u16_2, v_u16_3));

        vst1q_u8(output_image + i, vcombine_u8(v_u8_0, v_u8_1));
    }
}

void print_image_sample(const char* label, void* image, int num_pixels, int is_float) {
    printf("%s (first 16 pixels):\n", label);
    for (int i = 0; i < 16 && i < num_pixels; i++) {
        if (is_float) {
            printf("%.6f ", ((float*)image)[i]);
        } else {
            printf("%3d ", ((uint8_t*)image)[i]);
        }
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
}

int main() {
    int width = 640;
    int height = 480;
    int num_pixels = width * height;

    uint8_t* original_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));
    float* normalized_image = (float*)malloc(num_pixels * sizeof(float));
    uint8_t* denormalized_image = (uint8_t*)malloc(num_pixels * sizeof(uint8_t));

    if (!original_image || !normalized_image || !denormalized_image) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize original image with sample data
    for (int i = 0; i < num_pixels; i++) {
        original_image[i] = (uint8_t)(i % 256);
    }

    // Print sample of original image
    print_image_sample("Original Image", original_image, num_pixels, 0);

    // Normalize the image
    normalize_image_neon(original_image, normalized_image, num_pixels);

    // Print sample of normalized image
    print_image_sample("Normalized Image", normalized_image, num_pixels, 1);

    // Denormalize the image
    denormalize_image_neon(normalized_image, denormalized_image, num_pixels);

    // Print sample of denormalized image
    print_image_sample("Denormalized Image", denormalized_image, num_pixels, 0);

    // Check if denormalized image matches the original
    int mismatch_count = 0;
    for (int i = 0; i < num_pixels; i++) {
        if (original_image[i] != denormalized_image[i]) {
            mismatch_count++;
        }
    }

    printf("Number of mismatches between original and denormalized: %d out of %d pixels\n", 
           mismatch_count, num_pixels);

    free(original_image);
    free(normalized_image);
    free(denormalized_image);

    return 0;
}