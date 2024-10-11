#include <arm_neon.h>
#include <stdio.h>
#include <math.h>

#define MAX_GRAY_LEVEL 256

// Function to convert RGB image to YCbCr color space
void RGB_to_YCbCr_neon(uint8_t *rgb_image, uint8_t *Y_channel, uint8_t *Cb_channel, uint8_t *Cr_channel, int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; i += 8) {
        // Load 8 RGB pixels
        uint8x8x3_t rgb_neon = vld3_u8(&rgb_image[i * 3]);

        // Convert to YCbCr
        // Y  =  0.299 * R + 0.587 * G + 0.114 * B
        // Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
        // Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128

        // Convert uint8 to float32
        float32x4_t R_low = vcvtq_f32_u32(vmovl_u16(vget_low_u8(rgb_neon.val[0])));
        float32x4_t G_low = vcvtq_f32_u32(vmovl_u16(vget_low_u8(rgb_neon.val[1])));
        float32x4_t B_low = vcvtq_f32_u32(vmovl_u16(vget_low_u8(rgb_neon.val[2])));

        float32x4_t R_high = vcvtq_f32_u32(vmovl_u16(vget_high_u8(rgb_neon.val[0])));
        float32x4_t G_high = vcvtq_f32_u32(vmovl_u16(vget_high_u8(rgb_neon.val[1])));
        float32x4_t B_high = vcvtq_f32_u32(vmovl_u16(vget_high_u8(rgb_neon.val[2])));

        // Compute Y channel
        float32x4_t Y_low = vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(R_low, 0.299f), G_low, 0.587f), B_low, 0.114f);
        float32x4_t Y_high = vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(R_high, 0.299f), G_high, 0.587f), B_high, 0.114f);

        // Compute Cb channel
        float32x4_t Cb_low = vaddq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(R_low, -0.169f), G_low, -0.331f), B_low, 0.5f), 128.0f);
        float32x4_t Cb_high = vaddq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(R_high, -0.169f), G_high, -0.331f), B_high, 0.5f), 128.0f);

        // Compute Cr channel
        float32x4_t Cr_low = vaddq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(R_low, 0.5f), G_low, -0.419f), B_low, -0.081f), 128.0f);
        float32x4_t Cr_high = vaddq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(R_high, 0.5f), G_high, -0.419f), B_high, -0.081f), 128.0f);

        // Convert float32 back to uint8
        uint8x8_t Y_neon = vcombine_u8(vqmovn_u16(vcvtq_u16_f32(Y_low)), vqmovn_u16(vcvtq_u16_f32(Y_high)));
        uint8x8_t Cb_neon = vcombine_u8(vqmovn_u16(vcvtq_u16_f32(Cb_low)), vqmovn_u16(vcvtq_u16_f32(Cb_high)));
        uint8x8_t Cr_neon = vcombine_u8(vqmovn_u16(vcvtq_u16_f32(Cr_low)), vqmovn_u16(vcvtq_u16_f32(Cr_high)));

        // Store the Y, Cb, Cr channels
        vst1_u8(&Y_channel[i], Y_neon);
        vst1_u8(&Cb_channel[i], Cb_neon);
        vst1_u8(&Cr_channel[i], Cr_neon);
    }
}

// Function to convert YCbCr image back to RGB color space
void YCbCr_to_RGB_neon(uint8_t *Y_channel, uint8_t *Cb_channel, uint8_t *Cr_channel, uint8_t *rgb_image, int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; i += 8) {
        // Load 8 pixels of Y, Cb, Cr channels
        uint8x8_t Y_neon = vld1_u8(&Y_channel[i]);
        uint8x8_t Cb_neon = vld1_u8(&Cb_channel[i]);
        uint8x8_t Cr_neon = vld1_u8(&Cr_channel[i]);

        // Convert uint8 to float32
        float32x4_t Y_low = vcvtq_f32_u32(vmovl_u16(vget_low_u8(Y_neon)));
        float32x4_t Cb_low = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u8(Cb_neon))), vdupq_n_f32(128.0f));
        float32x4_t Cr_low = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u8(Cr_neon))), vdupq_n_f32(128.0f));

        float32x4_t Y_high = vcvtq_f32_u32(vmovl_u16(vget_high_u8(Y_neon)));
        float32x4_t Cb_high = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u8(Cb_neon))), vdupq_n_f32(128.0f));
        float32x4_t Cr_high = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u8(Cr_neon))), vdupq_n_f32(128.0f));

        // Compute R channel
        float32x4_t R_low = vaddq_f32(Y_low, vmulq_n_f32(Cr_low, 1.402f));
        float32x4_t R_high = vaddq_f32(Y_high, vmulq_n_f32(Cr_high, 1.402f));

        // Compute G channel
        float32x4_t G_low = vsubq_f32(vsubq_f32(Y_low, vmulq_n_f32(Cb_low, 0.344136f)), vmulq_n_f32(Cr_low, 0.714136f));
        float32x4_t G_high = vsubq_f32(vsubq_f32(Y_high, vmulq_n_f32(Cb_high, 0.344136f)), vmulq_n_f32(Cr_high, 0.714136f));

        // Compute B channel
        float32x4_t B_low = vaddq_f32(Y_low, vmulq_n_f32(Cb_low, 1.772f));
        float32x4_t B_high = vaddq_f32(Y_high, vmulq_n_f32(Cb_high, 1.772f));

        // Clamp the results to [0, 255]
        uint8x8_t R_neon = vcombine_u8(vqmovun_s16(vcvtq_s16_f32(R_low)), vqmovun_s16(vcvtq_s16_f32(R_high)));
        uint8x8_t G_neon = vcombine_u8(vqmovun_s16(vcvtq_s16_f32(G_low)), vqmovun_s16(vcvtq_s16_f32(G_high)));
        uint8x8_t B_neon = vcombine_u8(vqmovun_s16(vcvtq_s16_f32(B_low)), vqmovun_s16(vcvtq_s16_f32(B_high)));

        // Store the RGB pixels
        uint8x8x3_t rgb_neon;
        rgb_neon.val[0] = R_neon;
        rgb_neon.val[1] = G_neon;
        rgb_neon.val[2] = B_neon;
        vst3_u8(&rgb_image[i * 3], rgb_neon);
    }
}

// Function for Weighted Threshold Histogram Equalization (WTHE) on the Y channel
void WTHE_neon(uint8_t *Y_channel, uint8_t *enhanced_Y_channel, int width, int height, float r) {
    int size = width * height;
    uint32_t histogram[MAX_GRAY_LEVEL] = {0};
    float PDF[MAX_GRAY_LEVEL] = {0.0f};
    float CDF[MAX_GRAY_LEVEL] = {0.0f};
    float total_pixels = (float)size;

    // 1. Compute the Histogram of the Y channel
    for (int i = 0; i < size; i++) {
        histogram[Y_channel[i]]++;
    }

    // 2. Modify the PDF with weighting and thresholding
    float max_P = 0.0f;
    for (int i = 0; i < MAX_GRAY_LEVEL; i++) {
        PDF[i] = (float)histogram[i] / total_pixels;
        if (PDF[i] > max_P) {
            max_P = PDF[i];
        }
    }
    float P_u = max_P * 0.8f;  // Upper threshold (e.g., 80% of max_P)
    for (int i = 0; i < MAX_GRAY_LEVEL; i++) {
        if (PDF[i] > P_u) {
            PDF[i] = P_u;
        }
        PDF[i] = powf(PDF[i], r);  // Apply weighting
    }

    // 3. Compute the CDF
    CDF[0] = PDF[0];
    for (int i = 1; i < MAX_GRAY_LEVEL; i++) {
        CDF[i] = CDF[i - 1] + PDF[i];
    }

    // Normalize CDF to [0, 255]
    for (int i = 0; i < MAX_GRAY_LEVEL; i++) {
        CDF[i] = (CDF[i] / CDF[MAX_GRAY_LEVEL - 1]) * (MAX_GRAY_LEVEL - 1);
    }

    // 4. Map input pixel intensities using the CDF
    for (int i = 0; i < size; i++) {
        enhanced_Y_channel[i] = (uint8_t)CDF[Y_channel[i]];
    }
}

int main() {
    // Sample usage with a 256x256 RGB image
    int width = 256;
    int height = 256;
    int size = width * height;
    uint8_t rgb_image[256 * 256 * 3];    // Initialize with your RGB image data
    uint8_t Y_channel[256 * 256];        // Y channel
    uint8_t Cb_channel[256 * 256];       // Cb channel
    uint8_t Cr_channel[256 * 256];       // Cr channel
    uint8_t enhanced_Y_channel[256 * 256]; // Enhanced Y channel
    uint8_t output_rgb_image[256 * 256 * 3]; // Output RGB image

    // 1. Convert RGB to YCbCr
    RGB_to_YCbCr_neon(rgb_image, Y_channel, Cb_channel, Cr_channel, width, height);

    // 2. Apply WTHE to the Y channel
    float r = 0.5f;  // Degree of enhancement
    WTHE_neon(Y_channel, enhanced_Y_channel, width, height, r);

    // 3. Replace the original Y channel with the enhanced Y channel
    // (Cb and Cr channels remain the same)
    // This step is already done by storing the result in enhanced_Y_channel

    // 4. Convert YCbCr back to RGB using the enhanced Y channel
    YCbCr_to_RGB_neon(enhanced_Y_channel, Cb_channel, Cr_channel, output_rgb_image, width, height);

    // Now, output_rgb_image contains the final enhanced RGB image
    // You can save it or display it as needed

    return 0;
}