#include <arm_neon.h>
#include <math.h>
#include <stdio.h>

void imhist(uint8_t *image, int height, int width, int *hist) {
    for (int i = 0; i < 256; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 4) {
            uint8x16_t pixels = vld1q_u8(&image[i * width + j]);
            int32x4_t counts = vdupq_n_s32(0);
            for (int k = 0; k < 16; k++) {
                uint8_t pixel = vgetq_lane_u8(pixels, k);
                counts = vaddq_s32(counts, vreinterpretq_s32_u8(vceqq_u8(vdupq_n_u8(pixel), pixels)));
            }
            int32x2_t low = vget_low_s32(counts);
            int32x2_t high = vget_high_s32(counts);
            hist[vget_lane_s32(low, 0)] += vget_lane_s32(low, 1);
            hist[vget_lane_s32(high, 0)] += vget_lane_s32(high, 1);
        }
    }
}

void imWTHeq(uint8_t *image, int height, int width, float *Wout_list, uint8_t *image_heq, float *Wout) {
    int hist[256];
    imhist(image, height, width, hist);
    float PMF[256];
    for (int i = 0; i < 256; i++) {
        PMF[i] = (float)hist[i] / (height * width);
    }
    float Pl = 1e-4;
    float Pu = v * PMF[255];
    float Pwt[256];
    for (int i = 0; i < 256; i++) {
        if (PMF[i] < Pl) {
            Pwt[i] = 0;
        } else if (PMF[i] > Pu) {
            Pwt[i] = Pu;
        } else {
            Pwt[i] = powf((PMF[i] - Pl) / (Pu - Pl), r) * Pu;
        }
    }
    float Cwt[256];
    Cwt[0] = Pwt[0];
    for (int i = 1; i < 256; i++) {
        Cwt[i] = Cwt[i - 1] + Pwt[i];
    }
    float Cwtn[256];
    for (int i = 0; i < 256; i++) {
        Cwtn[i] = Cwt[i] / Cwt[255];
    }
    int Win = 0;
    for (int i = 0; i < 256; i++) {
        if (PMF[i] > 0) {
            Win++;
        }
    }
    float Gmax = 1.5;
    *Wout = fminf(255, Gmax * Win);
    int num_Wout_list = 0;
    for (int i = 0; i < 10; i++) {
        if (Wout_list[i] > 0) {
            num_Wout_list++;
        }
    }
    if (num_Wout_list == 10) {
        float sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += Wout_list[i];
        }
         *Wout = (sum + *Wout) / (11);
    }
    float32_t mean_image = 0;
    for (int i = 0; i < height * width; i++) {
        mean_image += image[i];
    }
    mean_image /= height * width;
    float32_t mean_Ftilde = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 4) {
            uint8x16_t pixels = vld1q_u8(&image[i * width + j]);
            float32x4_t Ftilde = vmulq_n_f32(vcvtq_f32_s32(vreinterpretq_s32_u8(pixels)), *Wout);
            Ftilde = vmulq_f32(Ftilde, vld1q_f32(&Cwtn[vgetq_lane_u8(pixels, 0)]));
            Ftilde = vmulq_f32(Ftilde, vld1q_f32(&Cwtn[vgetq_lane_u8(pixels, 1)]));
            Ftilde = vmulq_f32(Ftilde, vld1q_f32(&Cwtn[vgetq_lane_u8(pixels, 2)]));
            Ftilde = vmulq_f32(Ftilde, vld1q_f32(&Cwtn[vgetq_lane_u8(pixels, 3)]));
            vst1q_f32(&image_heq[i * width + j], Ftilde);
            mean_Ftilde += vaddvq_f32(Ftilde);
        }
    }
    mean_Ftilde /= height * width;
    float32_t Madj = mean_image - mean_Ftilde;
    for (int i = 0; i < height * width; i++) {
        float32_t Ftilde = image_heq[i] + Madj;
        if (Ftilde < 0) {
            Ftilde = 0;
        } else if (Ftilde > 255) {
            Ftilde = 255;
        }
        image_heq[i] = (uint8_t)Ftilde;
    }
}