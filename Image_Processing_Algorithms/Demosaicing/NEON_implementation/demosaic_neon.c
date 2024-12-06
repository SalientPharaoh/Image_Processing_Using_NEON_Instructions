#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <pthread.h>

#define MAX_THREADS 8
#define BLOCK_SIZE 32

typedef struct {
    const unsigned char* bayer;
    unsigned char* rgb;
    int width;
    int height;
    int start_row;
    int end_row;
    int pattern;
} ThreadArgs;

// NEON-optimized gradient computation
void compute_gradients_neon(const unsigned char* bayer,
                          float* h_grad, float* v_grad,
                          int width, int height) {
    for (int y = 2; y < height - 2; y += 1) {
        for (int x = 2; x < width - 2; x += 8) {
            // Load 8 pixels at a time
            uint8x8_t center = vld1_u8(&bayer[y * width + x]);
            uint8x8_t left2 = vld1_u8(&bayer[y * width + (x - 2)]);
            uint8x8_t right2 = vld1_u8(&bayer[y * width + (x + 2)]);
            uint8x8_t top2 = vld1_u8(&bayer[(y - 2) * width + x]);
            uint8x8_t bottom2 = vld1_u8(&bayer[(y + 2) * width + x]);
            
            // Convert to 16-bit for intermediate calculations
            int16x8_t h_diff = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(right2)),
                                       vreinterpretq_s16_u16(vmovl_u8(left2)));
            int16x8_t v_diff = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(bottom2)),
                                       vreinterpretq_s16_u16(vmovl_u8(top2)));
            
            // Convert to float and store
            float32x4_t h_grad_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(h_diff)));
            float32x4_t h_grad_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(h_diff)));
            float32x4_t v_grad_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_diff)));
            float32x4_t v_grad_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_diff)));
            
            vst1q_f32(&h_grad[y * width + x], h_grad_low);
            vst1q_f32(&h_grad[y * width + x + 4], h_grad_high);
            vst1q_f32(&v_grad[y * width + x], v_grad_low);
            vst1q_f32(&v_grad[y * width + x + 4], v_grad_high);
        }
    }
}

// NEON-optimized green channel interpolation
void interpolate_green_neon(const unsigned char* bayer,
                          unsigned char* green,
                          int width, int height,
                          int pattern) {
    // Allocate gradient buffers
    float* h_grad = (float*)aligned_alloc(32, width * height * sizeof(float));
    float* v_grad = (float*)aligned_alloc(32, width * height * sizeof(float));
    
    // Compute gradients
    compute_gradients_neon(bayer, h_grad, v_grad, width, height);
    
    // Interpolate green channel
    for (int y = 2; y < height - 2; y += 1) {
        for (int x = 2; x < width - 2; x += 8) {
            // Check if current pixel needs green interpolation
            if ((x + y) % 2 == 0) {
                // Load gradient values
                float32x4_t h_grad_val = vld1q_f32(&h_grad[y * width + x]);
                float32x4_t v_grad_val = vld1q_f32(&v_grad[y * width + x]);
                
                // Load neighboring green pixels
                uint8x8_t left = vld1_u8(&bayer[y * width + (x - 1)]);
                uint8x8_t right = vld1_u8(&bayer[y * width + (x + 1)]);
                uint8x8_t top = vld1_u8(&bayer[(y - 1) * width + x]);
                uint8x8_t bottom = vld1_u8(&bayer[(y + 1) * width + x]);
                
                // Convert to float for calculations
                float32x4_t h_val = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(left))));
                float32x4_t v_val = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(top))));
                
                // Compute weights based on gradients
                float32x4_t h_weight = vrecpeq_f32(vaddq_f32(h_grad_val, vdupq_n_f32(1.0f)));
                float32x4_t v_weight = vrecpeq_f32(vaddq_f32(v_grad_val, vdupq_n_f32(1.0f)));
                
                // Weighted interpolation
                float32x4_t result = vaddq_f32(
                    vmulq_f32(h_val, h_weight),
                    vmulq_f32(v_val, v_weight)
                );
                result = vmulq_f32(result, vrecpeq_f32(vaddq_f32(h_weight, v_weight)));
                
                // Convert back to uint8 and store
                uint8x8_t final = vqmovn_u16(vcombine_u16(
                    vqmovn_u32(vcvtq_u32_f32(result)),
                    vdup_n_u16(0)
                ));
                vst1_u8(&green[y * width + x], final);
            } else {
                // Copy original green values
                vst1_u8(&green[y * width + x],
                        vld1_u8(&bayer[y * width + x]));
            }
        }
    }
    
    free(h_grad);
    free(v_grad);
}

// NEON-optimized red/blue interpolation
void interpolate_red_blue_neon(const unsigned char* bayer,
                             unsigned char* red,
                             unsigned char* blue,
                             const unsigned char* green,
                             int width, int height,
                             int pattern) {
    for (int y = 2; y < height - 2; y += 1) {
        for (int x = 2; x < width - 2; x += 8) {
            // Determine pixel type based on Bayer pattern
            int is_red = ((x + y) % 2 == 0 && pattern == BAYER_RGGB) ||
                        ((x + y) % 2 == 1 && pattern == BAYER_GRBG);
            int is_blue = ((x + y) % 2 == 0 && pattern == BAYER_BGGR) ||
                         ((x + y) % 2 == 1 && pattern == BAYER_GBRG);
            
            if (is_red || is_blue) {
                // Load center and neighboring pixels
                uint8x8_t center = vld1_u8(&bayer[y * width + x]);
                uint8x8_t green_center = vld1_u8(&green[y * width + x]);
                
                // Load diagonal neighbors
                uint8x8_t top_left = vld1_u8(&bayer[(y - 1) * width + (x - 1)]);
                uint8x8_t top_right = vld1_u8(&bayer[(y - 1) * width + (x + 1)]);
                uint8x8_t bottom_left = vld1_u8(&bayer[(y + 1) * width + (x - 1)]);
                uint8x8_t bottom_right = vld1_u8(&bayer[(y + 1) * width + (x + 1)]);
                
                // Convert to 16-bit for calculations
                uint16x8_t sum = vaddl_u8(top_left, top_right);
                sum = vaddw_u8(sum, bottom_left);
                sum = vaddw_u8(sum, bottom_right);
                
                // Average the diagonal values
                uint8x8_t avg = vqmovn_u16(vshrq_n_u16(sum, 2));
                
                // Store results based on Bayer pattern
                if (is_red) {
                    vst1_u8(&red[y * width + x], center);
                    vst1_u8(&blue[y * width + x], avg);
                } else {
                    vst1_u8(&red[y * width + x], avg);
                    vst1_u8(&blue[y * width + x], center);
                }
            } else {
                // Interpolate red/blue at green pixels
                uint8x8_t left = vld1_u8(&bayer[y * width + (x - 1)]);
                uint8x8_t right = vld1_u8(&bayer[y * width + (x + 1)]);
                uint8x8_t top = vld1_u8(&bayer[(y - 1) * width + x]);
                uint8x8_t bottom = vld1_u8(&bayer[(y + 1) * width + x]);
                
                uint16x8_t h_sum = vaddl_u8(left, right);
                uint16x8_t v_sum = vaddl_u8(top, bottom);
                
                uint8x8_t h_avg = vqmovn_u16(vshrq_n_u16(h_sum, 1));
                uint8x8_t v_avg = vqmovn_u16(vshrq_n_u16(v_sum, 1));
                
                // Store interpolated values based on pattern
                if ((y % 2 == 0 && pattern == BAYER_RGGB) ||
                    (y % 2 == 1 && pattern == BAYER_GRBG)) {
                    vst1_u8(&red[y * width + x], h_avg);
                    vst1_u8(&blue[y * width + x], v_avg);
                } else {
                    vst1_u8(&red[y * width + x], v_avg);
                    vst1_u8(&blue[y * width + x], h_avg);
                }
            }
        }
    }
}

// Thread function for parallel processing
void* demosaic_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    // Allocate temporary buffers for this thread
    unsigned char* green = (unsigned char*)aligned_alloc(32,
        args->width * (args->end_row - args->start_row) * sizeof(unsigned char));
    unsigned char* red = (unsigned char*)aligned_alloc(32,
        args->width * (args->end_row - args->start_row) * sizeof(unsigned char));
    unsigned char* blue = (unsigned char*)aligned_alloc(32,
        args->width * (args->end_row - args->start_row) * sizeof(unsigned char));
    
    // Process green channel
    interpolate_green_neon(
        args->bayer + args->start_row * args->width,
        green,
        args->width,
        args->end_row - args->start_row,
        args->pattern
    );
    
    // Process red and blue channels
    interpolate_red_blue_neon(
        args->bayer + args->start_row * args->width,
        red, blue, green,
        args->width,
        args->end_row - args->start_row,
        args->pattern
    );
    
    // Interleave RGB channels into final output
    for (int y = args->start_row; y < args->end_row; y++) {
        for (int x = 0; x < args->width; x++) {
            int idx = (y - args->start_row) * args->width + x;
            args->rgb[(y * args->width + x) * 3 + 0] = red[idx];
            args->rgb[(y * args->width + x) * 3 + 1] = green[idx];
            args->rgb[(y * args->width + x) * 3 + 2] = blue[idx];
        }
    }
    
    free(green);
    free(red);
    free(blue);
    
    return NULL;
}

// Main demosaicing function
void demosaic_neon(const unsigned char* bayer,
                  unsigned char* rgb,
                  int width, int height,
                  int pattern) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];
    
    // Create threads for parallel processing
    int rows_per_thread = height / MAX_THREADS;
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_args[i].bayer = bayer;
        thread_args[i].rgb = rgb;
        thread_args[i].width = width;
        thread_args[i].height = height;
        thread_args[i].start_row = i * rows_per_thread;
        thread_args[i].end_row = (i == MAX_THREADS - 1) ?
                                height : (i + 1) * rows_per_thread;
        thread_args[i].pattern = pattern;
        
        pthread_create(&threads[i], NULL, demosaic_thread, &thread_args[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}
