#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <pthread.h>

#define MAX_THREADS 8
#define BLOCK_SIZE 16

typedef struct {
    const float* input;
    float* output;
    int width;
    int height;
    int start_row;
    int end_row;
    int radius;
} ThreadArgs;

// NEON-optimized box filter implementation
void box_filter_neon(const float* input, float* output,
                    int width, int height, int radius) {
    float32x4_t area = vdupq_n_f32(1.0f / ((2 * radius + 1) * (2 * radius + 1)));
    
    // Horizontal pass with NEON
    float* temp = (float*)aligned_alloc(32, width * height * sizeof(float));
    
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float sum = 0;
        // Initialize sum for the first window
        for (int x = 0; x <= radius; x++) {
            sum += input[y * width + x];
        }
        
        for (int x = 0; x < width; x++) {
            if (x > radius) {
                sum -= input[y * width + (x - radius - 1)];
            }
            if (x + radius < width) {
                sum += input[y * width + (x + radius)];
            }
            temp[y * width + x] = sum;
        }
    }
    
    // Vertical pass with NEON
    #pragma omp parallel for
    for (int x = 0; x < width; x += 4) {
        float32x4_t sum_vec = vdupq_n_f32(0);
        
        // Initialize sums for the first window
        for (int y = 0; y <= radius; y++) {
            float32x4_t val = vld1q_f32(&temp[y * width + x]);
            sum_vec = vaddq_f32(sum_vec, val);
        }
        
        for (int y = 0; y < height; y++) {
            if (y > radius) {
                float32x4_t prev = vld1q_f32(&temp[(y - radius - 1) * width + x]);
                sum_vec = vsubq_f32(sum_vec, prev);
            }
            if (y + radius < height) {
                float32x4_t next = vld1q_f32(&temp[(y + radius) * width + x]);
                sum_vec = vaddq_f32(sum_vec, next);
            }
            
            // Normalize and store
            float32x4_t result = vmulq_f32(sum_vec, area);
            vst1q_f32(&output[y * width + x], result);
        }
    }
    
    free(temp);
}

// NEON-optimized covariance computation
void compute_covariance_neon(const float* I, const float* P,
                            float* cov, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            float32x4_t vi = vld1q_f32(&I[y * width + x]);
            float32x4_t vp = vld1q_f32(&P[y * width + x]);
            float32x4_t vcov = vmulq_f32(vi, vp);
            vst1q_f32(&cov[y * width + x], vcov);
        }
    }
}

// Thread function for parallel processing
void* guided_filter_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    for (int y = args->start_row; y < args->end_row; y += BLOCK_SIZE) {
        for (int x = 0; x < args->width; x += BLOCK_SIZE) {
            // Process block
            int block_width = min(BLOCK_SIZE, args->width - x);
            int block_height = min(BLOCK_SIZE, args->end_row - y);
            
            // Local mean computation using box filter
            float* mean_I = (float*)aligned_alloc(32, block_width * block_height * sizeof(float));
            float* mean_P = (float*)aligned_alloc(32, block_width * block_height * sizeof(float));
            
            box_filter_neon(&args->input[y * args->width + x], mean_I,
                          block_width, block_height, args->radius);
            box_filter_neon(&args->input[y * args->width + x], mean_P,
                          block_width, block_height, args->radius);
            
            // Compute covariance
            float* cov_IP = (float*)aligned_alloc(32, block_width * block_height * sizeof(float));
            compute_covariance_neon(mean_I, mean_P, cov_IP,
                                  block_width, block_height);
            
            // Process block with NEON
            for (int by = 0; by < block_height; by += 4) {
                for (int bx = 0; bx < block_width; bx += 4) {
                    float32x4_t vmean_I = vld1q_f32(&mean_I[by * block_width + bx]);
                    float32x4_t vmean_P = vld1q_f32(&mean_P[by * block_width + bx]);
                    float32x4_t vcov_IP = vld1q_f32(&cov_IP[by * block_width + bx]);
                    
                    // Compute a and b
                    float32x4_t va = vdivq_f32(vcov_IP, vaddq_f32(vmulq_f32(vmean_I, vmean_I),
                                             vdupq_n_f32(args->epsilon)));
                    float32x4_t vb = vsubq_f32(vmean_P, vmulq_f32(va, vmean_I));
                    
                    // Store results
                    vst1q_f32(&args->output[(y + by) * args->width + x + bx],
                             vaddq_f32(vmulq_f32(va, vmean_I), vb));
                }
            }
            
            free(mean_I);
            free(mean_P);
            free(cov_IP);
        }
    }
    
    return NULL;
}

// Main guided filter function
void guided_filter_neon(const float* input, const float* guidance,
                       float* output, int width, int height,
                       int radius, float epsilon, int subsample) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];
    
    // Calculate subsampled dimensions
    int sub_width = width / subsample;
    int sub_height = height / subsample;
    
    // Allocate memory for subsampled images
    float* sub_input = (float*)aligned_alloc(32, sub_width * sub_height * sizeof(float));
    float* sub_guidance = (float*)aligned_alloc(32, sub_width * sub_height * sizeof(float));
    float* sub_output = (float*)aligned_alloc(32, sub_width * sub_height * sizeof(float));
    
    // Subsample input and guidance images
    for (int y = 0; y < sub_height; y++) {
        for (int x = 0; x < sub_width; x++) {
            sub_input[y * sub_width + x] = input[(y * subsample) * width + (x * subsample)];
            sub_guidance[y * sub_width + x] = guidance[(y * subsample) * width + (x * subsample)];
        }
    }
    
    // Process subsampled image with multiple threads
    int rows_per_thread = sub_height / MAX_THREADS;
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_args[i].input = sub_input;
        thread_args[i].output = sub_output;
        thread_args[i].width = sub_width;
        thread_args[i].height = sub_height;
        thread_args[i].start_row = i * rows_per_thread;
        thread_args[i].end_row = (i == MAX_THREADS - 1) ? 
                                sub_height : (i + 1) * rows_per_thread;
        thread_args[i].radius = radius / subsample;
        
        pthread_create(&threads[i], NULL, guided_filter_thread, &thread_args[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Upsample result using bilinear interpolation with NEON
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 4) {
            float sx = (float)x / subsample;
            float sy = (float)y / subsample;
            
            int x0 = (int)sx;
            int y0 = (int)sy;
            int x1 = min(x0 + 1, sub_width - 1);
            int y1 = min(y0 + 1, sub_height - 1);
            
            float32x4_t fx = vdup_n_f32(sx - x0);
            float32x4_t fy = vdup_n_f32(sy - y0);
            
            float32x4_t v00 = vld1q_f32(&sub_output[y0 * sub_width + x0]);
            float32x4_t v01 = vld1q_f32(&sub_output[y0 * sub_width + x1]);
            float32x4_t v10 = vld1q_f32(&sub_output[y1 * sub_width + x0]);
            float32x4_t v11 = vld1q_f32(&sub_output[y1 * sub_width + x1]);
            
            // Bilinear interpolation
            float32x4_t result = vmlaq_f32(
                vmlaq_f32(
                    vmulq_f32(v00, vsubq_f32(vdup_n_f32(1.0f), fx)),
                    v01, fx
                ),
                vmlaq_f32(
                    vmulq_f32(v10, vsubq_f32(vdup_n_f32(1.0f), fx)),
                    v11, fx
                ),
                fy
            );
            
            vst1q_f32(&output[y * width + x], result);
        }
    }
    
    // Free memory
    free(sub_input);
    free(sub_guidance);
    free(sub_output);
}
