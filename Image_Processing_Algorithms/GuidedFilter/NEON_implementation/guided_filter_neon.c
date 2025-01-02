/**
 * @file guided_filter_neon.c
 * @brief NEON-optimized implementation of the Guided Image Filter
 *
 * This implementation provides a high-performance version of the Guided Image Filter
 * using ARM NEON SIMD instructions. The algorithm performs edge-preserving smoothing
 * by using a guidance image to compute local linear models. Key features:
 * 1. Fast box filter implementation using NEON
 * 2. Efficient local linear model computation
 * 3. Multi-threaded processing with cache-friendly blocking
 * 4. Support for subsampling to improve performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <pthread.h>

#define MAX_THREADS 8      // Maximum number of parallel threads
#define BLOCK_SIZE 16     // Block size for cache-efficient processing

/**
 * @brief Thread arguments for parallel processing
 */
typedef struct {
    const float* input;     // Input image data
    float* output;          // Output filtered image
    int width;             // Image width
    int height;            // Image height
    int start_row;         // Starting row for this thread
    int end_row;           // Ending row for this thread
    int radius;            // Filter radius
    float epsilon;         // Regularization parameter
} ThreadArgs;

/**
 * @brief NEON-optimized box filter implementation
 *
 * Implements a separable box filter using NEON SIMD instructions.
 * The filter is applied in two passes (horizontal and vertical) for efficiency.
 *
 * @param input Input image
 * @param output Output filtered image
 * @param width Image width
 * @param height Image height
 * @param radius Filter radius (window size = 2*radius + 1)
 */
void box_filter_neon(const float* input, float* output,
                    int width, int height, int radius) {
    // Precompute normalization factor
    float32x4_t area = vdupq_n_f32(1.0f / ((2 * radius + 1) * (2 * radius + 1)));
    
    // Allocate temporary buffer for separable filtering
    float* temp = (float*)aligned_alloc(32, width * height * sizeof(float));
    
    // Horizontal pass - process each row independently
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float sum = 0;
        // Initialize sum for the first window
        for (int x = 0; x <= radius; x++) {
            sum += input[y * width + x];
        }
        
        // Slide window horizontally
        for (int x = 0; x < width; x++) {
            // Remove leftmost pixel from sum if available
            if (x > radius) {
                sum -= input[y * width + (x - radius - 1)];
            }
            // Add rightmost pixel to sum if available
            if (x + radius < width) {
                sum += input[y * width + (x + radius)];
            }
            temp[y * width + x] = sum;
        }
    }
    
    // Vertical pass with NEON vectorization
    #pragma omp parallel for
    for (int x = 0; x < width; x += 4) {
        // Initialize vector accumulator
        float32x4_t sum_vec = vdupq_n_f32(0);
        
        // Initialize sums for the first window
        for (int y = 0; y <= radius; y++) {
            float32x4_t val = vld1q_f32(&temp[y * width + x]);
            sum_vec = vaddq_f32(sum_vec, val);
        }
        
        // Slide window vertically
        for (int y = 0; y < height; y++) {
            // Remove top pixel from sum if available
            if (y > radius) {
                float32x4_t prev = vld1q_f32(&temp[(y - radius - 1) * width + x]);
                sum_vec = vsubq_f32(sum_vec, prev);
            }
            // Add bottom pixel to sum if available
            if (y + radius < height) {
                float32x4_t next = vld1q_f32(&temp[(y + radius) * width + x]);
                sum_vec = vaddq_f32(sum_vec, next);
            }
            
            // Normalize and store result
            float32x4_t result = vmulq_f32(sum_vec, area);
            vst1q_f32(&output[y * width + x], result);
        }
    }
    
    free(temp);
}

/**
 * @brief Compute covariance between guidance and input images using NEON
 *
 * Calculates the local covariance between the guidance image I and
 * input image P using NEON vectorized operations.
 *
 * @param I Guidance image
 * @param P Input image
 * @param cov Output covariance
 * @param width Image width
 * @param height Image height
 */
void compute_covariance_neon(const float* I, const float* P,
                            float* cov, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            // Load 4 pixels from each image
            float32x4_t vi = vld1q_f32(&I[y * width + x]);
            float32x4_t vp = vld1q_f32(&P[y * width + x]);
            // Compute element-wise multiplication
            float32x4_t vcov = vmulq_f32(vi, vp);
            vst1q_f32(&cov[y * width + x], vcov);
        }
    }
}

/**
 * @brief Thread function for parallel guided filter processing
 *
 * Processes a portion of the image using block-based computation
 * for better cache utilization. For each block:
 * 1. Computes local means using box filter
 * 2. Computes covariance between guidance and input
 * 3. Solves for local linear coefficients a and b
 * 4. Applies the linear transform
 *
 * @param arg Thread arguments (ThreadArgs structure)
 * @return void* NULL
 */
void* guided_filter_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    // Process image in blocks for cache efficiency
    for (int y = args->start_row; y < args->end_row; y += BLOCK_SIZE) {
        for (int x = 0; x < args->width; x += BLOCK_SIZE) {
            // Calculate actual block dimensions
            int block_width = min(BLOCK_SIZE, args->width - x);
            int block_height = min(BLOCK_SIZE, args->end_row - y);
            
            // Allocate aligned buffers for local computations
            float* mean_I = (float*)aligned_alloc(32, block_width * block_height * sizeof(float));
            float* mean_P = (float*)aligned_alloc(32, block_width * block_height * sizeof(float));
            
            // Compute local means using box filter
            box_filter_neon(&args->input[y * args->width + x], mean_I,
                          block_width, block_height, args->radius);
            box_filter_neon(&args->input[y * args->width + x], mean_P,
                          block_width, block_height, args->radius);
            
            // Compute local covariance
            float* cov_IP = (float*)aligned_alloc(32, block_width * block_height * sizeof(float));
            compute_covariance_neon(mean_I, mean_P, cov_IP,
                                  block_width, block_height);
            
            // Process block using NEON vectorization
            for (int by = 0; by < block_height; by += 4) {
                for (int bx = 0; bx < block_width; bx += 4) {
                    // Load mean values and covariance
                    float32x4_t vmean_I = vld1q_f32(&mean_I[by * block_width + bx]);
                    float32x4_t vmean_P = vld1q_f32(&mean_P[by * block_width + bx]);
                    float32x4_t vcov_IP = vld1q_f32(&cov_IP[by * block_width + bx]);
                    
                    // Compute linear coefficients a and b
                    float32x4_t va = vdivq_f32(vcov_IP, vaddq_f32(vmulq_f32(vmean_I, vmean_I),
                                             vdupq_n_f32(args->epsilon)));
                    float32x4_t vb = vsubq_f32(vmean_P, vmulq_f32(va, vmean_I));
                    
                    // Apply linear transform and store result
                    vst1q_f32(&args->output[(y + by) * args->width + x + bx],
                             vaddq_f32(vmulq_f32(va, vmean_I), vb));
                }
            }
            
            // Clean up local buffers
            free(mean_I);
            free(mean_P);
            free(cov_IP);
        }
    }
    
    return NULL;
}

/**
 * @brief Main guided filter function using NEON optimization
 *
 * Implements the guided filter with the following optimizations:
 * 1. Subsampling for faster processing
 * 2. Multi-threaded computation
 * 3. NEON SIMD instructions
 * 4. Cache-efficient blocking
 *
 * @param input Input image to be filtered
 * @param guidance Guidance image
 * @param output Output filtered image
 * @param width Image width
 * @param height Image height
 * @param radius Filter radius
 * @param epsilon Regularization parameter
 * @param subsample Subsampling factor for speedup
 */
void guided_filter_neon(const float* input, const float* guidance,
                       float* output, int width, int height,
                       int radius, float epsilon, int subsample) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];
    
    // Calculate subsampled dimensions
    int sub_width = width / subsample;
    int sub_height = height / subsample;
    
    // Allocate aligned memory for subsampled images
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
    
    // Create threads for parallel processing
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
        thread_args[i].epsilon = epsilon;
        
        pthread_create(&threads[i], NULL, guided_filter_thread, &thread_args[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Upsample result using bilinear interpolation with NEON
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 4) {
            // Calculate sampling coordinates
            float sx = (float)x / subsample;
            float sy = (float)y / subsample;
            
            // Find nearest pixel coordinates
            int x0 = (int)sx;
            int y0 = (int)sy;
            int x1 = min(x0 + 1, sub_width - 1);
            int y1 = min(y0 + 1, sub_height - 1);
            
            // Calculate interpolation weights
            float32x4_t fx = vdup_n_f32(sx - x0);
            float32x4_t fy = vdup_n_f32(sy - y0);
            
            // Load neighboring pixels
            float32x4_t v00 = vld1q_f32(&sub_output[y0 * sub_width + x0]);
            float32x4_t v01 = vld1q_f32(&sub_output[y0 * sub_width + x1]);
            float32x4_t v10 = vld1q_f32(&sub_output[y1 * sub_width + x0]);
            float32x4_t v11 = vld1q_f32(&sub_output[y1 * sub_width + x1]);
            
            // Perform bilinear interpolation using NEON
            float32x4_t result = vmlaq_f32(
                vmlaq_f32(
                    vmulq_f32(v00, vsubq_f32(vdupq_n_f32(1.0f), fx)),
                    v01, fx
                ),
                vmlaq_f32(
                    vmulq_f32(v10, vsubq_f32(vdupq_n_f32(1.0f), fx)),
                    v11, fx
                ),
                fy
            );
            
            // Store interpolated result
            vst1q_f32(&output[y * width + x], result);
        }
    }
    
    // Clean up
    free(sub_input);
    free(sub_guidance);
    free(sub_output);
}
