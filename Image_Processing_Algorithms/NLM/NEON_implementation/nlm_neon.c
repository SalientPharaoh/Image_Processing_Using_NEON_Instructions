/**
 * @file nlm_neon.c
 * @brief NEON-optimized implementation of Non-Local Means (NLM) denoising algorithm
 *
 * This implementation provides a high-performance version of the NLM denoising
 * algorithm using ARM NEON SIMD instructions. The algorithm reduces noise while
 * preserving image details by:
 * 1. Computing weighted averages of pixels with similar neighborhoods
 * 2. Using patch-based similarity measures for robust noise reduction
 * 3. Employing parallel processing and SIMD optimizations for efficiency
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>

#define MAX_THREADS 8      // Maximum number of parallel threads
#define BLOCK_SIZE 16     // Block size for cache-efficient processing

/**
 * @brief Thread arguments for parallel processing
 */
typedef struct {
    const float* input;     // Input image data
    float* output;          // Output denoised image
    int width;             // Image width
    int height;            // Image height
    int start_row;         // Starting row for this thread
    int end_row;           // Ending row for this thread
    float h;               // Filtering parameter (controls smoothing strength)
    int patch_size;        // Size of patches for similarity comparison
    int search_window;     // Size of search window around each pixel
} ThreadArgs;

/**
 * @brief Compute L2 distance between two patches using NEON
 *
 * Calculates the squared Euclidean distance between two image patches
 * using NEON SIMD instructions for vectorized computation.
 *
 * @param p1 Pointer to first patch
 * @param p2 Pointer to second patch
 * @param patch_size Size of the patch
 * @param stride Image width (for row-wise traversal)
 * @return float Normalized patch distance
 */
float compute_patch_distance_neon(const float* p1, const float* p2, 
                                int patch_size, int stride) {
    // Initialize accumulator for sum of squared differences
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    // Process patch pixels in groups of 4 using NEON
    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j += 4) {
            // Load 4 pixels from each patch
            float32x4_t v1 = vld1q_f32(&p1[i * stride + j]);
            float32x4_t v2 = vld1q_f32(&p2[i * stride + j]);
            
            // Compute squared differences using NEON
            float32x4_t diff = vsubq_f32(v1, v2);
            sum_vec = vmlaq_f32(sum_vec, diff, diff);
        }
    }
    
    // Reduce vector sum and normalize
    float sum = vaddvq_f32(sum_vec);
    return sum / (patch_size * patch_size);
}

/**
 * @brief Compute weights for patch similarities using NEON
 *
 * Converts patch distances to weights using an exponential function
 * and NEON vectorized operations.
 *
 * @param distances Array of patch distances
 * @param weights Output array for computed weights
 * @param size Number of weights to compute
 * @param h Filtering parameter (controls decay of weights)
 */
void compute_weights_neon(const float* distances, float* weights, 
                         int size, float h) {
    // Precompute constant for exponential
    float32x4_t vh = vdupq_n_f32(-1.0f / (h * h));
    
    // Process weights in groups of 4 using NEON
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t vdist = vld1q_f32(&distances[i]);
        float32x4_t vexp = exp_ps(vmulq_f32(vdist, vh));
        vst1q_f32(&weights[i], vexp);
    }
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        weights[i] = expf(-distances[i] / (h * h));
    }
}

/**
 * @brief Process one block of the image using NLM algorithm
 *
 * Implements the core NLM algorithm for a block of pixels:
 * 1. Computes patch distances within search window
 * 2. Converts distances to weights
 * 3. Applies weighted averaging for denoising
 *
 * @param input Input image
 * @param output Output denoised image
 * @param width Image width
 * @param height Image height
 * @param x Block x-coordinate
 * @param y Block y-coordinate
 * @param h Filtering parameter
 * @param patch_size Size of comparison patches
 * @param search_window Size of search window
 */
void process_block_neon(const float* input, float* output,
                       int width, int height, int x, int y,
                       float h, int patch_size, int search_window) {
    int half_patch = patch_size / 2;
    int half_search = search_window / 2;
    
    // Allocate aligned memory for intermediate results
    int search_size = search_window * search_window;
    float* distances = (float*)aligned_alloc(32, search_size * sizeof(float));
    float* weights = (float*)aligned_alloc(32, search_size * sizeof(float));
    
    // Process each pixel in the block
    for (int by = y; by < min(y + BLOCK_SIZE, height - patch_size); by++) {
        for (int bx = x; bx < min(x + BLOCK_SIZE, width - patch_size); bx++) {
            // Initialize accumulators for weighted sum
            float32x4_t sum = vdupq_n_f32(0.0f);
            float32x4_t weight_sum = vdupq_n_f32(0.0f);
            
            // Compute distances to all patches in search window
            int dist_idx = 0;
            for (int sy = max(by - half_search, half_patch); 
                 sy <= min(by + half_search, height - half_patch - 1); sy++) {
                for (int sx = max(bx - half_search, half_patch);
                     sx <= min(bx + half_search, width - half_patch - 1); sx++) {
                    // Skip the reference patch
                    if (sx == bx && sy == by) continue;
                    
                    // Compute patch distance using NEON
                    distances[dist_idx++] = compute_patch_distance_neon(
                        &input[by * width + bx],
                        &input[sy * width + sx],
                        patch_size, width
                    );
                }
            }
            
            // Convert distances to weights using NEON
            compute_weights_neon(distances, weights, dist_idx, h);
            
            // Apply weights and accumulate result using NEON
            dist_idx = 0;
            for (int sy = max(by - half_search, half_patch);
                 sy <= min(by + half_search, height - half_patch - 1); sy++) {
                for (int sx = max(bx - half_search, half_patch);
                     sx <= min(bx + half_search, width - half_patch - 1); sx++) {
                    if (sx == bx && sy == by) continue;
                    
                    // Vectorized weighted accumulation
                    float32x4_t vweight = vdupq_n_f32(weights[dist_idx]);
                    float32x4_t vpixel = vld1q_f32(&input[sy * width + sx]);
                    sum = vmlaq_f32(sum, vpixel, vweight);
                    weight_sum = vaddq_f32(weight_sum, vweight);
                    dist_idx++;
                }
            }
            
            // Normalize and store final result
            float32x4_t vresult = vdivq_f32(sum, weight_sum);
            vst1q_f32(&output[by * width + bx], vresult);
        }
    }
    
    // Clean up
    free(distances);
    free(weights);
}

/**
 * @brief Thread function for parallel NLM processing
 *
 * Processes assigned image rows in blocks for better cache utilization.
 *
 * @param arg Thread arguments (ThreadArgs structure)
 * @return void* NULL
 */
void* nlm_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    // Process image in blocks for cache efficiency
    for (int y = args->start_row; y < args->end_row; y += BLOCK_SIZE) {
        for (int x = 0; x < args->width; x += BLOCK_SIZE) {
            process_block_neon(
                args->input, args->output,
                args->width, args->height,
                x, y, args->h,
                args->patch_size, args->search_window
            );
        }
    }
    
    return NULL;
}

/**
 * @brief Main NLM denoising function using NEON optimization
 *
 * Implements parallel Non-Local Means denoising using multiple threads
 * and NEON SIMD instructions for optimal performance.
 *
 * @param input Input noisy image
 * @param output Output denoised image
 * @param width Image width
 * @param height Image height
 * @param h Filtering parameter (controls smoothing strength)
 * @param patch_size Size of patches for similarity comparison
 * @param search_window Size of search window around each pixel
 */
void nlm_denoise_neon(const float* input, float* output,
                     int width, int height,
                     float h, int patch_size, int search_window) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];
    
    // Initialize output buffer
    memset(output, 0, width * height * sizeof(float));
    
    // Create threads for parallel processing
    int rows_per_thread = height / MAX_THREADS;
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_args[i].input = input;
        thread_args[i].output = output;
        thread_args[i].width = width;
        thread_args[i].height = height;
        thread_args[i].start_row = i * rows_per_thread;
        thread_args[i].end_row = (i == MAX_THREADS - 1) ? 
                                height : (i + 1) * rows_per_thread;
        thread_args[i].h = h;
        thread_args[i].patch_size = patch_size;
        thread_args[i].search_window = search_window;
        
        pthread_create(&threads[i], NULL, nlm_thread, &thread_args[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}
