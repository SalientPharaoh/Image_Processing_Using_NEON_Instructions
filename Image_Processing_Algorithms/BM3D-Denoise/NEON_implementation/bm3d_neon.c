/**
 * @file bm3d_neon.c
 * @brief NEON-optimized implementation of the BM3D (Block-Matching and 3D filtering) denoising algorithm
 *
 * This implementation provides a high-performance version of the BM3D algorithm using ARM NEON SIMD
 * instructions. BM3D works in three main steps:
 * 1. Block-matching: Find similar blocks in the image
 * 2. Collaborative filtering: Transform similar blocks and filter them together
 * 3. Aggregation: Combine the filtered blocks back into the final image
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>
#include <float.h>

// Algorithm parameters
#define BLOCK_SIZE 8          // Size of each block (8x8 pixels)
#define SEARCH_WINDOW 39      // Size of search window for similar blocks
#define MAX_SIMILAR_BLOCKS 16 // Maximum number of similar blocks to group
#define HARD_THRESHOLD 2.7f   // Threshold for hard thresholding in transform domain
#define STEP_SIZE 3          // Step size for processing reference blocks
#define NUM_THREADS 4        // Number of parallel threads

// Memory alignment for NEON operations
#define ALIGN32(x) ((((uintptr_t)(x) + 31) & ~31))

/**
 * @brief Structure to store block matching results
 */
typedef struct {
    float distance;  // L2 distance between blocks
    int x;          // X coordinate of the matched block
    int y;          // Y coordinate of the matched block
} BlockMatch;

/**
 * @brief Thread arguments for parallel processing
 */
typedef struct {
    const float* input;   // Input noisy image
    float* output;        // Output denoised image
    int width;           // Image width
    int height;          // Image height
    int start_row;       // Starting row for this thread
    int end_row;         // Ending row for this thread
    float sigma;         // Noise standard deviation
} ThreadArgs;

/**
 * @brief Compute L2 distance between two 8x8 blocks using NEON
 *
 * This function efficiently computes the mean squared error between two blocks
 * using NEON SIMD instructions. It processes 4 pixels at a time using float32x4_t.
 *
 * @param block1 Pointer to first block
 * @param block2 Pointer to second block
 * @param stride Image stride (width)
 * @return float Distance between blocks (normalized MSE)
 */
float compute_block_distance_neon(const float* block1, const float* block2, int stride) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    // Process each row of the blocks
    for (int i = 0; i < BLOCK_SIZE; i++) {
        // Process 4 pixels at a time using NEON
        for (int j = 0; j < BLOCK_SIZE; j += 4) {
            // Load 4 pixels from each block
            float32x4_t v1 = vld1q_f32(&block1[i * stride + j]);
            float32x4_t v2 = vld1q_f32(&block2[i * stride + j]);
            
            // Compute squared difference
            float32x4_t diff = vsubq_f32(v1, v2);
            sum_vec = vmlaq_f32(sum_vec, diff, diff);
        }
    }
    
    // Reduce sum vector to single value and normalize
    float sum = vaddvq_f32(sum_vec);
    return sum / (BLOCK_SIZE * BLOCK_SIZE);
}

/**
 * @brief Perform 2D DCT on 8x8 block using NEON
 *
 * Implements a fast 2D DCT using NEON SIMD instructions. The transform is
 * separable, so it's performed row-wise first, then column-wise.
 * Uses constant multiplication and butterfly operations for efficiency.
 *
 * @param block Input/output block (in-place transform)
 * @param stride Image stride
 */
void dct_2d_8x8_neon(float* block, int stride) {
    // DCT coefficients
    static const float32_t c1 = 0.98078528f;  // cos(1*pi/16)
    static const float32_t c2 = 0.92387953f;  // cos(2*pi/16)
    static const float32_t c3 = 0.83146961f;  // cos(3*pi/16)
    static const float32_t c4 = 0.70710678f;  // cos(4*pi/16)
    static const float32_t c5 = 0.55557023f;  // cos(5*pi/16)
    static const float32_t c6 = 0.38268343f;  // cos(6*pi/16)
    static const float32_t c7 = 0.19509032f;  // cos(7*pi/16)
    
    float32x4_t row_buf[8];
    
    // Row-wise DCT
    for (int i = 0; i < 8; i++) {
        // Load 8 pixels from the row
        float32x4_t row0 = vld1q_f32(&block[i * stride]);
        float32x4_t row1 = vld1q_f32(&block[i * stride + 4]);
        
        // Stage 1: Initial butterfly
        float32x4_t sum04 = vaddq_f32(row0, row1);
        float32x4_t diff04 = vsubq_f32(row0, row1);
        
        // Stage 2: Apply DCT coefficients
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        // Stage 3: Final butterfly
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        // Store intermediate results
        row_buf[i] = out0;
        row_buf[i + 4] = out1;
    }
    
    // Column-wise DCT using intermediate results
    for (int j = 0; j < 8; j++) {
        // Load column from row buffer
        float32x4_t col = vld1q_f32((float*)&row_buf[j]);
        
        // Repeat similar butterfly stages for columns
        float32x4_t sum04 = vaddq_f32(col, vrev64q_f32(col));
        float32x4_t diff04 = vsubq_f32(col, vrev64q_f32(col));
        
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        // Store final results
        vst1q_f32(&block[j * stride], out0);
        vst1q_f32(&block[j * stride + 4], out1);
    }
}

/**
 * @brief Perform inverse 2D DCT on 8x8 block using NEON
 *
 * Implements the inverse DCT transform using NEON SIMD instructions.
 * Similar to forward DCT but with transposed coefficients and final scaling.
 *
 * @param block Input/output block (in-place transform)
 * @param stride Image stride
 */
void idct_2d_8x8_neon(float* block, int stride) {
    // Use same coefficients as forward DCT
    static const float32_t c1 = 0.98078528f;
    static const float32_t c2 = 0.92387953f;
    static const float32_t c3 = 0.83146961f;
    static const float32_t c4 = 0.70710678f;
    static const float32_t c5 = 0.55557023f;
    static const float32_t c6 = 0.38268343f;
    static const float32_t c7 = 0.19509032f;
    
    float32x4_t row_buf[8];
    
    // Row-wise IDCT
    for (int i = 0; i < 8; i++) {
        float32x4_t row0 = vld1q_f32(&block[i * stride]);
        float32x4_t row1 = vld1q_f32(&block[i * stride + 4]);
        
        // Inverse butterfly stages
        float32x4_t sum04 = vaddq_f32(row0, row1);
        float32x4_t diff04 = vsubq_f32(row0, row1);
        
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        row_buf[i] = out0;
        row_buf[i + 4] = out1;
    }
    
    // Column-wise IDCT
    for (int j = 0; j < 8; j++) {
        float32x4_t col = vld1q_f32((float*)&row_buf[j]);
        
        float32x4_t sum04 = vaddq_f32(col, vrev64q_f32(col));
        float32x4_t diff04 = vsubq_f32(col, vrev64q_f32(col));
        
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        vst1q_f32(&block[j * stride], out0);
        vst1q_f32(&block[j * stride + 4], out1);
    }
    
    // Scale the results by 1/8
    float32x4_t scale = vdupq_n_f32(1.0f / 8.0f);
    for (int i = 0; i < 8; i++) {
        float32x4_t row0 = vld1q_f32(&block[i * stride]);
        float32x4_t row1 = vld1q_f32(&block[i * stride + 4]);
        vst1q_f32(&block[i * stride], vmulq_f32(row0, scale));
        vst1q_f32(&block[i * stride + 4], vmulq_f32(row1, scale));
    }
}

/**
 * @brief Apply Wiener filter to a transformed block using NEON
 *
 * Implements the Wiener filter in the transform domain using NEON SIMD.
 * The filter is optimized for the assumed noise model with given sigma.
 *
 * @param block Input/output block (in-place filtering)
 * @param sigma Noise standard deviation
 * @param stride Image stride
 */
void apply_wiener_filter_neon(float* block, float sigma, int stride) {
    // Precompute constants
    float32x4_t sigma_sq = vdupq_n_f32(sigma * sigma);
    float32x4_t one = vdupq_n_f32(1.0f);
    
    // Process block coefficients
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j += 4) {
            // Load 4 coefficients
            float32x4_t coef = vld1q_f32(&block[i * stride + j]);
            
            // Compute Wiener filter weights
            float32x4_t coef_sq = vmulq_f32(coef, coef);
            float32x4_t wiener = vdivq_f32(coef_sq, vaddq_f32(coef_sq, sigma_sq));
            
            // Apply filter and store results
            coef = vmulq_f32(coef, wiener);
            vst1q_f32(&block[i * stride + j], coef);
        }
    }
}

/**
 * @brief Find similar blocks in the image using NEON
 *
 * Searches for blocks similar to the reference block within a search window.
 * Uses NEON-optimized distance computation for efficiency.
 *
 * @param image Input image
 * @param width Image width
 * @param height Image height
 * @param ref_x Reference block x coordinate
 * @param ref_y Reference block y coordinate
 * @param matches Output array of similar block matches
 */
void find_similar_blocks_neon(const float* image, int width, int height,
                            int ref_x, int ref_y, BlockMatch* matches) {
    const float* ref_block = &image[ref_y * width + ref_x];
    int match_count = 0;
    
    // Calculate search window boundaries
    int search_start_x = max(ref_x - SEARCH_WINDOW/2, 0);
    int search_end_x = min(ref_x + SEARCH_WINDOW/2, width - BLOCK_SIZE);
    int search_start_y = max(ref_y - SEARCH_WINDOW/2, 0);
    int search_end_y = min(ref_y + SEARCH_WINDOW/2, height - BLOCK_SIZE);
    
    // Search for similar blocks
    for (int y = search_start_y; y <= search_end_y; y++) {
        for (int x = search_start_x; x <= search_end_x; x++) {
            // Skip the reference block itself
            if (x == ref_x && y == ref_y) continue;
            
            // Compute block distance
            float dist = compute_block_distance_neon(&image[y * width + x], ref_block, width);
            
            // Store if distance is small enough
            if (dist < HARD_THRESHOLD) {
                matches[match_count].distance = dist;
                matches[match_count].x = x;
                matches[match_count].y = y;
                match_count++;
                
                if (match_count >= MAX_SIMILAR_BLOCKS) break;
            }
        }
        if (match_count >= MAX_SIMILAR_BLOCKS) break;
    }
}

/**
 * @brief Process a reference block and its similar blocks
 *
 * Main processing function for BM3D:
 * 1. Find similar blocks
 * 2. Apply 2D DCT to each block
 * 3. Apply collaborative filtering
 * 4. Apply inverse 2D DCT
 * 5. Aggregate results
 *
 * @param input Input noisy image
 * @param output Output denoised image
 * @param width Image width
 * @param height Image height
 * @param ref_x Reference block x coordinate
 * @param ref_y Reference block y coordinate
 * @param sigma Noise standard deviation
 */
void process_reference_block(const float* input, float* output,
                           int width, int height, int ref_x, int ref_y,
                           float sigma) {
    // Allocate memory for block processing
    float* block_group = (float*)malloc(BLOCK_SIZE * BLOCK_SIZE * MAX_SIMILAR_BLOCKS * sizeof(float));
    BlockMatch* matches = (BlockMatch*)malloc(MAX_SIMILAR_BLOCKS * sizeof(BlockMatch));
    
    // Find similar blocks
    find_similar_blocks_neon(input, width, height, ref_x, ref_y, matches);
    
    // Extract and transform similar blocks
    for (int i = 0; i < MAX_SIMILAR_BLOCKS; i++) {
        int x = matches[i].x;
        int y = matches[i].y;
        
        // Copy block to processing buffer
        for (int by = 0; by < BLOCK_SIZE; by++) {
            memcpy(&block_group[i * BLOCK_SIZE * BLOCK_SIZE + by * BLOCK_SIZE],
                   &input[y * width + x + by * width],
                   BLOCK_SIZE * sizeof(float));
        }
        
        // Apply 2D DCT
        dct_2d_8x8_neon(&block_group[i * BLOCK_SIZE * BLOCK_SIZE], BLOCK_SIZE);
    }
    
    // Apply collaborative filtering
    for (int i = 0; i < MAX_SIMILAR_BLOCKS; i++) {
        apply_wiener_filter_neon(&block_group[i * BLOCK_SIZE * BLOCK_SIZE],
                               sigma, BLOCK_SIZE);
    }
    
    // Inverse transform and aggregate
    for (int i = 0; i < MAX_SIMILAR_BLOCKS; i++) {
        idct_2d_8x8_neon(&block_group[i * BLOCK_SIZE * BLOCK_SIZE], BLOCK_SIZE);
        
        int x = matches[i].x;
        int y = matches[i].y;
        
        // Aggregate filtered block back to output
        for (int by = 0; by < BLOCK_SIZE; by++) {
            for (int bx = 0; bx < BLOCK_SIZE; bx++) {
                output[(y + by) * width + x + bx] +=
                    block_group[i * BLOCK_SIZE * BLOCK_SIZE + by * BLOCK_SIZE + bx];
            }
        }
    }
    
    // Clean up
    free(block_group);
    free(matches);
}

/**
 * @brief Thread function for parallel BM3D processing
 *
 * Processes a subset of the image rows assigned to this thread.
 *
 * @param arg Thread arguments (ThreadArgs structure)
 */
void* bm3d_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    // Process reference blocks in assigned rows
    for (int y = args->start_row; y < args->end_row; y += STEP_SIZE) {
        for (int x = 0; x < args->width; x += STEP_SIZE) {
            process_reference_block(args->input, args->output,
                                 args->width, args->height,
                                 x, y, args->sigma);
        }
    }
    
    return NULL;
}

/**
 * @brief Main BM3D denoising function using NEON
 *
 * Implements the complete BM3D algorithm with NEON optimization:
 * 1. Divides the image into blocks for parallel processing
 * 2. Creates threads to process blocks
 * 3. Combines results for final output
 *
 * @param input Input noisy image
 * @param output Output denoised image
 * @param width Image width
 * @param height Image height
 * @param sigma Noise standard deviation
 */
void bm3d_denoise_neon(const float* input, float* output, int width, int height, float sigma) {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];
    
    // Initialize output image
    memset(output, 0, width * height * sizeof(float));
    
    // Create threads for parallel processing
    int rows_per_thread = height / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].input = input;
        thread_args[i].output = output;
        thread_args[i].width = width;
        thread_args[i].height = height;
        thread_args[i].start_row = i * rows_per_thread;
        thread_args[i].end_row = (i == NUM_THREADS-1) ? height : (i+1) * rows_per_thread;
        thread_args[i].sigma = sigma;
        
        pthread_create(&threads[i], NULL, bm3d_thread, &thread_args[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Normalize output
    float scale = 1.0f / (BLOCK_SIZE * BLOCK_SIZE);
    for (int i = 0; i < width * height; i++) {
        output[i] *= scale;
    }
}
