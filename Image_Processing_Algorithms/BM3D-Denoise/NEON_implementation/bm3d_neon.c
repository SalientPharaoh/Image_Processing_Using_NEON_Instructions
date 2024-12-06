#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>
#include <float.h>

// Constants for BM3D algorithm
#define BLOCK_SIZE 8
#define SEARCH_WINDOW 39
#define MAX_SIMILAR_BLOCKS 16
#define HARD_THRESHOLD 2.7f
#define STEP_SIZE 3
#define NUM_THREADS 4

// Align memory to 32-byte boundary for better NEON performance
#define ALIGN32(x) ((((uintptr_t)(x) + 31) & ~31))

// Structure for block matching
typedef struct {
    float distance;
    int x;
    int y;
} BlockMatch;

// Thread argument structure
typedef struct {
    const float* input;
    float* output;
    int width;
    int height;
    int start_row;
    int end_row;
    float sigma;
} ThreadArgs;

// NEON-optimized block distance computation
float compute_block_distance_neon(const float* block1, const float* block2, int stride) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j += 4) {
            float32x4_t v1 = vld1q_f32(&block1[i * stride + j]);
            float32x4_t v2 = vld1q_f32(&block2[i * stride + j]);
            float32x4_t diff = vsubq_f32(v1, v2);
            sum_vec = vmlaq_f32(sum_vec, diff, diff);
        }
    }
    
    float sum = vaddvq_f32(sum_vec);
    return sum / (BLOCK_SIZE * BLOCK_SIZE);
}

// NEON-optimized 2D DCT for 8x8 blocks
void dct_2d_8x8_neon(float* block, int stride) {
    static const float32_t c1 = 0.98078528f;
    static const float32_t c2 = 0.92387953f;
    static const float32_t c3 = 0.83146961f;
    static const float32_t c4 = 0.70710678f;
    static const float32_t c5 = 0.55557023f;
    static const float32_t c6 = 0.38268343f;
    static const float32_t c7 = 0.19509032f;
    
    float32x4_t row_buf[8];
    
    // Row-wise DCT
    for (int i = 0; i < 8; i++) {
        // Load row
        float32x4_t row0 = vld1q_f32(&block[i * stride]);
        float32x4_t row1 = vld1q_f32(&block[i * stride + 4]);
        
        // Stage 1
        float32x4_t sum04 = vaddq_f32(row0, row1);
        float32x4_t diff04 = vsubq_f32(row0, row1);
        
        // Stage 2
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        // Stage 3
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        // Store intermediate results
        row_buf[i] = out0;
        row_buf[i + 4] = out1;
    }
    
    // Column-wise DCT using intermediate results
    for (int j = 0; j < 8; j++) {
        float32x4_t col = vld1q_f32((float*)&row_buf[j]);
        
        // Stage 1
        float32x4_t sum04 = vaddq_f32(col, vrev64q_f32(col));
        float32x4_t diff04 = vsubq_f32(col, vrev64q_f32(col));
        
        // Stage 2
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        // Stage 3
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        // Store final results
        vst1q_f32(&block[j * stride], out0);
        vst1q_f32(&block[j * stride + 4], out1);
    }
}

// NEON-optimized inverse 2D DCT for 8x8 blocks
void idct_2d_8x8_neon(float* block, int stride) {
    // Similar to dct_2d_8x8_neon but with inverse coefficients
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
        
        // Stage 1
        float32x4_t sum04 = vaddq_f32(row0, row1);
        float32x4_t diff04 = vsubq_f32(row0, row1);
        
        // Stage 2
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        // Stage 3
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        row_buf[i] = out0;
        row_buf[i + 4] = out1;
    }
    
    // Column-wise IDCT
    for (int j = 0; j < 8; j++) {
        float32x4_t col = vld1q_f32((float*)&row_buf[j]);
        
        // Stage 1
        float32x4_t sum04 = vaddq_f32(col, vrev64q_f32(col));
        float32x4_t diff04 = vsubq_f32(col, vrev64q_f32(col));
        
        // Stage 2
        float32x4_t temp0 = vaddq_f32(sum04, vrev64q_f32(sum04));
        float32x4_t temp1 = vmulq_n_f32(diff04, c4);
        
        // Stage 3
        float32x4_t out0 = vaddq_f32(temp0, temp1);
        float32x4_t out1 = vsubq_f32(temp0, temp1);
        
        vst1q_f32(&block[j * stride], out0);
        vst1q_f32(&block[j * stride + 4], out1);
    }
    
    // Scale the results
    float32x4_t scale = vdupq_n_f32(1.0f / 8.0f);
    for (int i = 0; i < 8; i++) {
        float32x4_t row0 = vld1q_f32(&block[i * stride]);
        float32x4_t row1 = vld1q_f32(&block[i * stride + 4]);
        vst1q_f32(&block[i * stride], vmulq_f32(row0, scale));
        vst1q_f32(&block[i * stride + 4], vmulq_f32(row1, scale));
    }
}

// NEON-optimized Wiener filter
void apply_wiener_filter_neon(float* block, float sigma, int stride) {
    float32x4_t sigma_sq = vdupq_n_f32(sigma * sigma);
    float32x4_t one = vdupq_n_f32(1.0f);
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j += 4) {
            float32x4_t coef = vld1q_f32(&block[i * stride + j]);
            float32x4_t coef_sq = vmulq_f32(coef, coef);
            float32x4_t wiener = vdivq_f32(coef_sq, vaddq_f32(coef_sq, sigma_sq));
            coef = vmulq_f32(coef, wiener);
            vst1q_f32(&block[i * stride + j], coef);
        }
    }
}

// Block matching function
void find_similar_blocks_neon(const float* image, int width, int height,
                            int ref_x, int ref_y, BlockMatch* matches) {
    const float* ref_block = &image[ref_y * width + ref_x];
    int match_count = 0;
    
    int search_start_x = max(ref_x - SEARCH_WINDOW/2, 0);
    int search_end_x = min(ref_x + SEARCH_WINDOW/2, width - BLOCK_SIZE);
    int search_start_y = max(ref_y - SEARCH_WINDOW/2, 0);
    int search_end_y = min(ref_y + SEARCH_WINDOW/2, height - BLOCK_SIZE);
    
    for (int y = search_start_y; y <= search_end_y; y += STEP_SIZE) {
        for (int x = search_start_x; x <= search_end_x; x += STEP_SIZE) {
            if (x == ref_x && y == ref_y) continue;
            
            const float* cur_block = &image[y * width + x];
            float distance = compute_block_distance_neon(ref_block, cur_block, width);
            
            if (match_count < MAX_SIMILAR_BLOCKS) {
                matches[match_count].distance = distance;
                matches[match_count].x = x;
                matches[match_count].y = y;
                match_count++;
            } else {
                // Replace the worst match if current is better
                int worst_idx = 0;
                float worst_dist = matches[0].distance;
                for (int i = 1; i < MAX_SIMILAR_BLOCKS; i++) {
                    if (matches[i].distance > worst_dist) {
                        worst_dist = matches[i].distance;
                        worst_idx = i;
                    }
                }
                if (distance < worst_dist) {
                    matches[worst_idx].distance = distance;
                    matches[worst_idx].x = x;
                    matches[worst_idx].y = y;
                }
            }
        }
    }
}

// Process one reference block and its similar blocks
void process_reference_block(const float* input, float* output,
                           int width, int height, int ref_x, int ref_y,
                           float sigma) {
    BlockMatch matches[MAX_SIMILAR_BLOCKS];
    float* group_3d = (float*)aligned_alloc(32, MAX_SIMILAR_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    float* weights = (float*)aligned_alloc(32, MAX_SIMILAR_BLOCKS * sizeof(float));
    
    // Find similar blocks
    find_similar_blocks_neon(input, width, height, ref_x, ref_y, matches);
    
    // Extract and transform similar blocks
    for (int i = 0; i < MAX_SIMILAR_BLOCKS; i++) {
        int x = matches[i].x;
        int y = matches[i].y;
        
        // Copy block to 3D group
        for (int by = 0; by < BLOCK_SIZE; by++) {
            for (int bx = 0; bx < BLOCK_SIZE; bx++) {
                group_3d[i * BLOCK_SIZE * BLOCK_SIZE + by * BLOCK_SIZE + bx] =
                    input[(y + by) * width + (x + bx)];
            }
        }
        
        // Apply 2D DCT
        dct_2d_8x8_neon(&group_3d[i * BLOCK_SIZE * BLOCK_SIZE], BLOCK_SIZE);
        
        // Calculate weight
        weights[i] = 1.0f / (matches[i].distance + 1e-6f);
    }
    
    // Collaborative filtering
    for (int i = 0; i < MAX_SIMILAR_BLOCKS; i++) {
        apply_wiener_filter_neon(&group_3d[i * BLOCK_SIZE * BLOCK_SIZE], sigma, BLOCK_SIZE);
        idct_2d_8x8_neon(&group_3d[i * BLOCK_SIZE * BLOCK_SIZE], BLOCK_SIZE);
    }
    
    // Aggregate filtered blocks
    float32x4_t weight_sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < MAX_SIMILAR_BLOCKS; i++) {
        int x = matches[i].x;
        int y = matches[i].y;
        float32x4_t weight = vdupq_n_f32(weights[i]);
        
        for (int by = 0; by < BLOCK_SIZE; by++) {
            for (int bx = 0; bx < BLOCK_SIZE; bx += 4) {
                float32x4_t block_val = vld1q_f32(&group_3d[i * BLOCK_SIZE * BLOCK_SIZE + by * BLOCK_SIZE + bx]);
                float32x4_t weighted_val = vmulq_f32(block_val, weight);
                float32x4_t curr_val = vld1q_f32(&output[(y + by) * width + (x + bx)]);
                vst1q_f32(&output[(y + by) * width + (x + bx)], vaddq_f32(curr_val, weighted_val));
            }
        }
        weight_sum = vaddq_f32(weight_sum, weight);
    }
    
    // Normalize
    float total_weight = vaddvq_f32(weight_sum);
    float32x4_t inv_weight = vdupq_n_f32(1.0f / total_weight);
    
    for (int by = 0; by < BLOCK_SIZE; by++) {
        for (int bx = 0; bx < BLOCK_SIZE; bx += 4) {
            float32x4_t val = vld1q_f32(&output[(ref_y + by) * width + (ref_x + bx)]);
            vst1q_f32(&output[(ref_y + by) * width + (ref_x + bx)], vmulq_f32(val, inv_weight));
        }
    }
    
    free(group_3d);
    free(weights);
}

// Thread function for parallel processing
void* bm3d_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    for (int y = args->start_row; y < args->end_row; y += STEP_SIZE) {
        for (int x = 0; x < args->width - BLOCK_SIZE; x += STEP_SIZE) {
            process_reference_block(args->input, args->output,
                                 args->width, args->height,
                                 x, y, args->sigma);
        }
    }
    
    return NULL;
}

// Main BM3D function
void bm3d_denoise_neon(const float* input, float* output, int width, int height, float sigma) {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];
    
    // Initialize output
    memset(output, 0, width * height * sizeof(float));
    
    // Create threads for parallel processing
    int rows_per_thread = height / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].input = input;
        thread_args[i].output = output;
        thread_args[i].width = width;
        thread_args[i].height = height;
        thread_args[i].start_row = i * rows_per_thread;
        thread_args[i].end_row = (i == NUM_THREADS - 1) ? height - BLOCK_SIZE : (i + 1) * rows_per_thread;
        thread_args[i].sigma = sigma;
        
        pthread_create(&threads[i], NULL, bm3d_thread, &thread_args[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}
