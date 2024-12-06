#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>

#define MAX_THREADS 8
#define BLOCK_SIZE 16

// Thread argument structure
typedef struct {
    const float* input;
    float* output;
    int width;
    int height;
    int start_row;
    int end_row;
    float h;
    int patch_size;
    int search_window;
} ThreadArgs;

// NEON-optimized patch distance computation
float compute_patch_distance_neon(const float* p1, const float* p2, 
                                int patch_size, int stride) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j += 4) {
            float32x4_t v1 = vld1q_f32(&p1[i * stride + j]);
            float32x4_t v2 = vld1q_f32(&p2[i * stride + j]);
            float32x4_t diff = vsubq_f32(v1, v2);
            sum_vec = vmlaq_f32(sum_vec, diff, diff);
        }
    }
    
    float sum = vaddvq_f32(sum_vec);
    return sum / (patch_size * patch_size);
}

// NEON-optimized weight computation
void compute_weights_neon(const float* distances, float* weights, 
                         int size, float h) {
    float32x4_t vh = vdupq_n_f32(-1.0f / (h * h));
    
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

// Process one block of the image
void process_block_neon(const float* input, float* output,
                       int width, int height, int x, int y,
                       float h, int patch_size, int search_window) {
    int half_patch = patch_size / 2;
    int half_search = search_window / 2;
    
    // Allocate memory for distances and weights
    int search_size = search_window * search_window;
    float* distances = (float*)aligned_alloc(32, search_size * sizeof(float));
    float* weights = (float*)aligned_alloc(32, search_size * sizeof(float));
    
    for (int by = y; by < min(y + BLOCK_SIZE, height - patch_size); by++) {
        for (int bx = x; bx < min(x + BLOCK_SIZE, width - patch_size); bx++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            float32x4_t weight_sum = vdupq_n_f32(0.0f);
            
            // Compute distances to all patches in search window
            int dist_idx = 0;
            for (int sy = max(by - half_search, half_patch); 
                 sy <= min(by + half_search, height - half_patch - 1); sy++) {
                for (int sx = max(bx - half_search, half_patch);
                     sx <= min(bx + half_search, width - half_patch - 1); sx++) {
                    if (sx == bx && sy == by) continue;
                    
                    distances[dist_idx++] = compute_patch_distance_neon(
                        &input[by * width + bx],
                        &input[sy * width + sx],
                        patch_size, width
                    );
                }
            }
            
            // Compute weights using NEON
            compute_weights_neon(distances, weights, dist_idx, h);
            
            // Apply weights and accumulate result
            dist_idx = 0;
            for (int sy = max(by - half_search, half_patch);
                 sy <= min(by + half_search, height - half_patch - 1); sy++) {
                for (int sx = max(bx - half_search, half_patch);
                     sx <= min(bx + half_search, width - half_patch - 1); sx++) {
                    if (sx == bx && sy == by) continue;
                    
                    float32x4_t vweight = vdupq_n_f32(weights[dist_idx]);
                    float32x4_t vpixel = vld1q_f32(&input[sy * width + sx]);
                    sum = vmlaq_f32(sum, vpixel, vweight);
                    weight_sum = vaddq_f32(weight_sum, vweight);
                    dist_idx++;
                }
            }
            
            // Normalize and store result
            float32x4_t vresult = vdivq_f32(sum, weight_sum);
            vst1q_f32(&output[by * width + bx], vresult);
        }
    }
    
    free(distances);
    free(weights);
}

// Thread function for parallel processing
void* nlm_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
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

// Main NLM denoising function
void nlm_denoise_neon(const float* input, float* output,
                     int width, int height,
                     float h, int patch_size, int search_window) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];
    
    // Initialize output
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
    
    // Wait for threads to complete
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}
