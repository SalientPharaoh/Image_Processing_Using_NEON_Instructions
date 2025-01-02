/**
 * @file lime.c
 * @brief NEON-optimized implementation of Low-light IMage Enhancement (LIME)
 *
 * This implementation provides a high-performance version of the LIME algorithm
 * using ARM NEON SIMD instructions. The algorithm enhances low-light images by:
 * 1. Initial illumination map estimation
 * 2. Structure-aware smoothing using weighted optimization
 * 3. Final enhancement using the refined illumination map
 *
 * Key optimizations include:
 * - NEON SIMD vectorization for core operations
 * - Cache-efficient matrix operations
 * - Fast approximate math functions
 * - Parallel processing where applicable
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

/**
 * @brief Structure to hold image data
 */
typedef struct {
    unsigned char* data;    // Raw image data (RGB interleaved)
    int width;             // Image width in pixels
    int height;            // Image height in pixels
    int channels;          // Number of color channels (typically 3 for RGB)
} Image;

/**
 * @brief Load image from PPM file
 *
 * Reads a PPM format image file and creates an Image structure.
 * Note: This is a basic implementation for demonstration.
 *
 * @param filename Path to the input PPM file
 * @return Image* Pointer to the loaded image, or NULL on failure
 */
Image* load_image(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    Image* img = (Image*)malloc(sizeof(Image));
    char header[100];
    fgets(header, sizeof(header), fp);  // P6 format identifier
    fgets(header, sizeof(header), fp);  // Width and height
    sscanf(header, "%d %d", &img->width, &img->height);
    fgets(header, sizeof(header), fp);  // Maximum value
    
    img->channels = 3;  // RGB format
    img->data = (unsigned char*)malloc(img->width * img->height * img->channels);
    fread(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
    return img;
}

/**
 * @brief Save image to PPM file
 *
 * Writes an Image structure to a PPM format file.
 *
 * @param filename Output file path
 * @param img Image to save
 */
void save_image(const char* filename, Image* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
}

/**
 * @brief Create spatial affinity kernel using NEON
 *
 * Computes a Gaussian spatial affinity kernel for structure-aware smoothing.
 * Uses NEON SIMD to compute 4 kernel values at once and implements a fast
 * exponential approximation using Taylor series.
 *
 * @param kernel Output kernel buffer (size x size)
 * @param size Kernel size (should be odd)
 * @param spatial_sigma Spatial standard deviation
 */
void create_spatial_affinity_kernel_neon(float* kernel, int size, float spatial_sigma) {
    // Precompute constants for NEON processing
    float32x4_t sigma_sq = vdupq_n_f32(spatial_sigma * spatial_sigma);
    float32x4_t half_size = vdupq_n_f32(size / 2.0f);
    
    for (int i = 0; i < size; i++) {
        float32x4_t i_vec = vdupq_n_f32(i);
        for (int j = 0; j < size; j += 4) {
            // Create vector of j indices [j, j+1, j+2, j+3]
            float32x4_t j_vec = {j + 0.0f, j + 1.0f, j + 2.0f, j + 3.0f};
            
            // Compute squared distances from kernel center
            float32x4_t dx = vsubq_f32(i_vec, half_size);
            float32x4_t dy = vsubq_f32(j_vec, half_size);
            float32x4_t dist_sq = vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy));
            float32x4_t exp_term = vmulq_f32(dist_sq, vdupq_n_f32(-0.5f / (spatial_sigma * spatial_sigma)));
            
            // Fast exponential approximation using Taylor series
            // exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
            float32x4_t one = vdupq_n_f32(1.0f);
            float32x4_t result = vaddq_f32(one, exp_term);
            float32x4_t term = exp_term;
            for(int k = 2; k <= 5; k++) {
                term = vmulq_f32(term, vdivq_f32(exp_term, vdupq_n_f32(k)));
                result = vaddq_f32(result, term);
            }
            
            vst1q_f32(&kernel[i * size + j], result);
        }
    }
}

/**
 * @brief Apply Sobel filter using NEON
 *
 * Computes image gradients using the Sobel operator. Optimized with
 * NEON SIMD to process 4 pixels at once.
 *
 * @param input Input image
 * @param output Output gradient map
 * @param width Image width
 * @param height Image height
 * @param direction 0 for horizontal, 1 for vertical gradient
 */
void sobel_filter_neon(const float* input, float* output, int width, int height, int direction) {
    const int VECTOR_SIZE = 4;
    float32x4_t kernel_x, kernel_y;
    
    // Set up Sobel kernels based on direction
    if (direction == 0) {  // Horizontal gradient
        kernel_x = vdupq_n_f32(-1.0f);
        kernel_y = vdupq_n_f32(1.0f);
    } else {  // Vertical gradient
        kernel_x = vdupq_n_f32(-1.0f);
        kernel_y = vdupq_n_f32(1.0f);
    }
    
    // Process image with NEON, 4 pixels at a time
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - VECTOR_SIZE - 1; j += VECTOR_SIZE) {
            // Load pixel neighborhoods
            float32x4_t center = vld1q_f32(&input[i * width + j]);
            float32x4_t left = vld1q_f32(&input[i * width + j - 1]);
            float32x4_t right = vld1q_f32(&input[i * width + j + 1]);
            float32x4_t top = vld1q_f32(&input[(i - 1) * width + j]);
            float32x4_t bottom = vld1q_f32(&input[(i + 1) * width + j]);
            
            // Compute gradient based on direction
            float32x4_t grad;
            if (direction == 0) {
                grad = vsubq_f32(right, left);  // Horizontal gradient
            } else {
                grad = vsubq_f32(bottom, top);  // Vertical gradient
            }
            
            vst1q_f32(&output[i * width + j], grad);
        }
    }
}

/**
 * @brief Compute smoothness weights using NEON
 *
 * Calculates structure-aware smoothness weights based on image gradients
 * and spatial affinity. Uses NEON SIMD for efficient computation.
 *
 * @param L Input illumination map
 * @param weights Output weights
 * @param width Image width
 * @param height Image height
 * @param direction Gradient direction (0=horizontal, 1=vertical)
 * @param kernel Spatial affinity kernel
 * @param kernel_size Size of the affinity kernel
 * @param eps Small constant to prevent division by zero
 */
void compute_smoothness_weights_neon(const float* L, float* weights, int width, int height, 
                                   int direction, const float* kernel, int kernel_size, float eps) {
    // Compute image gradients
    float* gradient = (float*)malloc(width * height * sizeof(float));
    sobel_filter_neon(L, gradient, width, height, direction);
    
    const int half_k = kernel_size / 2;
    float32x4_t eps_vec = vdupq_n_f32(eps);
    
    // Process image with NEON, 4 pixels at a time
    for (int i = half_k; i < height - half_k; i++) {
        for (int j = half_k; j < width - half_k; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            float32x4_t sum_kernel = vdupq_n_f32(0.0f);
            
            // Apply spatial kernel to local neighborhood
            for (int ki = -half_k; ki <= half_k; ki++) {
                for (int kj = -half_k; kj <= half_k; kj++) {
                    float32x4_t k_val = vld1q_f32(&kernel[(ki + half_k) * kernel_size + (kj + half_k)]);
                    float32x4_t g_val = vld1q_f32(&gradient[(i + ki) * width + (j + kj)]);
                    sum = vaddq_f32(sum, vmulq_f32(k_val, g_val));
                    sum_kernel = vaddq_f32(sum_kernel, k_val);
                }
            }
            
            // Compute final weights with regularization
            float32x4_t abs_sum = vabsq_f32(sum);
            float32x4_t denom = vaddq_f32(abs_sum, eps_vec);
            float32x4_t w = vdivq_f32(sum_kernel, denom);
            
            vst1q_f32(&weights[i * width + j], w);
        }
    }
    
    free(gradient);
}

/**
 * @brief Solve sparse linear system using Conjugate Gradient method with NEON
 *
 * Implements the Conjugate Gradient method for solving Ax = b where A is
 * a sparse matrix. Uses NEON SIMD to accelerate vector operations.
 *
 * @param A_diag Diagonal elements of matrix A
 * @param A_offdiag Off-diagonal elements of matrix A
 * @param b Right-hand side vector
 * @param x Solution vector (initial guess and output)
 * @param size System size
 * @param max_iter Maximum iterations
 */
void solve_linear_system_neon(const float* A_diag, const float* A_offdiag, 
                            const float* b, float* x, int size, int max_iter) {
    // Allocate vectors for CG method
    float* r = (float*)malloc(size * sizeof(float));  // Residual
    float* p = (float*)malloc(size * sizeof(float));  // Search direction
    float* Ap = (float*)malloc(size * sizeof(float)); // Matrix-vector product
    
    // Initialize vectors
    memcpy(r, b, size * sizeof(float));
    memcpy(p, r, size * sizeof(float));
    
    float32x4_t eps = vdupq_n_f32(1e-6);
    
    // Main CG iteration loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute matrix-vector product Ap = A*p using NEON
        for (int i = 0; i < size; i += 4) {
            float32x4_t pi = vld1q_f32(&p[i]);
            float32x4_t diag = vld1q_f32(&A_diag[i]);
            float32x4_t result = vmulq_f32(diag, pi);
            
            // Handle off-diagonal elements
            if (i > 0) {
                float32x4_t off_prev = vld1q_f32(&A_offdiag[i-1]);
                float32x4_t p_prev = vld1q_f32(&p[i-1]);
                result = vaddq_f32(result, vmulq_f32(off_prev, p_prev));
            }
            if (i < size - 4) {
                float32x4_t off_next = vld1q_f32(&A_offdiag[i]);
                float32x4_t p_next = vld1q_f32(&p[i+1]);
                result = vaddq_f32(result, vmulq_f32(off_next, p_next));
            }
            
            vst1q_f32(&Ap[i], result);
        }
        
        // Compute step size alpha = (r'r)/(p'Ap)
        float32x4_t rr_sum = vdupq_n_f32(0.0f);
        float32x4_t pAp_sum = vdupq_n_f32(0.0f);
        for (int i = 0; i < size; i += 4) {
            float32x4_t ri = vld1q_f32(&r[i]);
            float32x4_t pi = vld1q_f32(&p[i]);
            float32x4_t Api = vld1q_f32(&Ap[i]);
            rr_sum = vaddq_f32(rr_sum, vmulq_f32(ri, ri));
            pAp_sum = vaddq_f32(pAp_sum, vmulq_f32(pi, Api));
        }
        float rr = vaddvq_f32(rr_sum);
        float pAp = vaddvq_f32(pAp_sum);
        float alpha = rr / pAp;
        
        // Update solution and residual
        float32x4_t alpha_vec = vdupq_n_f32(alpha);
        for (int i = 0; i < size; i += 4) {
            float32x4_t xi = vld1q_f32(&x[i]);
            float32x4_t pi = vld1q_f32(&p[i]);
            float32x4_t ri = vld1q_f32(&r[i]);
            float32x4_t Api = vld1q_f32(&Ap[i]);
            
            // x = x + alpha*p
            xi = vaddq_f32(xi, vmulq_f32(alpha_vec, pi));
            // r = r - alpha*Ap
            ri = vsubq_f32(ri, vmulq_f32(alpha_vec, Api));
            
            vst1q_f32(&x[i], xi);
            vst1q_f32(&r[i], ri);
        }
        
        // Check convergence
        float32x4_t r_norm = vdupq_n_f32(0.0f);
        for (int i = 0; i < size; i += 4) {
            float32x4_t ri = vld1q_f32(&r[i]);
            r_norm = vaddq_f32(r_norm, vmulq_f32(ri, ri));
        }
        if (vaddvq_f32(r_norm) < 1e-6) break;
    }
    
    // Clean up
    free(r);
    free(p);
    free(Ap);
}

/**
 * @brief Main LIME enhancement function using NEON
 *
 * Implements the complete LIME algorithm with NEON optimizations:
 * 1. Initial illumination estimation
 * 2. Structure-aware smoothing
 * 3. Final enhancement
 *
 * @param input Input low-light image
 * @param output Enhanced output image
 * @param gamma Gamma correction parameter
 * @param lambda Smoothing strength
 */
void enhance_image_lime_neon(const Image* input, Image* output, float gamma, float lambda) {
    int width = input->width;
    int height = input->height;
    int size = width * height;
    
    // Allocate memory for intermediate results
    float* L = (float*)malloc(size * sizeof(float));
    float* T = (float*)malloc(size * sizeof(float));
    float* weights_x = (float*)malloc(size * sizeof(float));
    float* weights_y = (float*)malloc(size * sizeof(float));
    
    // Step 1: Initial illumination estimation
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            int rgb_idx = idx * 3;
            // Find maximum RGB value as initial illumination
            float max_val = fmaxf(input->data[rgb_idx],
                                fmaxf(input->data[rgb_idx + 1],
                                     input->data[rgb_idx + 2])) / 255.0f;
            L[idx] = max_val;
        }
    }
    
    // Step 2: Structure-aware smoothing
    // Create spatial affinity kernel
    int kernel_size = 15;
    float* kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));
    create_spatial_affinity_kernel_neon(kernel, kernel_size, 3.0f);
    
    // Compute smoothness weights
    compute_smoothness_weights_neon(L, weights_x, width, height, 0, kernel, kernel_size, 0.0001f);
    compute_smoothness_weights_neon(L, weights_y, width, height, 1, kernel, kernel_size, 0.0001f);
    
    // Solve optimization problem
    solve_linear_system_neon(weights_x, weights_y, L, T, size, 50);
    
    // Step 3: Final enhancement
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            int rgb_idx = idx * 3;
            
            // Apply gamma correction and enhance each channel
            float t = powf(T[idx], gamma);
            for (int c = 0; c < 3; c++) {
                float enhanced = input->data[rgb_idx + c] / (t * 255.0f);
                output->data[rgb_idx + c] = (unsigned char)(fminf(enhanced * 255.0f, 255.0f));
            }
        }
    }
    
    // Clean up
    free(L);
    free(T);
    free(weights_x);
    free(weights_y);
    free(kernel);
}

/**
 * @brief Main function
 *
 * Handles command-line arguments and demonstrates LIME enhancement.
 */
int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage: %s input_image output_image gamma lambda\n", argv[0]);
        return 1;
    }
    
    // Load input image
    Image* input = load_image(argv[1]);
    if (!input) {
        printf("Error loading input image\n");
        return 1;
    }
    
    // Create output image
    Image* output = (Image*)malloc(sizeof(Image));
    output->width = input->width;
    output->height = input->height;
    output->channels = input->channels;
    output->data = (unsigned char*)malloc(output->width * output->height * output->channels);
    
    // Parse parameters
    float gamma = atof(argv[3]);
    float lambda = atof(argv[4]);
    
    // Enhance image
    enhance_image_lime_neon(input, output, gamma, lambda);
    
    // Save result
    save_image(argv[2], output);
    
    // Clean up
    free(input->data);
    free(input);
    free(output->data);
    free(output);
    
    return 0;
}