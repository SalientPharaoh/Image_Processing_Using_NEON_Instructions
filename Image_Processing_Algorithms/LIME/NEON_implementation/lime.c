#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

// Structure to hold image data
typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

// Load image from file (basic PPM format for demonstration)
Image* load_image(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    Image* img = (Image*)malloc(sizeof(Image));
    char header[100];
    fgets(header, sizeof(header), fp);  // P6
    fgets(header, sizeof(header), fp);  // dimensions
    sscanf(header, "%d %d", &img->width, &img->height);
    fgets(header, sizeof(header), fp);  // max value
    
    img->channels = 3;
    img->data = (unsigned char*)malloc(img->width * img->height * img->channels);
    fread(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
    return img;
}

// Save image to file (PPM format)
void save_image(const char* filename, Image* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
}

// NEON optimized spatial affinity kernel computation
void create_spatial_affinity_kernel_neon(float* kernel, int size, float spatial_sigma) {
    float32x4_t sigma_sq = vdupq_n_f32(spatial_sigma * spatial_sigma);
    float32x4_t half_size = vdupq_n_f32(size / 2.0f);
    
    for (int i = 0; i < size; i++) {
        float32x4_t i_vec = vdupq_n_f32(i);
        for (int j = 0; j < size; j += 4) {
            float32x4_t j_vec = {j + 0.0f, j + 1.0f, j + 2.0f, j + 3.0f};
            
            float32x4_t dx = vsubq_f32(i_vec, half_size);
            float32x4_t dy = vsubq_f32(j_vec, half_size);
            float32x4_t dist_sq = vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy));
            float32x4_t exp_term = vmulq_f32(dist_sq, vdupq_n_f32(-0.5f / (spatial_sigma * spatial_sigma)));
            
            // Fast exponential approximation using Taylor series
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

// NEON optimized Sobel filter
void sobel_filter_neon(const float* input, float* output, int width, int height, int direction) {
    const int VECTOR_SIZE = 4;
    float32x4_t kernel_x, kernel_y;
    
    if (direction == 0) {  // Horizontal
        kernel_x = vdupq_n_f32(-1.0f);
        kernel_y = vdupq_n_f32(1.0f);
    } else {  // Vertical
        kernel_x = vdupq_n_f32(-1.0f);
        kernel_y = vdupq_n_f32(1.0f);
    }
    
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - VECTOR_SIZE - 1; j += VECTOR_SIZE) {
            float32x4_t center = vld1q_f32(&input[i * width + j]);
            float32x4_t left = vld1q_f32(&input[i * width + j - 1]);
            float32x4_t right = vld1q_f32(&input[i * width + j + 1]);
            float32x4_t top = vld1q_f32(&input[(i - 1) * width + j]);
            float32x4_t bottom = vld1q_f32(&input[(i + 1) * width + j]);
            
            float32x4_t grad;
            if (direction == 0) {
                grad = vsubq_f32(right, left);
            } else {
                grad = vsubq_f32(bottom, top);
            }
            
            vst1q_f32(&output[i * width + j], grad);
        }
    }
}

// NEON optimized smoothness weights computation
void compute_smoothness_weights_neon(const float* L, float* weights, int width, int height, 
                                   int direction, const float* kernel, int kernel_size, float eps) {
    float* gradient = (float*)malloc(width * height * sizeof(float));
    sobel_filter_neon(L, gradient, width, height, direction);
    
    const int half_k = kernel_size / 2;
    float32x4_t eps_vec = vdupq_n_f32(eps);
    
    for (int i = half_k; i < height - half_k; i++) {
        for (int j = half_k; j < width - half_k; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            float32x4_t sum_kernel = vdupq_n_f32(0.0f);
            
            for (int ki = -half_k; ki <= half_k; ki++) {
                for (int kj = -half_k; kj <= half_k; kj++) {
                    float32x4_t k_val = vld1q_f32(&kernel[(ki + half_k) * kernel_size + (kj + half_k)]);
                    float32x4_t g_val = vld1q_f32(&gradient[(i + ki) * width + (j + kj)]);
                    sum = vaddq_f32(sum, vmulq_f32(k_val, g_val));
                    sum_kernel = vaddq_f32(sum_kernel, k_val);
                }
            }
            
            float32x4_t abs_sum = vabsq_f32(sum);
            float32x4_t denom = vaddq_f32(abs_sum, eps_vec);
            float32x4_t w = vdivq_f32(sum_kernel, denom);
            
            vst1q_f32(&weights[i * width + j], w);
        }
    }
    
    free(gradient);
}

// NEON optimized sparse matrix solver using Conjugate Gradient method
void solve_linear_system_neon(const float* A_diag, const float* A_offdiag, 
                            const float* b, float* x, int size, int max_iter) {
    float* r = (float*)malloc(size * sizeof(float));
    float* p = (float*)malloc(size * sizeof(float));
    float* Ap = (float*)malloc(size * sizeof(float));
    
    // Initialize
    memcpy(r, b, size * sizeof(float));
    memcpy(p, r, size * sizeof(float));
    
    float32x4_t eps = vdupq_n_f32(1e-6);
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p
        for (int i = 0; i < size; i += 4) {
            float32x4_t pi = vld1q_f32(&p[i]);
            float32x4_t diag = vld1q_f32(&A_diag[i]);
            float32x4_t result = vmulq_f32(diag, pi);
            
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
        
        // Compute alpha = (r'r)/(p'Ap)
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
        
        // Update x and r
        float32x4_t alpha_vec = vdupq_n_f32(alpha);
        for (int i = 0; i < size; i += 4) {
            float32x4_t xi = vld1q_f32(&x[i]);
            float32x4_t pi = vld1q_f32(&p[i]);
            float32x4_t ri = vld1q_f32(&r[i]);
            float32x4_t Api = vld1q_f32(&Ap[i]);
            
            xi = vaddq_f32(xi, vmulq_f32(alpha_vec, pi));
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
        if (vgetq_lane_f32(r_norm, 0) < 1e-6) break;
        
        // Update p
        float beta = vaddvq_f32(r_norm) / rr;
        float32x4_t beta_vec = vdupq_n_f32(beta);
        for (int i = 0; i < size; i += 4) {
            float32x4_t ri = vld1q_f32(&r[i]);
            float32x4_t pi = vld1q_f32(&p[i]);
            pi = vaddq_f32(ri, vmulq_f32(beta_vec, pi));
            vst1q_f32(&p[i], pi);
        }
    }
    
    free(r);
    free(p);
    free(Ap);
}

// Main LIME enhancement function using NEON
void enhance_image_lime_neon(const Image* input, Image* output, float gamma, float lambda) {
    int width = input->width;
    int height = input->height;
    int size = width * height;
    
    // Allocate memory for intermediate results
    float* L = (float*)malloc(size * sizeof(float));
    float* L_refined = (float*)malloc(size * sizeof(float));
    float* kernel = (float*)malloc(15 * 15 * sizeof(float));
    
    // Create spatial affinity kernel
    create_spatial_affinity_kernel_neon(kernel, 15, 3.0f);
    
    // Initial illumination map estimation
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 4) {
            int idx = i * width + j;
            float32x4x3_t rgb = vld3q_f32((float*)&input->data[idx * 3]);
            float32x4_t max_val = vmaxq_f32(vmaxq_f32(rgb.val[0], rgb.val[1]), rgb.val[2]);
            vst1q_f32(&L[idx], max_val);
        }
    }
    
    // Compute smoothness weights and build linear system
    float* wx = (float*)malloc(size * sizeof(float));
    float* wy = (float*)malloc(size * sizeof(float));
    compute_smoothness_weights_neon(L, wx, width, height, 0, kernel, 15, 1e-3f);
    compute_smoothness_weights_neon(L, wy, width, height, 1, kernel, 15, 1e-3f);
    
    // Build and solve linear system
    float* A_diag = (float*)malloc(size * sizeof(float));
    float* A_offdiag = (float*)malloc((size-1) * sizeof(float));
    float* b = (float*)malloc(size * sizeof(float));
    
    // Build system matrix (simplified 5-point stencil)
    for (int i = 0; i < size; i++) {
        A_diag[i] = 1.0f + lambda * (wx[i] + wy[i]);
        if (i < size - 1) {
            A_offdiag[i] = -lambda * sqrtf(wx[i] * wx[i+1] + wy[i] * wy[i+1]);
        }
        b[i] = L[i];
    }
    
    // Solve system
    solve_linear_system_neon(A_diag, A_offdiag, b, L_refined, size, 100);
    
    // Apply gamma correction and enhance image
    float32x4_t gamma_vec = vdupq_n_f32(gamma);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 4) {
            int idx = i * width + j;
            
            // Load illumination map values
            float32x4_t l = vld1q_f32(&L_refined[idx]);
            l = pow_ps(l, gamma_vec);  // Apply gamma correction
            
            // Load and process RGB channels
            float32x4x3_t rgb = vld3q_f32((float*)&input->data[idx * 3]);
            rgb.val[0] = vdivq_f32(rgb.val[0], l);
            rgb.val[1] = vdivq_f32(rgb.val[1], l);
            rgb.val[2] = vdivq_f32(rgb.val[2], l);
            
            // Convert to uint8 and store
            uint8x8x3_t result;
            result.val[0] = vqmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(rgb.val[0])), 
                                                   vmovn_u32(vcvtq_u32_f32(vdupq_n_f32(0.0f)))));
            result.val[1] = vqmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(rgb.val[1])), 
                                                   vmovn_u32(vcvtq_u32_f32(vdupq_n_f32(0.0f)))));
            result.val[2] = vqmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(rgb.val[2])), 
                                                   vmovn_u32(vcvtq_u32_f32(vdupq_n_f32(0.0f)))));
            
            vst3_u8(&output->data[idx * 3], result);
        }
    }
    
    // Clean up
    free(L);
    free(L_refined);
    free(kernel);
    free(wx);
    free(wy);
    free(A_diag);
    free(A_offdiag);
    free(b);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return -1;
    }
    
    // Load input image
    Image* input = load_image(argv[1]);
    if (!input) {
        printf("Error: Cannot load image %s\n", argv[1]);
        return -1;
    }
    
    // Create output image
    Image* output = (Image*)malloc(sizeof(Image));
    output->width = input->width;
    output->height = input->height;
    output->channels = input->channels;
    output->data = (unsigned char*)malloc(output->width * output->height * output->channels);
    
    // Enhance image
    enhance_image_lime_neon(input, output, 0.6f, 0.15f);
    
    // Save output image
    save_image(argv[2], output);
    printf("Enhanced image saved as %s\n", argv[2]);
    
    // Clean up
    free(input->data);
    free(input);
    free(output->data);
    free(output);
    
    return 0;
}