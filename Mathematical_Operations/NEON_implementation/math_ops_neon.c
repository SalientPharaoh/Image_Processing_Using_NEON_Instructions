#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <math.h>
#include <string.h>

// Optimized NEON vector multiplication
void multiply_vectors_neon(const float* a, const float* b, float* result, int size) {
    // Process 16 elements at a time using NEON
    for (int i = 0; i <= size - 16; i += 16) {
        float32x4x4_t va = vld1q_f32_x4(&a[i]);
        float32x4x4_t vb = vld1q_f32_x4(&b[i]);
        
        float32x4x4_t vresult;
        vresult.val[0] = vmulq_f32(va.val[0], vb.val[0]);
        vresult.val[1] = vmulq_f32(va.val[1], vb.val[1]);
        vresult.val[2] = vmulq_f32(va.val[2], vb.val[2]);
        vresult.val[3] = vmulq_f32(va.val[3], vb.val[3]);
        
        vst1q_f32_x4(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (int i = (size & ~15); i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

// Optimized NEON vector division
void divide_vectors_neon(const float* a, const float* b, float* result, int size) {
    // Process 16 elements at a time using NEON
    for (int i = 0; i <= size - 16; i += 16) {
        float32x4x4_t va = vld1q_f32_x4(&a[i]);
        float32x4x4_t vb = vld1q_f32_x4(&b[i]);
        
        float32x4x4_t vresult;
        // Use reciprocal approximation for faster division
        float32x4_t recip0 = vrecpeq_f32(vb.val[0]);
        float32x4_t recip1 = vrecpeq_f32(vb.val[1]);
        float32x4_t recip2 = vrecpeq_f32(vb.val[2]);
        float32x4_t recip3 = vrecpeq_f32(vb.val[3]);
        
        // Refine approximation (Newton-Raphson)
        recip0 = vmulq_f32(vrecpsq_f32(vb.val[0], recip0), recip0);
        recip1 = vmulq_f32(vrecpsq_f32(vb.val[1], recip1), recip1);
        recip2 = vmulq_f32(vrecpsq_f32(vb.val[2], recip2), recip2);
        recip3 = vmulq_f32(vrecpsq_f32(vb.val[3], recip3), recip3);
        
        vresult.val[0] = vmulq_f32(va.val[0], recip0);
        vresult.val[1] = vmulq_f32(va.val[1], recip1);
        vresult.val[2] = vmulq_f32(va.val[2], recip2);
        vresult.val[3] = vmulq_f32(va.val[3], recip3);
        
        vst1q_f32_x4(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (int i = (size & ~15); i < size; i++) {
        result[i] = a[i] / b[i];
    }
}

// Optimized NEON vector normalization
void normalize_vector_neon(float* vector, int size) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    // Calculate sum of squares using NEON
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t v = vld1q_f32(&vector[i]);
        sum_vec = vmlaq_f32(sum_vec, v, v);
    }
    
    // Reduce sum
    float sum = vaddvq_f32(sum_vec);
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        sum += vector[i] * vector[i];
    }
    
    // Calculate reciprocal square root
    float32x4_t rsqrt = vrsqrteq_f32(vdupq_n_f32(sum));
    rsqrt = vmulq_f32(vrsqrtsq_f32(vmulq_f32(rsqrt, rsqrt), vdupq_n_f32(sum)), rsqrt);
    float scale = vgetq_lane_f32(rsqrt, 0);
    
    // Normalize using NEON
    for (int i = 0; i <= size - 16; i += 16) {
        float32x4x4_t v = vld1q_f32_x4(&vector[i]);
        v.val[0] = vmulq_n_f32(v.val[0], scale);
        v.val[1] = vmulq_n_f32(v.val[1], scale);
        v.val[2] = vmulq_n_f32(v.val[2], scale);
        v.val[3] = vmulq_n_f32(v.val[3], scale);
        vst1q_f32_x4(&vector[i], v);
    }
    
    // Handle remaining elements
    for (int i = (size & ~15); i < size; i++) {
        vector[i] *= scale;
    }
}

// Optimized NEON vector denormalization
void denormalize_vector_neon(float* vector, float scale, int size) {
    float32x4_t vscale = vdupq_n_f32(scale);
    
    // Process 16 elements at a time
    for (int i = 0; i <= size - 16; i += 16) {
        float32x4x4_t v = vld1q_f32_x4(&vector[i]);
        v.val[0] = vmulq_f32(v.val[0], vscale);
        v.val[1] = vmulq_f32(v.val[1], vscale);
        v.val[2] = vmulq_f32(v.val[2], vscale);
        v.val[3] = vmulq_f32(v.val[3], vscale);
        vst1q_f32_x4(&vector[i], v);
    }
    
    // Handle remaining elements
    for (int i = (size & ~15); i < size; i++) {
        vector[i] *= scale;
    }
}

// Advanced mathematical operations using NEON

// Exponential approximation using NEON
void exp_vector_neon(const float* input, float* output, int size) {
    // Constants for exp approximation
    float32x4_t v_1 = vdupq_n_f32(1.0f);
    float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    float32x4_t v_ln2 = vdupq_n_f32(0.693147181f);
    float32x4_t v_1_div_ln2 = vdupq_n_f32(1.442695041f);
    
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        
        // exp(x) = 2^(x/ln(2))
        float32x4_t tx = vmulq_f32(x, v_1_div_ln2);
        
        // Split into integer and fractional parts
        int32x4_t n = vcvtq_s32_f32(tx);
        float32x4_t f = vsubq_f32(tx, vcvtq_f32_s32(n));
        
        // Polynomial approximation for 2^f where 0 <= f < 1
        float32x4_t p = vaddq_f32(v_1, vmulq_f32(f, vaddq_f32(v_1, vmulq_f32(f, v_0_5))));
        
        // Combine with integer power
        float32x4_t result = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n, vreinterpretq_s32_f32(p)), 23));
        
        vst1q_f32(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        output[i] = expf(input[i]);
    }
}

// Logarithm approximation using NEON
void log_vector_neon(const float* input, float* output, int size) {
    float32x4_t v_1 = vdupq_n_f32(1.0f);
    float32x4_t v_ln2 = vdupq_n_f32(0.693147181f);
    
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        
        // Extract exponent and mantissa
        int32x4_t exp = vshrq_n_s32(vreinterpretq_s32_f32(x), 23);
        float32x4_t mantissa = vreinterpretq_f32_s32(vorrq_s32(
            vandq_s32(vreinterpretq_s32_f32(x), vdupq_n_s32(0x007FFFFF)),
            vreinterpretq_s32_f32(v_1)
        ));
        
        // Polynomial approximation for log(1+x) where -1 < x < 1
        float32x4_t y = vsubq_f32(mantissa, v_1);
        float32x4_t y2 = vmulq_f32(y, y);
        float32x4_t p = vaddq_f32(y, vmulq_f32(y2, vdupq_n_f32(-0.5f)));
        
        // Combine with exponent
        float32x4_t result = vaddq_f32(
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(exp, vdupq_n_s32(127))), v_ln2),
            p
        );
        
        vst1q_f32(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        output[i] = logf(input[i]);
    }
}

// Power function using NEON (x^y)
void pow_vector_neon(const float* x, const float* y, float* output, int size) {
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        float32x4_t vy = vld1q_f32(&y[i]);
        
        // log(x)
        float32x4_t log_x = vdupq_n_f32(0.0f);
        log_vector_neon(&vgetq_lane_f32(vx, 0), &vgetq_lane_f32(log_x, 0), 4);
        
        // y * log(x)
        float32x4_t prod = vmulq_f32(vy, log_x);
        
        // exp(y * log(x))
        float32x4_t result = vdupq_n_f32(0.0f);
        exp_vector_neon(&vgetq_lane_f32(prod, 0), &vgetq_lane_f32(result, 0), 4);
        
        vst1q_f32(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        output[i] = powf(x[i], y[i]);
    }
}

// Trigonometric functions using NEON approximations

// Sine approximation using NEON
void sin_vector_neon(const float* input, float* output, int size) {
    float32x4_t v_pi = vdupq_n_f32(M_PI);
    float32x4_t v_2pi = vdupq_n_f32(2.0f * M_PI);
    
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        
        // Normalize to [-pi, pi]
        float32x4_t normalized = vsubq_f32(x, vmulq_f32(
            v_2pi,
            vcvtq_f32_s32(vcvtq_s32_f32(vdivq_f32(x, v_2pi)))
        ));
        
        // Taylor series approximation
        float32x4_t x2 = vmulq_f32(normalized, normalized);
        float32x4_t x3 = vmulq_f32(x2, normalized);
        float32x4_t x5 = vmulq_f32(x3, x2);
        float32x4_t x7 = vmulq_f32(x5, x2);
        
        float32x4_t result = vsubq_f32(
            normalized,
            vaddq_f32(
                vmulq_f32(x3, vdupq_n_f32(1.0f/6.0f)),
                vsubq_f32(
                    vmulq_f32(x5, vdupq_n_f32(1.0f/120.0f)),
                    vmulq_f32(x7, vdupq_n_f32(1.0f/5040.0f))
                )
            )
        );
        
        vst1q_f32(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        output[i] = sinf(input[i]);
    }
}

// Cosine approximation using NEON
void cos_vector_neon(const float* input, float* output, int size) {
    float32x4_t v_pi = vdupq_n_f32(M_PI);
    float32x4_t v_2pi = vdupq_n_f32(2.0f * M_PI);
    float32x4_t v_pi_2 = vdupq_n_f32(M_PI_2);
    
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        
        // Normalize to [-pi, pi] and shift by pi/2 for cosine
        float32x4_t normalized = vsubq_f32(
            vaddq_f32(x, v_pi_2),
            vmulq_f32(
                v_2pi,
                vcvtq_f32_s32(vcvtq_s32_f32(vdivq_f32(vaddq_f32(x, v_pi_2), v_2pi)))
            )
        );
        
        // Use sine approximation for cosine
        float32x4_t x2 = vmulq_f32(normalized, normalized);
        float32x4_t x3 = vmulq_f32(x2, normalized);
        float32x4_t x5 = vmulq_f32(x3, x2);
        float32x4_t x7 = vmulq_f32(x5, x2);
        
        float32x4_t result = vsubq_f32(
            normalized,
            vaddq_f32(
                vmulq_f32(x3, vdupq_n_f32(1.0f/6.0f)),
                vsubq_f32(
                    vmulq_f32(x5, vdupq_n_f32(1.0f/120.0f)),
                    vmulq_f32(x7, vdupq_n_f32(1.0f/5040.0f))
                )
            )
        );
        
        vst1q_f32(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size & ~3); i < size; i++) {
        output[i] = cosf(input[i]);
    }
}

// Matrix operations using NEON

// Matrix multiplication using NEON
void matrix_multiply_neon(const float* A, const float* B, float* C, 
                         int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            // Process 4 elements at a time
            for (int k = 0; k <= K - 4; k += 4) {
                float32x4_t a_vec = vld1q_f32(&A[i * K + k]);
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }
            
            // Reduce sum
            float sum = vaddvq_f32(sum_vec);
            
            // Handle remaining elements
            for (int k = (K & ~3); k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            
            C[i * N + j] = sum;
        }
    }
}

// Matrix transpose using NEON
void matrix_transpose_neon(const float* input, float* output, int rows, int cols) {
    // Process 4x4 blocks at a time
    for (int i = 0; i <= rows - 4; i += 4) {
        for (int j = 0; j <= cols - 4; j += 4) {
            float32x4x4_t in = vld1q_f32_x4(&input[i * cols + j]);
            
            // Transpose 4x4 block
            float32x4x4_t out;
            out.val[0] = vtrn1q_f32(in.val[0], in.val[1]);
            out.val[1] = vtrn2q_f32(in.val[0], in.val[1]);
            out.val[2] = vtrn1q_f32(in.val[2], in.val[3]);
            out.val[3] = vtrn2q_f32(in.val[2], in.val[3]);
            
            // Store transposed block
            vst1q_f32(&output[j * rows + i], out.val[0]);
            vst1q_f32(&output[(j+1) * rows + i], out.val[1]);
            vst1q_f32(&output[(j+2) * rows + i], out.val[2]);
            vst1q_f32(&output[(j+3) * rows + i], out.val[3]);
        }
    }
    
    // Handle remaining elements
    for (int i = 0; i < rows; i++) {
        for (int j = (cols & ~3); j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}
