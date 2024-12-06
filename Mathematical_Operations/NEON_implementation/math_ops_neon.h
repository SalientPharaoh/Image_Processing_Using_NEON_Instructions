#ifndef MATH_OPS_NEON_H
#define MATH_OPS_NEON_H

// Basic vector operations
void multiply_vectors_neon(const float* a, const float* b, float* result, int size);
void divide_vectors_neon(const float* a, const float* b, float* result, int size);
void normalize_vector_neon(float* vector, int size);
void denormalize_vector_neon(float* vector, float scale, int size);

// Advanced mathematical operations
void exp_vector_neon(const float* input, float* output, int size);
void log_vector_neon(const float* input, float* output, int size);
void pow_vector_neon(const float* x, const float* y, float* output, int size);

// Trigonometric functions
void sin_vector_neon(const float* input, float* output, int size);
void cos_vector_neon(const float* input, float* output, int size);

// Matrix operations
void matrix_multiply_neon(const float* A, const float* B, float* C, int M, int N, int K);
void matrix_transpose_neon(const float* input, float* output, int rows, int cols);

#endif // MATH_OPS_NEON_H
