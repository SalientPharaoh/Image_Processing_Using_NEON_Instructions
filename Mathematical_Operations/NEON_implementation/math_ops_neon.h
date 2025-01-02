/**
 * @file math_ops_neon.h
 * @brief NEON-optimized mathematical operations for ARM processors
 * 
 * This header file declares optimized vector and matrix operations using ARM NEON
 * SIMD instructions for improved performance on ARM processors.
 */

#ifndef MATH_OPS_NEON_H
#define MATH_OPS_NEON_H

/**
 * @brief Multiply two vectors element-wise using NEON instructions
 * @param a First input vector
 * @param b Second input vector
 * @param result Output vector to store results
 * @param size Number of elements in vectors
 */
void multiply_vectors_neon(const float* a, const float* b, float* result, int size);

/**
 * @brief Divide two vectors element-wise using NEON instructions
 * @param a Numerator vector
 * @param b Denominator vector
 * @param result Output vector to store results
 * @param size Number of elements in vectors
 */
void divide_vectors_neon(const float* a, const float* b, float* result, int size);

/**
 * @brief Normalize a vector to unit length using NEON instructions
 * @param vector Input/output vector to normalize
 * @param size Number of elements in vector
 */
void normalize_vector_neon(float* vector, int size);

/**
 * @brief Scale a normalized vector by given factor using NEON instructions
 * @param vector Input/output vector to denormalize
 * @param scale Scaling factor
 * @param size Number of elements in vector
 */
void denormalize_vector_neon(float* vector, float scale, int size);

/**
 * @brief Compute exponential (e^x) for vector elements using NEON
 * @param input Input vector
 * @param output Output vector to store results
 * @param size Number of elements in vectors
 */
void exp_vector_neon(const float* input, float* output, int size);

/**
 * @brief Compute natural logarithm for vector elements using NEON
 * @param input Input vector (must be positive)
 * @param output Output vector to store results
 * @param size Number of elements in vectors
 */
void log_vector_neon(const float* input, float* output, int size);

/**
 * @brief Compute power function (x^y) for vectors using NEON
 * @param x Base vector
 * @param y Exponent vector
 * @param output Output vector to store results
 * @param size Number of elements in vectors
 */
void pow_vector_neon(const float* x, const float* y, float* output, int size);

/**
 * @brief Compute sine of vector elements using NEON
 * @param input Input vector (in radians)
 * @param output Output vector to store results
 * @param size Number of elements in vectors
 */
void sin_vector_neon(const float* input, float* output, int size);

/**
 * @brief Compute cosine of vector elements using NEON
 * @param input Input vector (in radians)
 * @param output Output vector to store results
 * @param size Number of elements in vectors
 */
void cos_vector_neon(const float* input, float* output, int size);

/**
 * @brief Multiply two matrices using NEON
 * @param A First input matrix (M x K)
 * @param B Second input matrix (K x N)
 * @param C Output matrix (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void matrix_multiply_neon(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * @brief Transpose a matrix using NEON
 * @param input Input matrix
 * @param output Output matrix
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 */
void matrix_transpose_neon(const float* input, float* output, int rows, int cols);

#endif // MATH_OPS_NEON_H
