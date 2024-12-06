#ifndef BM3D_NEON_H
#define BM3D_NEON_H

// Main BM3D denoising function
void bm3d_denoise_neon(const float* input, float* output, int width, int height, float sigma);

// Helper functions
float compute_block_distance_neon(const float* block1, const float* block2, int stride);
void dct_2d_8x8_neon(float* block, int stride);
void idct_2d_8x8_neon(float* block, int stride);
void apply_wiener_filter_neon(float* block, float sigma, int stride);

#endif // BM3D_NEON_H
