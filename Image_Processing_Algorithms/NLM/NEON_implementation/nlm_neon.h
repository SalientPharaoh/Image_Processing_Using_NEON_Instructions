#ifndef NLM_NEON_H
#define NLM_NEON_H

// Main NLM denoising function
void nlm_denoise_neon(const float* input, float* output, int width, int height, 
                     float h, int patch_size, int search_window);

// Helper functions
float compute_patch_distance_neon(const float* p1, const float* p2, int patch_size, int stride);
void compute_weights_neon(const float* distances, float* weights, int size, float h);

#endif // NLM_NEON_H
