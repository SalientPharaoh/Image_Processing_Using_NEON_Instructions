#ifndef GUIDED_FILTER_NEON_H
#define GUIDED_FILTER_NEON_H

// Main guided filter function
void guided_filter_neon(const float* input, const float* guidance,
                       float* output, int width, int height,
                       int radius, float epsilon, int subsample);

// Helper functions
void box_filter_neon(const float* input, float* output,
                    int width, int height, int radius);
void compute_covariance_neon(const float* I, const float* P,
                            float* cov, int width, int height);

#endif // GUIDED_FILTER_NEON_H
