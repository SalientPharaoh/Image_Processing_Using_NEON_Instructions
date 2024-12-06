#ifndef DEMOSAIC_NEON_H
#define DEMOSAIC_NEON_H

// Main demosaicing function
void demosaic_neon(const unsigned char* bayer, unsigned char* rgb,
                  int width, int height, int pattern);

// Helper functions
void interpolate_green_neon(const unsigned char* bayer, unsigned char* green,
                          int width, int height, int pattern);
void interpolate_red_blue_neon(const unsigned char* bayer, unsigned char* red,
                             unsigned char* blue, const unsigned char* green,
                             int width, int height, int pattern);

// Bayer pattern definitions
#define BAYER_RGGB 0
#define BAYER_BGGR 1
#define BAYER_GRBG 2
#define BAYER_GBRG 3

#endif // DEMOSAIC_NEON_H
