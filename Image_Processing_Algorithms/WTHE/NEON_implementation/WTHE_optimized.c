#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>

#define MAX_GRAY_LEVEL 256
#define NUM_THREADS 4
#define HISTOGRAM_BINS 256
#define VECTOR_SIZE 16

// Align memory to 16-byte boundary for better NEON performance
#define ALIGN16(x) ((((uintptr_t)(x) + 15) & ~15))

// Structure to hold image data
typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

// Thread argument structure
typedef struct {
    const uint8_t* image;
    int start;
    int end;
    uint32_t* local_histogram;
} ThreadArg;

// NEON-optimized histogram computation using local histograms per thread
void* compute_histogram_thread(void* arg) {
    ThreadArg* thread_arg = (ThreadArg*)arg;
    const uint8_t* image = thread_arg->image;
    const int start = thread_arg->start;
    const int end = thread_arg->end;
    uint32_t* local_hist = thread_arg->local_histogram;
    
    // Initialize local histogram
    memset(local_hist, 0, HISTOGRAM_BINS * sizeof(uint32_t));
    
    // Process 32 pixels at a time using NEON
    int i;
    for (i = start; i <= end - 32; i += 32) {
        uint8x16x2_t pixels = vld1q_u8_x2(&image[i]);
        
        // First 16 pixels
        uint8x16_t p1 = pixels.val[0];
        uint8x8_t p1_low = vget_low_u8(p1);
        uint8x8_t p1_high = vget_high_u8(p1);
        
        // Second 16 pixels
        uint8x16_t p2 = pixels.val[1];
        uint8x8_t p2_low = vget_low_u8(p2);
        uint8x8_t p2_high = vget_high_u8(p2);
        
        // Increment histogram bins using NEON
        #pragma unroll(8)
        for (int j = 0; j < 8; j++) {
            local_hist[vget_lane_u8(p1_low, j)]++;
            local_hist[vget_lane_u8(p1_high, j)]++;
            local_hist[vget_lane_u8(p2_low, j)]++;
            local_hist[vget_lane_u8(p2_high, j)]++;
        }
    }
    
    // Handle remaining pixels
    for (; i < end; i++) {
        local_hist[image[i]]++;
    }
    
    return NULL;
}

// Optimized histogram computation using multiple threads
void compute_histogram_neon_mt(const uint8_t* image, int size, uint32_t* histogram) {
    pthread_t threads[NUM_THREADS];
    ThreadArg thread_args[NUM_THREADS];
    uint32_t* local_histograms = (uint32_t*)aligned_alloc(16, NUM_THREADS * HISTOGRAM_BINS * sizeof(uint32_t));
    
    // Create threads
    int chunk_size = size / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].image = image;
        thread_args[i].start = i * chunk_size;
        thread_args[i].end = (i == NUM_THREADS - 1) ? size : (i + 1) * chunk_size;
        thread_args[i].local_histogram = &local_histograms[i * HISTOGRAM_BINS];
        pthread_create(&threads[i], NULL, compute_histogram_thread, &thread_args[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Merge local histograms using NEON
    memset(histogram, 0, HISTOGRAM_BINS * sizeof(uint32_t));
    for (int bin = 0; bin < HISTOGRAM_BINS; bin += 4) {
        uint32x4_t sum = vdupq_n_u32(0);
        for (int t = 0; t < NUM_THREADS; t++) {
            uint32x4_t local = vld1q_u32(&local_histograms[t * HISTOGRAM_BINS + bin]);
            sum = vaddq_u32(sum, local);
        }
        vst1q_u32(&histogram[bin], sum);
    }
    
    free(local_histograms);
}

// Optimized PDF computation using improved NEON vectorization
void compute_weighted_pdf_neon(const uint32_t* histogram, float* pdf, float total_pixels,
                             float r, float pl, float pu) {
    float32x4_t v_pl = vdupq_n_f32(pl);
    float32x4_t v_pu = vdupq_n_f32(pu);
    float32x4_t v_total = vdupq_n_f32(total_pixels);
    float32x4_t v_r = vdupq_n_f32(r);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    
    // Process 8 bins at a time using NEON
    #pragma unroll(2)
    for (int i = 0; i < MAX_GRAY_LEVEL; i += 8) {
        // Load 8 histogram values
        uint32x4_t h_val1 = vld1q_u32(&histogram[i]);
        uint32x4_t h_val2 = vld1q_u32(&histogram[i + 4]);
        
        // Convert to float and compute PDF
        float32x4_t pdf_val1 = vcvtq_f32_u32(h_val1);
        float32x4_t pdf_val2 = vcvtq_f32_u32(h_val2);
        
        pdf_val1 = vdivq_f32(pdf_val1, v_total);
        pdf_val2 = vdivq_f32(pdf_val2, v_total);
        
        // Apply thresholds
        uint32x4_t mask_low1 = vcltq_f32(pdf_val1, v_pl);
        uint32x4_t mask_high1 = vcgtq_f32(pdf_val1, v_pu);
        uint32x4_t mask_low2 = vcltq_f32(pdf_val2, v_pl);
        uint32x4_t mask_high2 = vcgtq_f32(pdf_val2, v_pu);
        
        pdf_val1 = vbslq_f32(mask_low1, v_zero, pdf_val1);
        pdf_val1 = vbslq_f32(mask_high1, v_pu, pdf_val1);
        pdf_val2 = vbslq_f32(mask_low2, v_zero, pdf_val2);
        pdf_val2 = vbslq_f32(mask_high2, v_pu, pdf_val2);
        
        // Compute weighted PDF using fast approximation
        float32x4_t numerator1 = vsubq_f32(pdf_val1, v_pl);
        float32x4_t numerator2 = vsubq_f32(pdf_val2, v_pl);
        float32x4_t denominator = vsubq_f32(v_pu, v_pl);
        
        float32x4_t ratio1 = vdivq_f32(numerator1, denominator);
        float32x4_t ratio2 = vdivq_f32(numerator2, denominator);
        
        // Fast power approximation
        float32x4_t log_val1 = vlogq_f32(vaddq_f32(ratio1, v_one));
        float32x4_t log_val2 = vlogq_f32(vaddq_f32(ratio2, v_one));
        float32x4_t weighted1 = vexpq_f32(vmulq_f32(log_val1, v_r));
        float32x4_t weighted2 = vexpq_f32(vmulq_f32(log_val2, v_r));
        
        weighted1 = vmulq_f32(weighted1, v_pu);
        weighted2 = vmulq_f32(weighted2, v_pu);
        
        // Store results
        vst1q_f32(&pdf[i], weighted1);
        vst1q_f32(&pdf[i + 4], weighted2);
    }
}

// Optimized CDF computation with improved NEON utilization
void compute_cdf_neon(const float* pdf, float* cdf) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    
    // First value
    cdf[0] = pdf[0];
    sum1 = vsetq_lane_f32(pdf[0], sum1, 0);
    
    // Process 8 values at a time using NEON
    #pragma unroll(2)
    for (int i = 1; i < MAX_GRAY_LEVEL; i += 8) {
        float32x4_t pdf_val1 = vld1q_f32(&pdf[i]);
        float32x4_t pdf_val2 = vld1q_f32(&pdf[i + 4]);
        
        sum1 = vaddq_f32(sum1, pdf_val1);
        sum2 = vaddq_f32(sum2, pdf_val2);
        
        vst1q_f32(&cdf[i], sum1);
        vst1q_f32(&cdf[i + 4], sum2);
    }
    
    // Normalize CDF using NEON
    float32x4_t v_max = vdupq_n_f32(cdf[MAX_GRAY_LEVEL-1]);
    #pragma unroll(4)
    for (int i = 0; i < MAX_GRAY_LEVEL; i += 4) {
        float32x4_t cdf_val = vld1q_f32(&cdf[i]);
        cdf_val = vdivq_f32(cdf_val, v_max);
        vst1q_f32(&cdf[i], cdf_val);
    }
}

// Enhanced image enhancement using optimized NEON operations
void enhance_image_neon(const uint8_t* input, uint8_t* output, int size,
                       const float* cdf, float w_out, float m_adj) {
    float32x4_t v_wout = vdupq_n_f32(w_out);
    float32x4_t v_madj = vdupq_n_f32(m_adj);
    float32x4_t v_255 = vdupq_n_f32(255.0f);
    float32x4_t v_0 = vdupq_n_f32(0.0f);
    
    // Process 32 pixels at a time
    #pragma unroll(2)
    for (int i = 0; i <= size - 32; i += 32) {
        uint8x16x2_t pixels = vld1q_u8_x2(&input[i]);
        
        // Process first 16 pixels
        uint8x16_t p1 = pixels.val[0];
        uint8x8_t p1_low = vget_low_u8(p1);
        uint8x8_t p1_high = vget_high_u8(p1);
        
        // Process second 16 pixels
        uint8x16_t p2 = pixels.val[1];
        uint8x8_t p2_low = vget_low_u8(p2);
        uint8x8_t p2_high = vget_high_u8(p2);
        
        // Convert to float32 for processing
        float32x4_t f1_low = vcvtq_f32_u32(vmovl_u16(vmovl_u8(p1_low)));
        float32x4_t f1_high = vcvtq_f32_u32(vmovl_u16(vmovl_u8(p1_high)));
        float32x4_t f2_low = vcvtq_f32_u32(vmovl_u16(vmovl_u8(p2_low)));
        float32x4_t f2_high = vcvtq_f32_u32(vmovl_u16(vmovl_u8(p2_high)));
        
        // Apply CDF mapping and enhancement
        #pragma unroll(4)
        for (int j = 0; j < 4; j++) {
            float val;
            
            val = vgetq_lane_f32(f1_low, j);
            f1_low = vsetq_lane_f32(cdf[(int)val] * 255.0f, f1_low, j);
            
            val = vgetq_lane_f32(f1_high, j);
            f1_high = vsetq_lane_f32(cdf[(int)val] * 255.0f, f1_high, j);
            
            val = vgetq_lane_f32(f2_low, j);
            f2_low = vsetq_lane_f32(cdf[(int)val] * 255.0f, f2_low, j);
            
            val = vgetq_lane_f32(f2_high, j);
            f2_high = vsetq_lane_f32(cdf[(int)val] * 255.0f, f2_high, j);
        }
        
        // Apply brightness adjustment
        f1_low = vmulq_f32(f1_low, v_wout);
        f1_low = vaddq_f32(f1_low, v_madj);
        f1_high = vmulq_f32(f1_high, v_wout);
        f1_high = vaddq_f32(f1_high, v_madj);
        f2_low = vmulq_f32(f2_low, v_wout);
        f2_low = vaddq_f32(f2_low, v_madj);
        f2_high = vmulq_f32(f2_high, v_wout);
        f2_high = vaddq_f32(f2_high, v_madj);
        
        // Clamp values
        f1_low = vminq_f32(vmaxq_f32(f1_low, v_0), v_255);
        f1_high = vminq_f32(vmaxq_f32(f1_high, v_0), v_255);
        f2_low = vminq_f32(vmaxq_f32(f2_low, v_0), v_255);
        f2_high = vminq_f32(vmaxq_f32(f2_high, v_0), v_255);
        
        // Convert back to uint8
        uint8x16_t result1 = vcombine_u8(
            vmovn_u16(vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(f1_low)),
                vmovn_u32(vcvtq_u32_f32(f1_high))
            )),
            vmovn_u16(vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(f2_low)),
                vmovn_u32(vcvtq_u32_f32(f2_high))
            ))
        );
        
        // Store results
        vst1q_u8(&output[i], result1);
    }
}

// Optimized RGB to Grayscale conversion with improved NEON utilization
void rgb_to_gray_neon(const uint8_t* rgb, uint8_t* gray, int width, int height) {
    const float32x4_t v_r_weight = vdupq_n_f32(0.299f);
    const float32x4_t v_g_weight = vdupq_n_f32(0.587f);
    const float32x4_t v_b_weight = vdupq_n_f32(0.114f);
    
    int size = width * height;
    #pragma unroll(2)
    for (int i = 0; i <= size - 8; i += 8) {
        // Load 8 RGB pixels (24 values)
        uint8x8x3_t rgb_pixels = vld3_u8(&rgb[i * 3]);
        
        // Convert first 4 pixels
        uint32x4_t r1 = vmovl_u16(vmovl_u8(rgb_pixels.val[0]));
        uint32x4_t g1 = vmovl_u16(vmovl_u8(rgb_pixels.val[1]));
        uint32x4_t b1 = vmovl_u16(vmovl_u8(rgb_pixels.val[2]));
        
        float32x4_t fr1 = vcvtq_f32_u32(r1);
        float32x4_t fg1 = vcvtq_f32_u32(g1);
        float32x4_t fb1 = vcvtq_f32_u32(b1);
        
        float32x4_t gray_f1 = vmulq_f32(fr1, v_r_weight);
        gray_f1 = vmlaq_f32(gray_f1, fg1, v_g_weight);
        gray_f1 = vmlaq_f32(gray_f1, fb1, v_b_weight);
        
        // Convert back to uint8
        uint32x4_t gray_u32 = vcvtq_u32_f32(gray_f1);
        uint16x4_t gray_u16 = vmovn_u32(gray_u32);
        uint8x8_t gray_u8 = vmovn_u16(vcombine_u16(gray_u16, gray_u16));
        
        // Store result
        vst1_u8(&gray[i], gray_u8);
    }
}

// Main WTHE function that combines all optimized components
void wthe_neon(const Image* input, Image* output, float r, float v) {
    int size = input->width * input->height;
    uint8_t* gray_image = NULL;
    
    // Convert to grayscale if needed
    if (input->channels == 3) {
        gray_image = (uint8_t*)malloc(size);
        rgb_to_gray_neon(input->data, gray_image, input->width, input->height);
    } else {
        gray_image = input->data;
    }
    
    // Compute histogram
    uint32_t histogram[MAX_GRAY_LEVEL];
    compute_histogram_neon_mt(gray_image, size, histogram);
    
    // Compute weighted PDF
    float pdf[MAX_GRAY_LEVEL];
    float pl = 1e-4f;  // Lower threshold
    float pu = v * histogram[0] / (float)size;  // Upper threshold
    compute_weighted_pdf_neon(histogram, pdf, (float)size, r, pl, pu);
    
    // Compute CDF
    float cdf[MAX_GRAY_LEVEL];
    compute_cdf_neon(pdf, cdf);
    
    // Calculate Win and Wout
    int win = 0;
    for (int i = 0; i < MAX_GRAY_LEVEL; i++) {
        if (histogram[i] > 0) win++;
    }
    float gmax = 1.5f;
    float w_out = fminf(255.0f, gmax * win);
    
    // Calculate mean adjustment
    float input_mean = 0, output_mean = 0;
    for (int i = 0; i < MAX_GRAY_LEVEL; i++) {
        input_mean += i * histogram[i];
        output_mean += i * cdf[i] * histogram[i];
    }
    input_mean /= size;
    output_mean /= size;
    float m_adj = input_mean - output_mean;
    
    // Enhance image
    enhance_image_neon(gray_image, output->data, size, cdf, w_out, m_adj);
    
    // Clean up
    if (input->channels == 3) {
        free(gray_image);
    }
}

// Image I/O functions (same as in the LIME implementation)
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

void save_image(const char* filename, Image* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
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
    
    // Apply WTHE
    wthe_neon(input, output, 0.5f, 0.5f);  // r=0.5, v=0.5 are default parameters
    
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
