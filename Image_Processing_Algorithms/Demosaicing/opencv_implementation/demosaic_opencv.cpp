#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

class Demosaicing {
private:
    int bayerPattern;
    bool useMultiThreading;
    bool useAdvancedMethod;

public:
    enum BayerPattern {
        RGGB = 0,
        BGGR = 1,
        GRBG = 2,
        GBRG = 3
    };

    Demosaicing(int pattern = RGGB, bool advanced = true, bool multiThread = true)
        : bayerPattern(pattern),
          useAdvancedMethod(advanced),
          useMultiThreading(multiThread) {}

    cv::Mat demosaic(const cv::Mat& bayerImage) {
        CV_Assert(!bayerImage.empty() && bayerImage.channels() == 1);
        
        cv::Mat output;
        
        if (useAdvancedMethod) {
            output = demosaicAdvanced(bayerImage);
        } else {
            // Use OpenCV's built-in demosaicing
            cv::cvtColor(bayerImage, output, cv::COLOR_BayerBG2BGR + bayerPattern);
        }
        
        return output;
    }

private:
    cv::Mat demosaicAdvanced(const cv::Mat& bayer) {
        cv::Mat float_bayer;
        bayer.convertTo(float_bayer, CV_32F);
        
        // Create output image
        cv::Mat output(bayer.size(), CV_32FC3);
        std::vector<cv::Mat> channels(3);
        for (int i = 0; i < 3; i++) {
            channels[i] = cv::Mat::zeros(bayer.size(), CV_32F);
        }

        // Determine offsets based on Bayer pattern
        int r_offset_y = (bayerPattern == RGGB || bayerPattern == GRBG) ? 0 : 1;
        int r_offset_x = (bayerPattern == RGGB || bayerPattern == GBRG) ? 0 : 1;
        int b_offset_y = 1 - r_offset_y;
        int b_offset_x = 1 - r_offset_x;

        // Copy known values
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int y = 0; y < bayer.rows; y += 2) {
            for (int x = 0; x < bayer.cols; x += 2) {
                // Red pixel
                if (y + r_offset_y < bayer.rows && x + r_offset_x < bayer.cols)
                    channels[0].at<float>(y + r_offset_y, x + r_offset_x) = 
                        float_bayer.at<float>(y + r_offset_y, x + r_offset_x);
                
                // Blue pixel
                if (y + b_offset_y < bayer.rows && x + b_offset_x < bayer.cols)
                    channels[2].at<float>(y + b_offset_y, x + b_offset_x) = 
                        float_bayer.at<float>(y + b_offset_y, x + b_offset_x);
                
                // Green pixels
                if (y < bayer.rows && x + 1 < bayer.cols)
                    channels[1].at<float>(y, x + 1) = float_bayer.at<float>(y, x + 1);
                if (y + 1 < bayer.rows && x < bayer.cols)
                    channels[1].at<float>(y + 1, x) = float_bayer.at<float>(y + 1, x);
            }
        }

        // Interpolate green channel
        interpolateGreen(channels[1], float_bayer);

        // Interpolate red and blue channels
        interpolateRedBlue(channels);

        // Merge channels
        cv::merge(channels, output);

        // Post-processing
        output = postProcess(output);

        // Convert back to 8-bit
        cv::Mat result;
        output.convertTo(result, CV_8UC3);
        return result;
    }

    void interpolateGreen(cv::Mat& green, const cv::Mat& bayer) {
        cv::Mat gradients_h = cv::Mat::zeros(bayer.size(), CV_32F);
        cv::Mat gradients_v = cv::Mat::zeros(bayer.size(), CV_32F);

        // Compute gradients
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int y = 2; y < bayer.rows - 2; y++) {
            for (int x = 2; x < bayer.cols - 2; x++) {
                if ((x + y) % 2 == 1) continue; // Skip existing green pixels

                // Horizontal gradient
                float grad_h = std::abs(bayer.at<float>(y, x-2) - bayer.at<float>(y, x)) +
                             std::abs(bayer.at<float>(y, x+2) - bayer.at<float>(y, x)) +
                             std::abs(bayer.at<float>(y, x-1) - bayer.at<float>(y, x+1));

                // Vertical gradient
                float grad_v = std::abs(bayer.at<float>(y-2, x) - bayer.at<float>(y, x)) +
                             std::abs(bayer.at<float>(y+2, x) - bayer.at<float>(y, x)) +
                             std::abs(bayer.at<float>(y-1, x) - bayer.at<float>(y+1, x));

                gradients_h.at<float>(y, x) = grad_h;
                gradients_v.at<float>(y, x) = grad_v;
            }
        }

        // Interpolate green based on gradients
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int y = 2; y < bayer.rows - 2; y++) {
            for (int x = 2; x < bayer.cols - 2; x++) {
                if ((x + y) % 2 == 1) continue; // Skip existing green pixels

                float gh = gradients_h.at<float>(y, x);
                float gv = gradients_v.at<float>(y, x);

                float h_val = (bayer.at<float>(y, x-1) + bayer.at<float>(y, x+1)) / 2.0f;
                float v_val = (bayer.at<float>(y-1, x) + bayer.at<float>(y+1, x)) / 2.0f;

                if (gh < gv) {
                    green.at<float>(y, x) = h_val;
                } else if (gv < gh) {
                    green.at<float>(y, x) = v_val;
                } else {
                    green.at<float>(y, x) = (h_val + v_val) / 2.0f;
                }
            }
        }
    }

    void interpolateRedBlue(std::vector<cv::Mat>& channels) {
        cv::Mat& red = channels[0];
        cv::Mat& green = channels[1];
        cv::Mat& blue = channels[2];

        cv::Mat red_temp = red.clone();
        cv::Mat blue_temp = blue.clone();

        // Interpolate red and blue at green pixels
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int y = 1; y < red.rows - 1; y++) {
            for (int x = 1; x < red.cols - 1; x++) {
                if ((x + y) % 2 == 1) { // Green pixel locations
                    // Red interpolation
                    if (y % 2 == 0) { // Horizontal neighbors
                        red_temp.at<float>(y, x) = (red.at<float>(y, x-1) + 
                                                   red.at<float>(y, x+1)) / 2.0f;
                    } else { // Vertical neighbors
                        red_temp.at<float>(y, x) = (red.at<float>(y-1, x) + 
                                                   red.at<float>(y+1, x)) / 2.0f;
                    }

                    // Blue interpolation
                    if (y % 2 == 1) { // Horizontal neighbors
                        blue_temp.at<float>(y, x) = (blue.at<float>(y, x-1) + 
                                                    blue.at<float>(y, x+1)) / 2.0f;
                    } else { // Vertical neighbors
                        blue_temp.at<float>(y, x) = (blue.at<float>(y-1, x) + 
                                                    blue.at<float>(y+1, x)) / 2.0f;
                    }
                }
            }
        }

        // Interpolate red and blue at blue and red pixels respectively
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int y = 1; y < red.rows - 1; y++) {
            for (int x = 1; x < red.cols - 1; x++) {
                if ((x + y) % 2 == 0) { // Red or Blue pixel locations
                    if (red.at<float>(y, x) > 0) { // Blue at red
                        blue_temp.at<float>(y, x) = (blue.at<float>(y-1, x-1) + 
                                                    blue.at<float>(y-1, x+1) +
                                                    blue.at<float>(y+1, x-1) + 
                                                    blue.at<float>(y+1, x+1)) / 4.0f;
                    } else { // Red at blue
                        red_temp.at<float>(y, x) = (red.at<float>(y-1, x-1) + 
                                                   red.at<float>(y-1, x+1) +
                                                   red.at<float>(y+1, x-1) + 
                                                   red.at<float>(y+1, x+1)) / 4.0f;
                    }
                }
            }
        }

        red = red_temp;
        blue = blue_temp;
    }

    cv::Mat postProcess(const cv::Mat& demosaiced) {
        cv::Mat result;
        
        // Apply median blur to reduce color artifacts
        cv::medianBlur(demosaiced, result, 3);
        
        // Apply bilateral filter to preserve edges while reducing noise
        cv::Mat temp;
        cv::bilateralFilter(result, temp, 5, 50, 50);
        
        // Enhance local contrast
        std::vector<cv::Mat> channels;
        cv::split(temp, channels);
        
        for (int i = 0; i < 3; i++) {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
            clahe->apply(channels[i], channels[i]);
        }
        
        cv::merge(channels, result);
        return result;
    }
};

// Example usage
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <bayer_image_path>" << std::endl;
        return -1;
    }

    // Load Bayer image (should be single channel)
    cv::Mat bayer = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (bayer.empty()) {
        std::cout << "Error: Could not read image." << std::endl;
        return -1;
    }

    // Create Demosaicing object
    Demosaicing demosaic(Demosaicing::RGGB, true, true);

    // Process image
    cv::Mat result = demosaic.demosaic(bayer);

    // Display results
    cv::imshow("Bayer", bayer);
    cv::imshow("Demosaiced", result);
    cv::waitKey(0);

    return 0;
}
