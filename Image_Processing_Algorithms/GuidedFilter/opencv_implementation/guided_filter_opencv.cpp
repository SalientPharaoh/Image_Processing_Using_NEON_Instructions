#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

class GuidedFilter {
private:
    int radius;
    float epsilon;
    bool useMultiThreading;

    cv::Mat boxFilter(const cv::Mat& src, int r) {
        cv::Mat dst;
        cv::blur(src, dst, cv::Size(2*r+1, 2*r+1));
        return dst;
    }

public:
    GuidedFilter(int r = 8, float eps = 0.02f, bool multiThread = true) 
        : radius(r), epsilon(eps), useMultiThreading(multiThread) {}

    cv::Mat filter(const cv::Mat& input, const cv::Mat& guidance) {
        CV_Assert(!input.empty() && !guidance.empty());
        
        // Convert input images to floating point
        cv::Mat I, p;
        guidance.convertTo(I, CV_32F, 1.0/255.0);
        input.convertTo(p, CV_32F, 1.0/255.0);

        cv::Mat output;
        
        if (I.channels() == 1) {
            // Grayscale guidance
            output = filterSingleChannel(p, I);
        } else {
            // Color guidance
            output = filterColorGuided(p, I);
        }

        // Convert back to original format
        output.convertTo(output, CV_8U, 255.0);
        return output;
    }

private:
    cv::Mat filterSingleChannel(const cv::Mat& p, const cv::Mat& I) {
        cv::Mat mean_I = boxFilter(I, radius);
        cv::Mat mean_p = boxFilter(p, radius);
        cv::Mat corr_I = boxFilter(I.mul(I), radius);
        cv::Mat corr_Ip = boxFilter(I.mul(p), radius);

        cv::Mat var_I = corr_I - mean_I.mul(mean_I);
        cv::Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);

        cv::Mat a = cov_Ip / (var_I + epsilon);
        cv::Mat b = mean_p - a.mul(mean_I);

        cv::Mat mean_a = boxFilter(a, radius);
        cv::Mat mean_b = boxFilter(b, radius);

        return mean_a.mul(I) + mean_b;
    }

    cv::Mat filterColorGuided(const cv::Mat& p, const cv::Mat& I) {
        std::vector<cv::Mat> rgb;
        cv::split(I, rgb);

        cv::Mat mean_I_r = boxFilter(rgb[0], radius);
        cv::Mat mean_I_g = boxFilter(rgb[1], radius);
        cv::Mat mean_I_b = boxFilter(rgb[2], radius);

        cv::Mat mean_p = boxFilter(p, radius);

        cv::Mat mean_Ip_r = boxFilter(rgb[0].mul(p), radius);
        cv::Mat mean_Ip_g = boxFilter(rgb[1].mul(p), radius);
        cv::Mat mean_Ip_b = boxFilter(rgb[2].mul(p), radius);

        // Variance/Covariance
        cv::Mat var_I_rr = boxFilter(rgb[0].mul(rgb[0]), radius) - mean_I_r.mul(mean_I_r) + epsilon;
        cv::Mat var_I_rg = boxFilter(rgb[0].mul(rgb[1]), radius) - mean_I_r.mul(mean_I_g);
        cv::Mat var_I_rb = boxFilter(rgb[0].mul(rgb[2]), radius) - mean_I_r.mul(mean_I_b);
        cv::Mat var_I_gg = boxFilter(rgb[1].mul(rgb[1]), radius) - mean_I_g.mul(mean_I_g) + epsilon;
        cv::Mat var_I_gb = boxFilter(rgb[1].mul(rgb[2]), radius) - mean_I_g.mul(mean_I_b);
        cv::Mat var_I_bb = boxFilter(rgb[2].mul(rgb[2]), radius) - mean_I_b.mul(mean_I_b) + epsilon;

        cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
        cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
        cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

        cv::Mat output = cv::Mat::zeros(p.size(), CV_32F);
        
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int i = 0; i < p.rows; i++) {
            for (int j = 0; j < p.cols; j++) {
                float* var = new float[6];
                var[0] = var_I_rr.at<float>(i,j);
                var[1] = var_I_rg.at<float>(i,j);
                var[2] = var_I_rb.at<float>(i,j);
                var[3] = var_I_gg.at<float>(i,j);
                var[4] = var_I_gb.at<float>(i,j);
                var[5] = var_I_bb.at<float>(i,j);

                cv::Mat Sigma = (cv::Mat_<float>(3,3) <<
                    var[0], var[1], var[2],
                    var[1], var[3], var[4],
                    var[2], var[4], var[5]);

                cv::Mat cov = (cv::Mat_<float>(3,1) <<
                    cov_Ip_r.at<float>(i,j),
                    cov_Ip_g.at<float>(i,j),
                    cov_Ip_b.at<float>(i,j));

                cv::Mat a = Sigma.inv() * cov;
                float b = mean_p.at<float>(i,j) - 
                         (a.at<float>(0) * mean_I_r.at<float>(i,j) +
                          a.at<float>(1) * mean_I_g.at<float>(i,j) +
                          a.at<float>(2) * mean_I_b.at<float>(i,j));

                output.at<float>(i,j) = 
                    a.at<float>(0) * rgb[0].at<float>(i,j) +
                    a.at<float>(1) * rgb[1].at<float>(i,j) +
                    a.at<float>(2) * rgb[2].at<float>(i,j) + b;

                delete[] var;
            }
        }

        return output;
    }
};

// Example usage
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat input = cv::imread(argv[1]);
    if (input.empty()) {
        std::cout << "Error: Could not read image." << std::endl;
        return -1;
    }

    // Create Guided Filter object
    GuidedFilter gf(8, 0.02f, true);

    // Process image using the input as both guidance and filtering target
    cv::Mat output = gf.filter(input, input);

    // Display results
    cv::imshow("Original", input);
    cv::imshow("Filtered", output);
    cv::waitKey(0);

    return 0;
}
