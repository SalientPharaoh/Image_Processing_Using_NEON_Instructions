#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

class NLMDenoising {
private:
    int templateWindowSize;
    int searchWindowSize;
    float h;
    bool useMultiThreading;

public:
    NLMDenoising(int templateWindow = 7, int searchWindow = 21, float hParam = 10.0f, 
                 bool multiThread = true) 
        : templateWindowSize(templateWindow), 
          searchWindowSize(searchWindow),
          h(hParam),
          useMultiThreading(multiThread) {}

    cv::Mat denoise(const cv::Mat& input) {
        CV_Assert(!input.empty() && input.channels() <= 4);
        
        cv::Mat output;
        if (input.channels() == 1) {
            // Grayscale image processing
            cv::fastNlMeansDenoising(input, output, h, templateWindowSize, searchWindowSize);
        } else {
            // Color image processing
            cv::fastNlMeansDenoisingColored(input, output, h, h, templateWindowSize, searchWindowSize);
        }
        
        return output;
    }

    cv::Mat denoiseCustom(const cv::Mat& input) {
        CV_Assert(!input.empty() && input.channels() <= 4);
        
        cv::Mat output = input.clone();
        cv::Mat inputFloat;
        input.convertTo(inputFloat, CV_32F, 1.0/255.0);

        // Parameters
        const int halfTemplate = templateWindowSize / 2;
        const int halfSearch = searchWindowSize / 2;
        const float hSquared = h * h;

        // Parallel processing if enabled
        #pragma omp parallel for if(useMultiThreading) collapse(2)
        for (int i = halfTemplate; i < input.rows - halfTemplate; i++) {
            for (int j = halfTemplate; j < input.cols - halfTemplate; j++) {
                cv::Mat patch = inputFloat(
                    cv::Range(i - halfTemplate, i + halfTemplate + 1),
                    cv::Range(j - halfTemplate, j + halfTemplate + 1)
                );

                float weightSum = 0;
                cv::Vec3f pixelSum = cv::Vec3f(0, 0, 0);

                // Search window
                for (int si = -halfSearch; si <= halfSearch; si++) {
                    for (int sj = -halfSearch; sj <= halfSearch; sj++) {
                        int ni = i + si;
                        int nj = j + sj;

                        if (ni < halfTemplate || ni >= input.rows - halfTemplate ||
                            nj < halfTemplate || nj >= input.cols - halfTemplate) {
                            continue;
                        }

                        cv::Mat neighborPatch = inputFloat(
                            cv::Range(ni - halfTemplate, ni + halfTemplate + 1),
                            cv::Range(nj - halfTemplate, nj + halfTemplate + 1)
                        );

                        // Compute patch distance
                        cv::Mat diff = patch - neighborPatch;
                        float distance = cv::norm(diff, cv::NORM_L2SQR) / (templateWindowSize * templateWindowSize);
                        float weight = std::exp(-distance / hSquared);

                        weightSum += weight;
                        pixelSum += weight * inputFloat.at<cv::Vec3f>(ni, nj);
                    }
                }

                output.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(pixelSum[0] * 255.0f / weightSum),
                    cv::saturate_cast<uchar>(pixelSum[1] * 255.0f / weightSum),
                    cv::saturate_cast<uchar>(pixelSum[2] * 255.0f / weightSum)
                );
            }
        }

        return output;
    }

    // Utility function to compare OpenCV's implementation with custom implementation
    void compareImplementations(const cv::Mat& input) {
        cv::Mat opencvResult = denoise(input);
        cv::Mat customResult = denoiseCustom(input);

        cv::Mat diff;
        cv::absdiff(opencvResult, customResult, diff);

        cv::Scalar meanDiff = cv::mean(diff);
        std::cout << "Average difference between implementations: "
                  << (meanDiff[0] + meanDiff[1] + meanDiff[2])/3.0 << std::endl;

        // Display results
        cv::imshow("Original", input);
        cv::imshow("OpenCV NLM", opencvResult);
        cv::imshow("Custom NLM", customResult);
        cv::imshow("Difference", diff);
        cv::waitKey(0);
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

    // Create NLM denoising object
    NLMDenoising nlm(7, 21, 10.0f, true);

    // Process and compare implementations
    nlm.compareImplementations(input);

    return 0;
}
