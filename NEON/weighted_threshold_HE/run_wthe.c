#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <math.h>
#include "wthe.h"
#include "opencv2/opencv.hpp"

int main() {
    const char *filename = "../../images/Plane.jpg";
    cv::Mat image = cv::imread(filename);
    cv::Mat image_hsv;
    cv::cvtColor(image, image_hsv, cv::COLOR_BGR2HSV);
    cv::Mat image_v = image_hsv(cv::Rect(0, 0, image.cols, image.rows)).col(2).clone();
    uint8_t *image_v_data = image_v.data;
    uint8_t *image_v_heq_data = (uint8_t *)malloc(image.rows * image.cols * sizeof(uint8_t));
    float Wout;
    imWTHeq(image_v_data, image.rows, image.cols, NULL, image_v_heq_data, &Wout, 0.5, 0.5);
    image_hsv(cv::Rect(0, 0, image.cols, image.rows)).col(2) = cv::Mat(image.rows, image.cols, CV_8UC1, image_v_heq_data);
    cv::Mat image_heq;
    cv::cvtColor(image_hsv, image_heq, cv::COLOR_HSV2BGR);
    cv::imwrite("Plane-imWeightedThresholdedheq.jpg", image_heq);
    free(image_v_heq_data);
    return 0;
}