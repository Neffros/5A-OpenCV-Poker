#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

class Image {
private:

public:
    cv::Mat rawImg;
    std::vector<cv::KeyPoint> objectKeyPoints;
    cv::Mat descriptor;
    explicit Image(const cv::Mat& img);
    Image() = default;

    Image& operator=(const cv::Mat& image);
};