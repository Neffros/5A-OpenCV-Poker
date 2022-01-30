#include "../headers/Image.h"

Image::Image(const cv::Mat& img) : rawImg(img){}

Image &Image::operator=(const cv::Mat& image) {
    this->rawImg = image;
    return *this;
}



