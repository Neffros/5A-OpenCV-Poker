#include "../include/Image.h"

Image::Image(cv::Mat&& originalPixelData, cv::Mat&& preprocessedPixelData, cv::FeatureDetector& featureDetector):
_originalPixelData(originalPixelData), _preprocessedPixelData(preprocessedPixelData)
{
	featureDetector.detectAndCompute(_preprocessedPixelData, cv::noArray(), _keyPoints, _descriptors);
}

const cv::Mat& Image::getOriginalPixelData() const
{
	return _originalPixelData;
}

const std::vector<cv::KeyPoint>& Image::getKeyPoints() const
{
	return _keyPoints;
}

const cv::Mat& Image::getDescriptors() const
{
	return _descriptors;
}

const cv::Mat &Image::getPreprocessedPixelData() const {
    return _preprocessedPixelData;
}
