#include "../include/Image.h"

Image::Image(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector):
_pixelData(pixelData)
{
	featureDetector.detectAndCompute(_pixelData, cv::noArray(), _keyPoints, _descriptors);
}

const cv::Mat& Image::getPixelData() const
{
	return _pixelData;
}

const std::vector<cv::KeyPoint>& Image::getKeyPoints() const
{
	return _keyPoints;
}

const cv::Mat& Image::getDescriptors() const
{
	return _descriptors;
}
