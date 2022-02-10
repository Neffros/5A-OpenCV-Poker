#include "../include/Image.h"

Image::Image(cv::Mat&& pixelData, cv::FeatureDetector* featureDetector):
_pixelData(pixelData)
{
	if (featureDetector)
	{
		featureDetector->detectAndCompute(_pixelData, cv::noArray(), _keyPoints, _descriptors);
	}
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

const std::vector<cv::Point2f> Image::getImageEdges() const
{
    std::vector<cv::Point2f> res(4);

    res[0] = cv::Point2f(0, 0);
    res[1] = cv::Point2f(_pixelData.cols, 0);
    res[2] = cv::Point2f(_pixelData.cols, _pixelData.rows);
    res[3] = cv::Point2f(0, _pixelData.rows);
    return res;
}
