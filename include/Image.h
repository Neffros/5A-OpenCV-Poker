#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

class Image
{
public:
    Image(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector);
	virtual ~Image() = default;
	
	const cv::Mat& getPixelData() const;
	const std::vector<cv::KeyPoint>& getKeyPoints() const;
	const cv::Mat& getDescriptors() const;
    const std::vector<cv::Point2f> getImageEdges();

private:
	cv::Mat _pixelData;
	std::vector<cv::KeyPoint> _keyPoints;
	cv::Mat _descriptors;
};