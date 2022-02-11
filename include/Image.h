#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

class Image
{
public:
    Image(cv::Mat&& originalPixelData, cv::Mat&& preprocessedPixelData, cv::FeatureDetector& featureDetector);
	virtual ~Image() = default;
	
	const cv::Mat& getOriginalPixelData() const;
	const cv::Mat& getPreprocessedPixelData() const;
	const std::vector<cv::KeyPoint>& getKeyPoints() const;
	const cv::Mat& getDescriptors() const;

protected:
    cv::Mat _originalPixelData;
	cv::Mat _preprocessedPixelData;
	std::vector<cv::KeyPoint> _keyPoints;
	cv::Mat _descriptors;
};