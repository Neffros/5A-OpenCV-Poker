#include "../include/PokerCardValue.h"

PokerCardValue::PokerCardValue(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, PokerCard::Value value):
Image(std::forward<cv::Mat>(pixelData), &featureDetector), _value(value)
{
//	int width = _pixelData.size().width;
//	int height = _pixelData.size().height;
//
//	int step = 5;
//
//	for (int y = 0; y < height - step; y += step)
//	{
//		for (int x = 0; x < width - step; x += step)
//		{
//			_keyPoints.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(step));
//		}
//	}
//
//	featureDetector.compute(_pixelData, _keyPoints, _descriptors);
}

PokerCard::Value PokerCardValue::getValue() const
{
	return _value;
}
