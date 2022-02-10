#include "../include/PokerCardValue.h"

PokerCardValue::PokerCardValue(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, PokerCard::Value value):
Image(std::forward<cv::Mat>(pixelData)), _value(value)
{
	int width = _pixelData.size().width;
	int height = _pixelData.size().height;
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			_keyPoints.emplace_back(static_cast<float>(x), static_cast<float>(y), 5.0f);
		}
	}
	
	featureDetector.compute(_pixelData, _keyPoints, _descriptors);
}

PokerCard::Value PokerCardValue::getValue() const
{
	return _value;
}
