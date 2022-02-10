#pragma once

#include "PokerCard.h"

class PokerCardValue : public Image
{
public:
	PokerCardValue(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, PokerCard::Value value);
	
	PokerCard::Value getValue() const;
	
private:
	PokerCard::Value _value;
};