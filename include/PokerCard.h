#pragma once

#include "Image.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class PokerCard : public Image
{
public:
	enum class Type : unsigned short
	{
		Clubs = 0,
		Spades = 1,
		Diamonds = 2,
		Hearts = 3
	};
	
	enum class Value
	{
		Ace = 14,
		King = 13,
		Queen = 12,
		Jack = 11,
		Ten = 10,
		Nine = 9,
		Eight = 8,
		Seven = 7,
		Six = 6,
		Five = 5,
		Four = 4,
		Three = 3,
		Two = 2
	};
	
	PokerCard(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, Type cardType, Value cardValue);
	
	Type getType() const;
	Value getValue() const;

private:
	Type _cardType;
	Value _cardValue;
};
