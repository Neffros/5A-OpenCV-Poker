#pragma once

#include "Image.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

enum class CardType : unsigned short
{
	Clubs = 0,
	Spades = 1,
	Diamonds = 2,
	Hearts = 3
};

enum class CardValue
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

class PokerCard : public Image
{
public:
	PokerCard(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, CardType cardType, CardValue cardValue);
	
	CardType getType() const;
	CardValue getValue() const;

private:
	CardType _cardType;
	CardValue _cardValue;
};
