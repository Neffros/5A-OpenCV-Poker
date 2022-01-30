#include "../include/PokerCard.h"

PokerCard::PokerCard(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, Type cardType, Value cardValue):
Image(std::forward<cv::Mat>(pixelData), featureDetector), _cardType(cardType), _cardValue(cardValue)
{

}

PokerCard::Type PokerCard::getType() const
{
	return _cardType;
}

PokerCard::Value PokerCard::getValue() const
{
	return _cardValue;
}