#include "../include/PokerCard.h"

PokerCard::PokerCard(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector, CardType cardType, CardValue cardValue):
Image(std::forward<cv::Mat>(pixelData), featureDetector), _cardType(cardType), _cardValue(cardValue)
{

}

CardType PokerCard::getType() const
{
	return _cardType;
}

CardValue PokerCard::getValue() const
{
	return _cardValue;
}