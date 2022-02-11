#include "../include/PokerCard.h"

PokerCard::PokerCard(cv::Mat&& originalPixelData, cv::Mat&& preprocessedPixelData, cv::FeatureDetector& featureDetector, Type cardType, Value cardValue):
Image(std::forward<cv::Mat>(originalPixelData), std::forward<cv::Mat>(preprocessedPixelData), featureDetector), _cardType(cardType), _cardValue(cardValue)
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
const std::vector<cv::Point2f> PokerCard::getImageEdges() const
{
    std::vector<cv::Point2f> res(4);

    res[0] = cv::Point2f(0, 0);
    res[1] = cv::Point2f(_preprocessedPixelData.cols, 0);
    res[2] = cv::Point2f(_preprocessedPixelData.cols, _preprocessedPixelData.rows);
    res[3] = cv::Point2f(0, _preprocessedPixelData.rows);
    return res;
}