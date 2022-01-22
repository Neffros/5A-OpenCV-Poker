#pragma once
#include "Card.h"

struct OffSets
{
    OffSets(int xOffset, int yOffset);

    int x;
    int y;
};
class PokerUtils
{
private:
    static OffSets getImageOffset(cv::Mat img);
    static const uint nbCardValues = 14;
public:
    static std::vector<Card> GetCardsFromImg(cv::Mat& cardsImg);

};