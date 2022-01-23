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

    //offsets amount of pixels to not store in card rawImg
    static const uint rectXOffset = 5;
    static const uint rectYOffSet = 15;
public:
    static std::vector<Card> GetCardsFromImg(cv::Mat& cardsImg);

};