#pragma once
#include <vector>
#include "Card.h"
class PokerUtils
{
private:
public:
    std::vector<Card> GetCardsFromImg(cv::Mat cardsImg);

};