#include "../headers/Card.h"
#include "../headers/PokerUtils.h"


Card::Card(const cv::Mat &img, const uint cardType, const uint cardValue) {
    this->rawImg = img;
    //this->cardImg = img;
    this->cardType = CardType(cardType);
    this->cardValue= CardValue(cardValue);
}


