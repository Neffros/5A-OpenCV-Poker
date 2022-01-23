#include "../headers/PokerUtils.h"
#include <iostream>

///
/// \param cardsImg image of all the cards
/// \return a list of all the cards seperate from each other
std::vector<Card> PokerUtils::GetCardsFromImg(cv::Mat& cardsImg) {

    std::vector<Card> cards;
    OffSets offsets = getImageOffset(cardsImg);

    //std::cout << "xoffset: " << offsets.x << std::endl;
    //std::cout << "yoffset: " << offsets.y << std::endl;
    unsigned int cardTypeId = 0;
    unsigned int cardValueId = nbCardValues;

    for(auto x = 0; x <= cardsImg.cols; x += offsets.x)
    {
        if(x + offsets.x > cardsImg.cols)
            break;

        for(auto y = 0; y <= cardsImg.rows; y+= offsets.y)
        {
            if(y + offsets.y > cardsImg.rows)
                break;
            //cv::Mat cardImg = cardsImg(cv::Rect(x,y,offsets.x, offsets.y));
            cv::Mat cardImg = cardsImg(cv::Rect(x + rectXOffset,y + rectYOffSet,
                                       offsets.x - rectXOffset, offsets.y - rectYOffSet));
            Card card(cardImg, cardTypeId, cardValueId);
            cards.push_back(card);

            cardValueId--;
        }
        ++cardTypeId;
        cardValueId = nbCardValues;
    }

    return cards;
}

OffSets PokerUtils::getImageOffset(cv::Mat img) {

    return {img.cols / 4 , img.rows / 13};
}

OffSets::OffSets(int xOffset, int yOffset) {
    this->x = xOffset;
    this->y = yOffset;
}
