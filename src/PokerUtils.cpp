#include "../headers/PokerUtils.h"
#include <iostream>
#include <filesystem>

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

std::vector<cv::Mat> PokerUtils::GetAllImagesInPath(const std::string& path) {
    std::vector<cv::Mat> images;
    int i = 0;
    std::cout << "getting images at path: "<< path << std::endl;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpeg" && ext != ".BMP" && ext != ".TGA" && ext != ".jpg")
        {
            std::cout << entry.path().string().c_str() << " is not a compatible image" << std::endl;;
            continue;
        }
        cv::Mat im;
        im = cv::imread(entry.path().string());
        images.push_back(im);
        i++;
    }
    return images;
}

/*std::vector<cv::Mat> PokerUtils::GetCardsDescriptors(std::vector<Card> &cards) {

    std::vector<cv::Mat> descriptors;

    //get descriptors of each card
    for(auto card : cards)
    {
        orbCards->detect(card.rawImg, card.objectKeyPoints);
        orbCards->compute(card.rawImg, card.objectKeyPoints, card.descriptor);
        descriptors.push_back(card.descriptor);
    }
    return descriptors;
}*/

void PokerUtils::GetCardsInTable(std::vector<Card> &cards, std::vector<cv::Mat> pokerTables) {

    /*for(auto pokerTable : pokerTables)
    {
        orbPokerTables->detect(pokerTable, )
    }*/

}

OffSets::OffSets(int xOffset, int yOffset) {
    this->x = xOffset;
    this->y = yOffset;
}
