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

std::vector<cv::Mat> PokerUtils::GetCardsDescriptors(std::vector<Card> &cards) {
    cv::Ptr<cv::ORB> orbCards = cv::ORB::create(nbCardOrbTries);
    std::vector<cv::Mat> descriptors;

    for(auto card : cards)
    {
        orbCards->detect(card.rawImg, card.objectKeyPoints);
        orbCards->compute(card.rawImg, card.objectKeyPoints, card.descriptor);
        descriptors.push_back(card.descriptor);
    }
    //get descriptors of each card
    for(auto card : cards)
    {
        cv::Mat descriptor = GetImageDescriptor(card, orbCards);
        descriptors.push_back(descriptor);
    }
    return descriptors;
}

cv::Mat PokerUtils::GetImageDescriptor(Image &img, const cv::Ptr<cv::ORB>& orb) {

    orb->detect(img.rawImg, img.objectKeyPoints);
    orb->compute(img.rawImg, img.objectKeyPoints, img.descriptor);

    return img.descriptor;
}

std::vector<std::vector<cv::DMatch>>
PokerUtils::GetCardMatchesInTable(Image &pokerTable, cv::BFMatcher& bfMatcher, const std::vector<Card>& cards) {

    std::vector<std::vector<cv::DMatch>> matches;
    cv::Ptr<cv::ORB> orbPokerTables = cv::ORB::create(nbTableOrbTries);

    GetImageDescriptor(pokerTable, orbPokerTables);
    //!!! verify if descriptor is filled in properly or get a reference
    bfMatcher.knnMatch(pokerTable.descriptor, matches, 2);

    return GetFilteredMatches(matches, cards);

}

std::vector<std::vector<cv::DMatch>> PokerUtils::GetFilteredMatches(const std::vector<std::vector<cv::DMatch>>& matches, const std::vector<Card>& cards) {

    std::vector<std::vector<cv::DMatch>> filteredMatches(2);

    filteredMatches.resize(matches.size());
    //std::vector<std::vector<cv::DMatch>> filteredMatches(2);
    for(auto match : matches)
    {
        std::cout << "match 0: " << match[0].distance << std::endl;
        std::cout << "match 1: " << match[1].distance << std::endl;
        if (match[0].distance < distanceCoeff * match[1].distance)
        {
            std::cout << "good match" << std::endl;
            filteredMatches[match[0].imgIdx].push_back(match[0]);
        }
    }

    return filteredMatches;
}

void PokerUtils::drawTable(const Image &pokerTable, std::vector<std::vector<cv::DMatch>> allCardsMatches) {
    cv::Mat res = pokerTable.rawImg.clone();

    for (auto & cardMatches : allCardsMatches)
    {
        //std::cout << cardMatches.size() << std::endl;
        if (cardMatches.size() > minPointMatches)
        {
            drawCardBindingBoxInTable(res, cardMatches);
        }
    }
}

void PokerUtils::drawCardBindingBoxInTable(const cv::Mat &pokerTable, std::vector<cv::DMatch> cardMatches) {

    std::vector<cv::Point2f> cardPoints;
    std::vector<cv::Point2f> tablePoints;
    std::vector<cv::Point2f> tableEdges(4);

    /*
    for (auto match : cardMatches)
    {
        cardPoints.push_back(objectsKeyPoints[index][match.trainIdx].pt); //reverse query and trainidx?
        scenePoints.push_back(sceneKeyPoints[match.queryIdx].pt);
    }

    //detemrine la matrice de transfomration entre l'image et l'objet dans la scène
    cv::Mat homography = cv::findHomography(objectPoints, scenePoints);
    cv::perspectiveTransform(objectsEdges[index], sceneEdges, homography);

    //dessine le carré autour de l'image dans la vidéo
    cv::line(image, sceneEdges[0], sceneEdges[1], colors[index], 5);
    cv::line(image, sceneEdges[1], sceneEdges[2], colors[index], 5);
    cv::line(image, sceneEdges[2], sceneEdges[3], colors[index], 5);
    cv::line(image, sceneEdges[3], sceneEdges[0], colors[index], 5);*/
}


OffSets::OffSets(int xOffset, int yOffset) {
    this->x = xOffset;
    this->y = yOffset;
}
