#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../headers/PokerUtils.h"


int main()
{
    std::string resourcePath = "../resources/";
    std::string pokerTablesFolder = "pokerTables";
    std::string cardsImgName = "cards.jpg";

    //get poker tables
    std::vector<cv::Mat> pokerTables = PokerUtils::GetAllImagesInPath(resourcePath + pokerTablesFolder);
    std::vector<Image> pokerTable;
    //get cardsImg
    cv::Mat cardsImg = cv::imread(resourcePath + cardsImgName);

    //get vector of all cards
    std::vector<Card> cards = PokerUtils::GetCardsFromImg(cardsImg);

    cv::imshow("card", cards[51].rawImg);
    cv::imshow("anothercard", cards[39].rawImg);


    /*std::vector<cv::Mat> descriptors = PokerUtils::GetCardsDescriptors(cards);
    cv::BFMatcher bfMatcher;
    bfMatcher.add(descriptors);
    bfMatcher.train();*/



    cv::waitKey();

    return 0;
}