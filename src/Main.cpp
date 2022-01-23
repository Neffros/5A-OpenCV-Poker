#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../headers/PokerUtils.h"

int main()
{
    std::cout << "PROJECT1" << std::endl;
    std::string resourcePath = "../resources/";
    std::string cardsImgName = "cards.jpg";

    std::cout << resourcePath + cardsImgName << std::endl;
    cv::Mat cardsImg = cv::imread(resourcePath + cardsImgName);

    cv::imshow("cards",cardsImg);
    std::vector<Card> cards = PokerUtils::GetCardsFromImg(cardsImg);


    cv::imshow("card", cards[51].rawImg);
    cv::imshow("anothercard", cards[39].rawImg);

    cv::waitKey();

    return 0;
}