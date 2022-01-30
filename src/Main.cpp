#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../headers/PokerUtils.h"

#include <filesystem>

int main()
{
    std::filesystem::path resourcePath = "resources";
	std::filesystem::path pokerTablesFolder = "pokerTables";
	std::filesystem::path cardsImgName = "cards.jpg";

    //get poker tables
    std::vector<cv::Mat> pokerTablesMat = PokerUtils::GetAllImagesInPath(resourcePath / pokerTablesFolder);
    std::vector<Image> pokerTables;

    for(const auto& mat : pokerTablesMat)
    {
        Image table(mat);
        pokerTables.push_back(table);
    }

    //get cardsImg
    cv::Mat cardsImg = cv::imread((resourcePath / cardsImgName).generic_string());

    //get vector of all cards
    std::vector<Card> cards = PokerUtils::GetCardsFromImg(cardsImg);

    cv::imshow("card", cards[51].rawImg);
    cv::imshow("anothercard", cards[39].rawImg);

    std::vector<cv::Mat> cardDescriptors = PokerUtils::GetCardsDescriptors(cards);
    cv::BFMatcher bfMatcher;
    bfMatcher.add(cardDescriptors);
    bfMatcher.train();

    std::vector<std::vector<cv::DMatch>> filteredMatches(cards.size());

    //cv::imshow("poker table", pokerTables[0].rawImg);

    //for(auto table : pokerTables)
    //{
        filteredMatches = PokerUtils::GetCardMatchesInTable(pokerTables[0], bfMatcher, cards);
    //}
    //PokerUtils::drawTable(pokerTables[0], filteredMatches);

    cv::waitKey();

    return 0;
}