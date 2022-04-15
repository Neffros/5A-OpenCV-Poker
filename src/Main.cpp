#include "Main.h"
#include <iostream>

#include <opencv2/core/core.hpp>

#include "../include/PokerTable.h"
#include "../include/PokerAnalyzer.h"

#include <filesystem>
#include <fmt/format.h>

namespace solA {
    SolutionAResult compute(const cv::Mat &testImage) {
        std::filesystem::path resourcePath = "resources";
        std::filesystem::path pokerTablesFolder = "pokerTables";
        std::filesystem::path cardsImgName = "cards.jpg";

        PokerAnalyzer pokerAnalyzer(resourcePath / cardsImgName);

        //std::vector<PokerTable> pokerTables;
        /*for (const auto &entry: std::filesystem::directory_iterator(resourcePath / pokerTablesFolder)) {
            std::filesystem::path path = entry.path();

            std::string ext = path.extension().generic_string();
            if (ext != ".png" && ext != ".jpeg" && ext != ".BMP" && ext != ".TGA" && ext != ".jpg") {
                std::cout << fmt::format("{} is not a compatible image", path.generic_string().c_str()) << std::endl;
                continue;
            }

            pokerTables.emplace_back(pokerAnalyzer.loadPokerTable(cv::imread(path.generic_string())));
            //break; //TODO: remove
        }*/

        /*for (const PokerCard& card : pokerAnalyzer.getCards())
        {
            if (card.getType() == PokerCard::Type::Clubs && card.getValue() == PokerCard::Value::Two)
            {
                cv::imshow("card image", card.getOriginalPixelData());

                cv::Mat output;
                cv::drawKeypoints(card.getOriginalPixelData(), card.getKeyPoints(), output);
                cv::imshow("card keypoints", output);
            }
            if(card.getType() == PokerCard::Type::Clubs && card.getValue() == PokerCard::Value::Nine)
            {
                cv::imshow("processedCard", card.getPreprocessedPixelData());
            }
        }*/

        pokerAnalyzer.analyze(pokerAnalyzer.loadPokerTable(std::move(testImage.clone())));
        //cv::imwrite("processedTable.jpg", pokerTables[0].getPreprocessedPixelData());

        //cv::waitKey();

        SolutionAResult results(pokerAnalyzer.getCards());
        return results;
    }
}