#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/PokerTable.h"
#include "../include/PokerAnalyzer.h"

#include <filesystem>
#include <fmt/format.h>

int main()
{
    std::filesystem::path resourcePath = "resources";
	
	PokerAnalyzer pokerAnalyzer(resourcePath / "characters");
	
	std::vector<PokerTable> pokerTables;
	for (const auto& entry : std::filesystem::directory_iterator(resourcePath / "pokerTables"))
	{
		std::filesystem::path path = entry.path();
		
		std::string ext = path.extension().generic_string();
		if (ext != ".png" && ext != ".jpeg" && ext != ".BMP" && ext != ".TGA" && ext != ".jpg")
		{
			std::cout << fmt::format("{} is not a compatible image", path.generic_string().c_str()) << std::endl;
			continue;
		}
		
		pokerTables.emplace_back(pokerAnalyzer.loadPokerTable(cv::imread(path.generic_string())));
		break; //TODO: remove
	}
	
//	for (const PokerCard& card : pokerAnalyzer.getCards())
//	{
//		if (card.getType() == PokerCard::Type::Clubs && card.getValue() == PokerCard::Value::Two)
//		{
//			cv::imshow("card image", card.getPixelData());
//
//			cv::Mat output;
//			cv::drawKeypoints(card.getPixelData(), card.getKeyPoints(), output);
//			cv::imshow("card keypoints", output);
//		}
//	}
	
	pokerAnalyzer.analyze(pokerTables[0]);

    cv::waitKey();

    return 0;
}