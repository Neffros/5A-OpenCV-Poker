#include "../include/PokerAnalyzer.h"
#include <stdexcept>

static const int NB_CARD_FEATURES = 20000;
static const int NB_TABLE_FEATURES = 20000;
static const int NB_CARD_VALUES = 14;
static const int RECT_X_OFFSET = 5;
static const int RECT_Y_OFFSET = 15;
static const float DISTANCE_COEFF = 0.65f;
static const int MIN_POINT_MATCHES = 10;

PokerAnalyzer::PokerAnalyzer(const std::filesystem::path& cardImagePath):
_cardFeatureDetector(cv::ORB::create(NB_CARD_FEATURES)),
_tableFeatureDetector(cv::ORB::create(NB_TABLE_FEATURES)),
_descriptorMatcher(cv::BFMatcher::create())
{
	loadPokerCards(cardImagePath);
	trainMatcher();
}

void PokerAnalyzer::loadPokerCards(const std::filesystem::path& cardImagePath)
{
	cv::Mat cardsImage = cv::imread(cardImagePath.generic_string());
	
	Offset offset = getCardsImageOffset(cardsImage);
	
	//std::cout << "xoffset: " << offsets.x << std::endl;
	//std::cout << "yoffset: " << offsets.y << std::endl;
	
	int cardTypeId = 0;
	for(auto x = 0; x <= cardsImage.cols; x += offset.x)
	{
		if(x + offset.x > cardsImage.cols)
			break;
		
		int cardValueId = NB_CARD_VALUES;
		for(auto y = 0; y <= cardsImage.rows; y+= offset.y)
		{
			if(y + offset.y > cardsImage.rows)
				break;
			
			cv::Mat cardImg = cardsImage(
				cv::Rect(
					x + RECT_X_OFFSET,
					y + RECT_Y_OFFSET,
					offset.x - RECT_X_OFFSET,
					offset.y - RECT_Y_OFFSET
			));
			
			_cards.emplace_back(std::move(cardImg), *_cardFeatureDetector, static_cast<PokerCard::Type>(cardTypeId), static_cast<PokerCard::Value>(cardValueId));
			
			cardValueId--;
		}
		cardTypeId++;
	}
}

PokerAnalyzer::Offset PokerAnalyzer::getCardsImageOffset(const cv::Mat& cardsImage)
{
	return {cardsImage.cols / 4 , cardsImage.rows / 13};
}

PokerTable PokerAnalyzer::loadPokerTable(cv::Mat&& tableImage) const
{
	return PokerTable(std::forward<cv::Mat>(tableImage), *_tableFeatureDetector);
}

void PokerAnalyzer::analyze(const PokerTable& table)
{
	std::vector<std::vector<cv::DMatch>> matches = doMatch(table);
    drawTable(table, matches);
}

std::vector<std::vector<cv::DMatch>> PokerAnalyzer::doMatch(const PokerTable& table)
{
	std::vector<std::vector<cv::DMatch>> matches;
	_descriptorMatcher->knnMatch(table.getDescriptors(), matches, 2);
	
	return getFilteredMatches(matches);
}

std::vector<std::vector<cv::DMatch>> PokerAnalyzer::getFilteredMatches(const std::vector<std::vector<cv::DMatch>>& matches)
{
	std::vector<std::vector<cv::DMatch>> filteredMatches(_cards.size());
	
	for(auto match : matches)
	{
		if (match[0].distance < DISTANCE_COEFF * match[1].distance)
		{
            filteredMatches[match[0].imgIdx].push_back(match[0]);
		}
	}
	
	return filteredMatches;
}

const std::vector<PokerCard>& PokerAnalyzer::getCards() const
{
	return _cards;
}

void PokerAnalyzer::trainMatcher()
{
	std::vector<cv::Mat> cardsDescriptors;
	for (PokerCard& card : _cards)
	{
		cardsDescriptors.emplace_back(card.getDescriptors());
	}
	
	_descriptorMatcher->add(cardsDescriptors);
	_descriptorMatcher->train();
}

void PokerAnalyzer::drawTable(const PokerTable& table, std::vector<std::vector<cv::DMatch>> allCardsMatches)
{
    PokerTable res = table;
	
	for (auto & cardMatches : allCardsMatches)
	{
		if (cardMatches.size() >= MIN_POINT_MATCHES)
		{

			drawCardBindingBoxInTable(res, cardMatches);
		}
	}

    cv::imwrite("tableWithCards.jpg", res.getPixelData());
}

void PokerAnalyzer::drawCardBindingBoxInTable(const PokerTable& outputImage, std::vector<cv::DMatch> cardMatches)
{
	std::vector<cv::Point2f> cardPoints;
	std::vector<cv::Point2f> tablePoints;
	std::vector<cv::Point2f> tableEdges(4);

    const int index = cardMatches[0].imgIdx;
    std::cout << "index: " << index << std::endl;
    for(auto match : cardMatches)
    {
        cardPoints.push_back(_cards[index].getKeyPoints()[match.trainIdx].pt);
        tablePoints.push_back(outputImage.getKeyPoints()[match.queryIdx].pt);
    }
    cv::Scalar randomColor(
            (double)std::rand() / RAND_MAX * 255,
            (double)std::rand() / RAND_MAX * 255,
            (double)std::rand() / RAND_MAX * 255
    );
    std::cout<<"cardpoints: " << cardPoints.size() << std::endl;
    std::cout<<"table points: " << tablePoints.size() << std::endl;
    /*for(auto match : cardMatches)
    {
        cardPoints.push_back(_cards[match.trainIdx].getKeyPoints()[match.trainIdx].pt);
        tablePoints.push_back(outputImage.getKeyPoints()[match.queryIdx].pt);
    }*/
    cv::Mat homography = cv::findHomography(cardPoints, tablePoints);
    cv::perspectiveTransform(_cards[index].getImageEdges(), tableEdges, homography);

    cv::line(outputImage.getPixelData(), tableEdges[0], tableEdges[1], randomColor, 5);
    cv::line(outputImage.getPixelData(), tableEdges[1], tableEdges[2], randomColor, 5);
    cv::line(outputImage.getPixelData(), tableEdges[2], tableEdges[3], randomColor, 5);
    cv::line(outputImage.getPixelData(), tableEdges[3], tableEdges[0], randomColor, 5);

}
