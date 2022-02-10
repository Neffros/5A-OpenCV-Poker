#include "../include/PokerAnalyzer.h"
#include "../include/PokerCardValue.h"
#include <stdexcept>
#include <fmt/format.h>

static const int NB_CHARACTER_FEATURES = 0;
static const int NB_SYMBOL_FEATURES = 1000;
static const int NB_TABLE_FEATURES = 100000;
static const int NB_CARD_VALUES = 14;
static const int RECT_X_OFFSET = 5;
static const int RECT_Y_OFFSET = 15;
static const float DISTANCE_COEFF = 0.65f;
static const int MIN_POINT_MATCHES = 15;

//#define USE_ORB
#define USE_SIFT

#define USE_BF
//#define USE_FLANN

PokerAnalyzer::PokerAnalyzer(const std::filesystem::path& cardValuesPath):
#if defined(USE_ORB)
	_characterFeatureDetector(cv::ORB::create(NB_CHARACTER_FEATURES == 0 ? 500 : NB_CHARACTER_FEATURES)),
	_symbolFeatureDetector(cv::ORB::create(NB_SYMBOL_FEATURES == 0 ? 500 : NB_SYMBOL_FEATURES)),
	_tableFeatureDetector(cv::ORB::create(NB_TABLE_FEATURES == 0 ? 500 : NB_TABLE_FEATURES))
#elif defined(USE_SIFT)
	_characterFeatureDetector(cv::SIFT::create(NB_CHARACTER_FEATURES)),
	_symbolFeatureDetector(cv::SIFT::create(NB_SYMBOL_FEATURES)),
	_tableFeatureDetector(cv::SIFT::create(NB_TABLE_FEATURES))
#endif
{
	loadCardValues(cardValuesPath);

#if defined(USE_BF) && defined(USE_ORB)
		_descriptorMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);
#elif defined(USE_BF) && defined(USE_SIFT)
		_descriptorMatcher = cv::BFMatcher::create(cv::NORM_L2);
#elif defined(USE_FLANN) && defined(USE_ORB)
		_descriptorMatcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
#elif defined(USE_FLANN) && defined(USE_SIFT)
		_descriptorMatcher = cv::FlannBasedMatcher::create();
#endif
	
//	trainMatchers();
}

//void PokerAnalyzer::loadPokerCards(const std::filesystem::path& cardImagePath)
//{
//	cv::Mat cardsImage = cv::imread(cardImagePath.generic_string());
//
//	Offset offset = getCardsImageOffset(cardsImage);
//
//	//std::cout << "xoffset: " << offsets.x << std::endl;
//	//std::cout << "yoffset: " << offsets.y << std::endl;
//
//	int cardTypeId = 0;
//	for(auto x = 0; x <= cardsImage.cols; x += offset.x)
//	{
//		if(x + offset.x > cardsImage.cols)
//			break;
//
//		int cardValueId = NB_CARD_VALUES;
//		for(auto y = 0; y <= cardsImage.rows; y+= offset.y)
//		{
//			if(y + offset.y > cardsImage.rows)
//				break;
//
//			cv::Mat cardImg = cardsImage(
//				cv::Rect(
//					x + RECT_X_OFFSET,
//					y + RECT_Y_OFFSET,
//					offset.x - RECT_X_OFFSET,
//					offset.y - RECT_Y_OFFSET
//			));
//
//			_cards.emplace_back(std::move(cardImg), *_cardFeatureDetector, static_cast<PokerCard::Type>(cardTypeId), static_cast<PokerCard::Value>(cardValueId));
//
//			cardValueId--;
//		}
//		cardTypeId++;
//	}
//}

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
	auto valueMatches = doValueMatches(table);
	
	std::vector<std::pair<const PokerCardValue*, std::vector<cv::DMatch>>> filteredMatches;
	
	for (auto& [value, matches] : valueMatches)
	{
		filteredMatches.emplace_back(std::make_pair(value, filterMatches(matches)));
	}
	
	for (auto& [value, matches] : filteredMatches)
	{
		drawValueMatches(table, *value, matches);
	}
	
//	std::vector<std::vector<cv::DMatch>> matches = doMatch(table);
//    drawTable(table, matches);
}

//std::vector<std::vector<cv::DMatch>> PokerAnalyzer::doMatch(const PokerTable& table)
//{
//	std::vector<std::vector<cv::DMatch>> allFilteredMatches;
//
//	for (int i = 0; i < _cardValueMatchData.size(); i++)
//	{
//		std::vector<std::vector<cv::DMatch>> matches;
//		_descriptorMatcher->knnMatch(table.getDescriptors(), _cardValueMatchData[i].blackValueImage->getDescriptors(), matches, 2);
//		allFilteredMatches[i] = filterMatches(matches);
//	}
//
//	return allFilteredMatches;
//}

std::vector<cv::DMatch> PokerAnalyzer::filterMatches(const std::vector<std::vector<cv::DMatch>>& matches)
{
	std::vector<cv::DMatch> filteredMatches;
	filteredMatches.reserve(matches.size());
	
	for(auto match : matches)
	{
		if (match[0].distance < DISTANCE_COEFF * match[1].distance)
		{
            filteredMatches.push_back(match[0]);
		}
	}
	
	filteredMatches.shrink_to_fit();
	
	return filteredMatches;
}

//const std::vector<PokerCard>& PokerAnalyzer::getCards() const
//{
//	return _cards;
//}

//void PokerAnalyzer::trainMatchers()
//{
//	for (int i = 0; i < _cards.size(); i++)
//	{
//		PokerCard& card = _cards[i];
//
//		_descriptorMatcher->add({card.getDescriptors()});
//		_descriptorMatcher->train();
//	}
//}

//void PokerAnalyzer::drawTable(const PokerTable& table, std::vector<std::vector<cv::DMatch>> allCardsMatches)
//{
//    cv::Mat outputImage = table.getPixelData().clone();
//
//	for (int i = 0; i < allCardsMatches.size(); i++)
//	{
//		if (allCardsMatches[i].size() >= MIN_POINT_MATCHES)
//		{
//			drawCardBindingBoxInTable(outputImage, table, _cards[i], allCardsMatches[i]);
//
//			cv::Mat outTemp;
//			cv::drawMatches(table.getPixelData(), table.getKeyPoints(), _cards[i].getPixelData(), _cards[i].getKeyPoints(), allCardsMatches[i], outTemp);
//			cv::imwrite(fmt::format("test_{:02d}_good.jpg", i), outTemp);
//		}
//		else
//		{
//			cv::Mat outTemp;
//			cv::drawMatches(table.getPixelData(), table.getKeyPoints(), _cards[i].getPixelData(), _cards[i].getKeyPoints(), allCardsMatches[i], outTemp);
//			cv::imwrite(fmt::format("test_{:02d}_bad.jpg", i), outTemp);
//		}
//	}
//
//    cv::imwrite("tableWithCards.jpg", outputImage);
//}

void PokerAnalyzer::drawCardBindingBoxInTable(cv::Mat& outputImage, const PokerTable& table, const PokerCard& card, std::vector<cv::DMatch> cardMatches)
{
	std::vector<cv::Point2f> cardPoints;
	std::vector<cv::Point2f> tablePoints;
	std::vector<cv::Point2f> tableEdges(4);

    for(auto match : cardMatches)
    {
        cardPoints.push_back(card.getKeyPoints()[match.trainIdx].pt);
        tablePoints.push_back(table.getKeyPoints()[match.queryIdx].pt);
    }
	
    std::cout<<"cardpoints: " << cardPoints.size() << std::endl;
    std::cout<<"table points: " << tablePoints.size() << std::endl;
	
    cv::Mat homography = cv::findHomography(cardPoints, tablePoints, cv::RANSAC);
	if (!homography.empty())
	{
		cv::Scalar randomColor(
			(double) std::rand() / RAND_MAX * 255,
			(double) std::rand() / RAND_MAX * 255,
			(double) std::rand() / RAND_MAX * 255
		);
		
		cv::perspectiveTransform(card.getImageEdges(), tableEdges, homography);
		
		cv::line(outputImage, tableEdges[0], tableEdges[1], randomColor, 5);
		cv::line(outputImage, tableEdges[1], tableEdges[2], randomColor, 5);
		cv::line(outputImage, tableEdges[2], tableEdges[3], randomColor, 5);
		cv::line(outputImage, tableEdges[3], tableEdges[0], randomColor, 5);
	}
}

void PokerAnalyzer::loadCardValues(const std::filesystem::path& cardImagePath)
{
	std::unordered_map<PokerCard::Value, const char*> valuesStr = {
		{PokerCard::Value::Ace, "a"},
		{PokerCard::Value::King, "k"},
		{PokerCard::Value::Queen, "q"},
		{PokerCard::Value::Jack, "j"},
		{PokerCard::Value::Ten, "10"},
		{PokerCard::Value::Nine, "9"},
		{PokerCard::Value::Eight, "8"},
		{PokerCard::Value::Seven, "7"},
		{PokerCard::Value::Six, "6"},
		{PokerCard::Value::Five, "5"},
		{PokerCard::Value::Four, "4"},
		{PokerCard::Value::Three, "3"},
		{PokerCard::Value::Two, "2"}
	};
	
	for (auto [value, valueStr] : valuesStr)
	{
		std::filesystem::path blackValueFilePath = cardImagePath / fmt::format("{}_black.png", valueStr);
		std::filesystem::path redValueFilePath = cardImagePath / fmt::format("{}_red.png", valueStr);
		
		_cardValues.emplace_back(cv::imread(blackValueFilePath.generic_string()), *_characterFeatureDetector, value);
		_cardValues.emplace_back(cv::imread(redValueFilePath.generic_string()), *_characterFeatureDetector, value);
	}
}

void PokerAnalyzer::loadCardSymbols(const std::filesystem::path& cardImagePath)
{
	/*std::vector<const char*> patterns = {
		""
	};*/
	
}

std::vector<std::pair<const PokerCardValue*, std::vector<std::vector<cv::DMatch>>>> PokerAnalyzer::doValueMatches(const PokerTable& table) const
{
	std::vector<std::pair<const PokerCardValue*, std::vector<std::vector<cv::DMatch>>>> result;
	
	for (const PokerCardValue& value : _cardValues)
	{
		std::vector<std::vector<cv::DMatch>> matches;
		_descriptorMatcher->knnMatch(table.getDescriptors(), value.getDescriptors(), matches, 2);
		result.emplace_back(std::make_pair(&value, std::move(matches)));
	}
	
	return result;
}

void PokerAnalyzer::drawValueMatchesUnfiltered(const PokerTable& table, const PokerCardValue& value, std::vector<std::vector<cv::DMatch>> matches) const
{
	static int i = 0;
	cv::Mat outTemp;
	cv::drawMatches(table.getPixelData(), table.getKeyPoints(), value.getPixelData(), value.getKeyPoints(), matches, outTemp);
	cv::imwrite(fmt::format("test_{:02d}.jpg", i++), outTemp);
}

void PokerAnalyzer::drawValueMatches(const PokerTable& table, const PokerCardValue& value, std::vector<cv::DMatch> matches) const
{
	static int i = 0;
	cv::Mat outTemp;
	cv::drawMatches(table.getPixelData(), table.getKeyPoints(), value.getPixelData(), value.getKeyPoints(), matches, outTemp);
	cv::imwrite(fmt::format("test_{:02d}.jpg", i++), outTemp);
}
