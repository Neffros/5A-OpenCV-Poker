#pragma once

#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "PokerTable.h"
#include "PokerCard.h"
#include "PokerCardValue.h"

class PokerAnalyzer
{
public:
	PokerAnalyzer(const std::filesystem::path& cardValuesPath);
	
	PokerTable loadPokerTable(cv::Mat&& tableImage) const;
	
	void analyze(const PokerTable& table);
	
//	const std::vector<PokerCard>& getCards() const;
	
private:
	struct Offset
	{
		int x;
		int y;
	};
	
//	std::vector<PokerCard> _cards;
    std::vector<PokerCardValue> _cardValues;
	
	cv::Ptr<cv::FeatureDetector> _characterFeatureDetector;
	cv::Ptr<cv::FeatureDetector> _symbolFeatureDetector;
	cv::Ptr<cv::FeatureDetector> _tableFeatureDetector;
	cv::Ptr<cv::DescriptorMatcher> _descriptorMatcher;
	
	void loadCardValues(const std::filesystem::path& cardImagePath);
	void loadCardSymbols(const std::filesystem::path& cardImagePath);
//	void loadPokerCards(const std::filesystem::path& cardImagePath);
//	void trainMatchers();
	
	std::vector<std::pair<const PokerCardValue*, std::vector<std::vector<cv::DMatch>>>> doValueMatches(const PokerTable& table) const;
//	std::vector<std::vector<cv::DMatch>> doMatch(const PokerTable& table);
	std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch>>& matches);
	
	static Offset getCardsImageOffset(const cv::Mat& cardsImage);
//	void drawTable(const PokerTable& table, std::vector<std::vector<cv::DMatch>> allCardsMatches);
	void drawValueMatchesUnfiltered(const PokerTable& table, const PokerCardValue& value, std::vector<std::vector<cv::DMatch>> matches) const;
	void drawValueMatches(const PokerTable& table, const PokerCardValue& value, std::vector<cv::DMatch> matches) const;
	static void drawCardBindingBoxInTable(cv::Mat& outputImage, const PokerTable& table, const PokerCard& card, std::vector<cv::DMatch> cardMatches);
};
