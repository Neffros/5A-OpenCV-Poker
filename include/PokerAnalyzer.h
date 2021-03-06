#pragma once

#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "PokerTable.h"
#include "PokerCard.h"

class PokerAnalyzer
{
public:
	PokerAnalyzer(const std::filesystem::path& cardImagePath);
	
	PokerTable loadPokerTable(cv::Mat&& tableImage) const;
	
	void analyze(const PokerTable& table);

	const std::vector<PokerCard>& getCards() const;

    static cv::Mat preprocessCard(const cv::Mat& img);

    static cv::Mat preprocessTable(const cv::Mat& img);
	
private:
	struct Offset
	{
		int x;
		int y;
	};
	
	std::vector<PokerCard> _cards;
	
	cv::Ptr<cv::FeatureDetector> _cardFeatureDetector;
	cv::Ptr<cv::FeatureDetector> _tableFeatureDetector;
	std::vector<cv::Ptr<cv::DescriptorMatcher>> _descriptorMatchers;
	
	void loadPokerCards(const std::filesystem::path& cardImagePath);
	void trainMatchers();
	
	std::vector<std::vector<cv::DMatch>> doMatch(const PokerTable& table);
	std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch>>& matches);
	
	static Offset getCardsImageOffset(const cv::Mat& cardsImage);
	void drawTable(const PokerTable& table, std::vector<std::vector<cv::DMatch>> allCardsMatches);
	static void drawCardBindingBoxInTable(cv::Mat& outputImage, const PokerTable& table, const PokerCard& card, std::vector<cv::DMatch> cardMatches);
};
