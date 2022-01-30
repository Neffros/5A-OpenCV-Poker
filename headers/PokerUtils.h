#pragma once
#include "Card.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

struct OffSets
{
    OffSets(int xOffset, int yOffset);

    int x;
    int y;
};
class PokerUtils
{
private:
    static OffSets getImageOffset(cv::Mat img);
    static const uint nbCardValues = 14;

    //offsets amount of pixels to not store in card rawImg
    static const uint rectXOffset = 5;
    static const uint rectYOffSet = 15;
    constexpr static const float distanceCoeff = 0.25;
    static const uint minPointMatches = 7;
    //static cv::Ptr<cv::ORB> orbCards;
    //static cv::Ptr<cv::ORB> orbPokerTables;
public:
    static const int nbCardOrbTries = 1000;
    static const int nbTableOrbTries = 1000;
    static std::vector<Card> GetCardsFromImg(cv::Mat& cardsImg);
    static std::vector<cv::Mat> GetAllImagesInPath(const std::string& path);
    static std::vector<cv::Mat> GetCardsDescriptors(std::vector<Card>& cards);
    static cv::Mat GetImageDescriptor(Image& img, const cv::Ptr<cv::ORB>& orb);
    static std::vector<std::vector<cv::DMatch>> GetCardMatchesInTable(Image& pokerTable, cv::BFMatcher& bfMatcher, const std::vector<Card>& cards);
    static std::vector<std::vector<cv::DMatch>> GetFilteredMatches(const std::vector<std::vector<cv::DMatch>>& matches, const std::vector<Card>& cards);
    static void drawTable(const Image& pokerTable, std::vector<std::vector<cv::DMatch>> allCardsMatches);
    static void drawCardBindingBoxInTable(const cv::Mat& pokerTable, std::vector<cv::DMatch> cardMatches);

};