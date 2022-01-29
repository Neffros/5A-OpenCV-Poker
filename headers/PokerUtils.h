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


    static cv::Ptr<cv::ORB> orbCards;
    static cv::Ptr<cv::ORB> orbPokerTables;
public:
    static const int nbCardOrbTries = 1000;
    static std::vector<Card> GetCardsFromImg(cv::Mat& cardsImg);
    static std::vector<cv::Mat> GetAllImagesInPath(const std::string& path);
    static std::vector<cv::Mat> GetCardsDescriptors(std::vector<Card>& cards);
    static void GetCardsInTable(std::vector<Card>& cards, std::vector<cv::Mat> pokerTables);


};