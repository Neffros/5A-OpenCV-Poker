#include "../include/PokerTable.h"

PokerTable::PokerTable(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector):
Image(std::forward<cv::Mat>(pixelData), &featureDetector)
{

}
