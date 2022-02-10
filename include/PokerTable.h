#pragma once

#include "Image.h"

class PokerTable : public Image
{
public:
	PokerTable(cv::Mat&& pixelData, cv::FeatureDetector& featureDetector);
};
