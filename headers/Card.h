#pragma once
#include <opencv2/core/core.hpp>

enum class CardType : unsigned short {
    Clubs = 0, Spades = 1, Diamonds = 2, Hearts = 3
};

enum class CardValue {
    Ace = 14,
    King = 13,
    Queen = 12,
    Jack = 11,
    Ten = 10,
    Nine = 9,
    Eight = 8,
    Seven = 7,
    Six = 6,
    Five = 5,
    Four = 4,
    Three = 3,
    Two = 2
};
class Card {
private:
    cv::Mat cardImg;
    CardType cardType;
    CardValue cardValue;
public:
    cv::Mat rawImg;
    explicit Card(const cv::Mat& img);
    Card(const cv::Mat& img, uint cardType, uint cardValue);
};
