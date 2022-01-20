#pragma once
#include <opencv2/core/core.hpp>

enum class CardType {
    Diamonds, Hearts, Clubs, Spades
};

enum class CardValue {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King
};
class Card {
private:
    cv::Mat cardImg;
    cv::Mat rawImg;
    CardType cardType;
    CardValue cardValue;
public:
    void Test();
};
