#pragma once

#include <utility>

#include "../include/PokerCard.h"

class SolutionAResult {
public:
    std::vector<PokerCard> cardsInImage;

    explicit SolutionAResult(std::vector<PokerCard> cards) : cardsInImage(std::move(cards)){

    }
};