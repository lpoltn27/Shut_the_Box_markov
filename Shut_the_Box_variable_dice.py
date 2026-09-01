
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from collections import Counter
from typing import Protocol
import random
import numpy as np

N_TILES = 9
MAX_DICE = 7

ALL_OPEN = (1 << N_TILES) - 1
NUM_STATES = 1 << N_TILES
LOSE_STATE = NUM_STATES
NUM_MATRIX_STATES = NUM_STATES + 1

@lru_cache(maxsize=None)
def dice_sum_probabilities(num_dice: int) -> dict[int, float]:
    outcomes = product(range(1, 7), repeat=num_dice)

    counts = Counter(
        sum(outcome)
        for outcome in outcomes
    )

    total = 6 ** num_dice

    return {
        dice_sum: count / total
        for dice_sum, count in sorted(counts.items())
    }

DICE_OPTIONS = {
    num_dice: dice_sum_probabilities(num_dice)
    for num_dice in range(1, MAX_DICE + 1)
}


def tile_to_bit(tile: int) -> int:
 
    return 1 << (tile - 1)


def tiles_to_mask(tiles) -> int:
 
    mask = 0

    for tile in tiles:
        mask |= tile_to_bit(tile)

    return mask


@lru_cache(maxsize=None)
def mask_to_tiles(mask: int) -> tuple[int, ...]:

    return tuple(
        tile
        for tile in range(1, N_TILES + 1)
        if mask & tile_to_bit(tile)
    )


@lru_cache(maxsize=None)
def mask_sum(mask: int) -> int:


    return sum(mask_to_tiles(mask))


def close_tiles(open_mask: int, move_mask: int) -> int:


    return open_mask & ~move_mask


@lru_cache(maxsize=None)
def legal_moves(open_mask: int, dice_sum: int) -> tuple[int, ...]:

    moves = []

    submask = open_mask

    while submask:

        if mask_sum(submask) == dice_sum:
            moves.append(submask)

        submask = (submask - 1) & open_mask

    return tuple(moves)

class Strategy(Protocol):

    name: str

    def dice_probabilities(
        self,
        open_mask: int
    ) -> dict[int, float]:
        ...

    def next_states(
        self,
        open_mask: int,
        dice_sum: int
    ) -> dict[int, float]:
        ...

@dataclass(frozen=True)
class PriorityStrategy:

    name: str
    num_dice: int 
    priorities: tuple[str, ...]

    def dice_probabilities(
        self,
        open_mask: int
    ) -> dict[int, float]:

        return DICE_OPTIONS[self.num_dice]

    def next_states(
        self,
        open_mask: int,
        dice_sum: int
    ) -> dict[int, float]:

        moves = legal_moves(
            open_mask,
            dice_sum
        )

        if not moves:
            return {}

        best_move = max(
            moves,
            key=lambda move:
                self._score(open_mask, move)
        )

        next_mask = close_tiles(
            open_mask,
            best_move
        )

        return {
            next_mask: 1.0
        }

    def _score(
        self,
        open_mask: int,
        move_mask: int
    ) -> tuple:

        score = []

        move_tiles = mask_to_tiles(move_mask)

        for priority in self.priorities:

            # möglichst viele Klappen
            if priority == "max_count":

                score.append(
                    len(move_tiles)
                )

            # möglichst hohe Klappen
            elif priority == "high_tiles":

                score.append(
                    tuple(
                        sorted(
                            move_tiles,
                            reverse=True
                        )
                    )
                )

            # möglichst niedrige Klappen
            elif priority == "low_tiles":

                score.append(
                    tuple(
                        -tile
                        for tile in sorted(move_tiles)
                    )
                )

            # möglichst viele zukünftige Spielmöglichkeiten
            elif priority == "leave_many_options":

                next_mask = close_tiles(
                    open_mask,
                    move_mask
                )

                options = 0


                probabilities = DICE_OPTIONS[
                    self.num_dice
                ]

                for dice_sum in probabilities:

                    options += len(
                        legal_moves(
                            next_mask,
                            dice_sum
                        )
                    )

                score.append(options)

            else:

                raise ValueError(
                    f"Unbekannte Priorität: {priority}"
                )

        return tuple(score)


@dataclass(frozen=True)
class RandomStrategy:

    num_dice: int

    @property
    def name(self):

        return (
            f"Zufallsstrategie "
            f"({self.num_dice} Würfel)"
        )

    def dice_probabilities(
        self,
        open_mask: int
    ) -> dict[int, float]:

        return DICE_OPTIONS[self.num_dice]

    def next_states(
        self,
        open_mask: int,
        dice_sum: int
    ) -> dict[int, float]:

        moves = legal_moves(
            open_mask,
            dice_sum
        )

        if not moves:
            return {}

        probability = 1.0 / len(moves)

        result = {}

        for move in moves:

            next_mask = close_tiles(
                open_mask,
                move
            )

            result[next_mask] = (
                result.get(next_mask, 0.0)
                + probability
            )

        return result

@dataclass(frozen=True)
class OptimalPenaltyStrategy:

    name: str = (
        "Bellman-optimal: minimale erwartete Restpunkte"
    )

    @staticmethod
    @lru_cache(maxsize=None)
    def value(open_mask: int) -> float:
       
        
        if open_mask == 0:
            return 0.0
            
        best_expected_value = float("inf")

        for num_dice in range(
            1,
            MAX_DICE + 1
        ):

            probabilities = DICE_OPTIONS[
                num_dice
            ]

            expected_value = 0.0

        
            for dice_sum, prob in (
                probabilities.items()
            ):

                moves = legal_moves(
                    open_mask,
                    dice_sum
                )


                if not moves:

                    expected_value += (
                        prob
                        * mask_sum(open_mask)
                    )

                else:

                    best_future_value = min(

                        OptimalPenaltyStrategy.value(
                            close_tiles(
                                open_mask,
                                move
                            )
                        )

                        for move in moves
                    )

                    expected_value += (
                        prob
                        * best_future_value
                    )

            best_expected_value = min(
                best_expected_value,
                expected_value
            )

        return best_expected_value

    @staticmethod
    @lru_cache(maxsize=None)
    def best_dice(open_mask: int) -> int:
 
        best_num_dice = None
        best_value = float("inf")

        for num_dice in range(1,MAX_DICE + 1):

            probabilities = DICE_OPTIONS[num_dice]
            expected_value = 0.0

            for dice_sum, prob in (
                probabilities.items()
            ):

                moves = legal_moves(
                    open_mask,
                    dice_sum
                )

                if not moves:

                    expected_value += (
                        prob
                        * mask_sum(open_mask)
                    )

                else:

                    best_future_value = min(

                        OptimalPenaltyStrategy.value(
                            close_tiles(
                                open_mask,
                                move
                            )
                        )

                        for move in moves
                    )

                    expected_value += (
                        prob
                        * best_future_value
                    )

            if expected_value < best_value:

                best_value = expected_value
                best_num_dice = num_dice

        return best_num_dice

    def dice_probabilities(
        self,
        open_mask: int
    ) -> dict[int, float]:

        num_dice = self.best_dice(
            open_mask
        )

        return DICE_OPTIONS[num_dice]

    @staticmethod
    def best_move(
        open_mask: int,
        dice_sum: int
    ) -> int | None:

        moves = legal_moves(
            open_mask,
            dice_sum
        )

        if not moves:
            return None

        return min(
            moves,
            key=lambda move:
                OptimalPenaltyStrategy.value(
                    close_tiles(
                        open_mask,
                        move
                    )
                )
        )

    def next_states(
        self,
        open_mask: int,
        dice_sum: int
    ) -> dict[int, float]:

        move = self.best_move(
            open_mask,
            dice_sum
        )

        if move is None:
            return {}

        next_mask = close_tiles(
            open_mask,
            move
        )

        return {
            next_mask: 1.0
        }


def expected_penalty(
    strategy: Strategy,
    start_mask: int = ALL_OPEN
) -> float:

    @lru_cache(maxsize=None)
    def value(open_mask: int) -> float:

        if open_mask == 0:
            return 0.0

        probabilities = (
            strategy.dice_probabilities(
                open_mask
            )
        )

        expected = 0.0

        for dice_sum, dice_prob in (
            probabilities.items()
        ):

            next_states = (
                strategy.next_states(
                    open_mask,
                    dice_sum
                )
            )
            if not next_states:

                expected += (
                    dice_prob
                    * mask_sum(open_mask)
                )

            else:

                future = sum(

                    transition_prob
                    * value(next_mask)

                    for (
                        next_mask,
                        transition_prob
                    ) in next_states.items()
                )

                expected += (
                    dice_prob
                    * future
                )

        return expected

    return value(start_mask)

def expected_rolls(
    strategy: Strategy,
    start_mask: int = ALL_OPEN
) -> float:

    @lru_cache(maxsize=None)
    def value(open_mask: int) -> float:

        if open_mask == 0:
            return 0.0

        probabilities = (
            strategy.dice_probabilities(
                open_mask
            )
        )

        expected = 1.0

        future_rolls = 0.0

        for dice_sum, dice_prob in (
            probabilities.items()
        ):

            next_states = (
                strategy.next_states(
                    open_mask,
                    dice_sum
                )
            )

            for (
                next_mask,
                transition_prob
            ) in next_states.items():

                future_rolls += (
                    dice_prob
                    * transition_prob
                    * value(next_mask)
                )

        return expected + future_rolls

    return value(start_mask)

def terminal_distribution(
    strategy: Strategy,
    start_mask: int = ALL_OPEN
) -> dict[int, float]:

    @lru_cache(maxsize=None)
    def distribution(
        open_mask: int
    ) -> tuple[tuple[int, float], ...]:

        if open_mask == 0:
            return (
                (0, 1.0),
            )

        result = {}

        probabilities = (
            strategy.dice_probabilities(
                open_mask
            )
        )

        for dice_sum, dice_prob in (
            probabilities.items()
        ):

            next_states = (
                strategy.next_states(
                    open_mask,
                    dice_sum
                )
            )

            if not next_states:

                result[open_mask] = (
                    result.get(
                        open_mask,
                        0.0
                    )
                    + dice_prob
                )

                continue


            for (
                next_mask,
                transition_prob
            ) in next_states.items():

                sub_distribution = dict(
                    distribution(next_mask)
                )

                for (
                    terminal_mask,
                    terminal_prob
                ) in sub_distribution.items():

                    result[terminal_mask] = (
                        result.get(
                            terminal_mask,
                            0.0
                        )
                        + dice_prob
                        * transition_prob
                        * terminal_prob
                    )

        return tuple(result.items())

    return dict(
        distribution(start_mask)
    )

def shut_probability(
    strategy: Strategy,
    start_mask: int = ALL_OPEN
) -> float:

    distribution = terminal_distribution(
        strategy,
        start_mask
    )

    return distribution.get(
        0,
        0.0
    )

def print_strategy_summary(
    strategy: Strategy,
    top_terminal_states: int = 8
):

    print("=" * 65)
    print(strategy.name)
    print("=" * 65)

    penalty = expected_penalty(
        strategy
    )

    win_prob = shut_probability(
        strategy
    )

    rolls = expected_rolls(
        strategy
    )

    print(
        f"Erwartete Restpunkte: "
        f"{penalty:.4f}"
    )

    print(
        f"Shut-the-Box-Wahrscheinlichkeit: "
        f"{100 * win_prob:.4f} %"
    )

    print(
        f"Erwartete Anzahl Würfe: "
        f"{rolls:.4f}"
    )

    distribution = terminal_distribution(
        strategy
    )

    sorted_states = sorted(
        distribution.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print()
    print("Häufigste Endzustände:")

    for (
        terminal_mask,
        probability
    ) in sorted_states[
        :top_terminal_states
    ]:

        tiles = mask_to_tiles(
            terminal_mask
        )

        score = mask_sum(
            terminal_mask
        )

        if terminal_mask == 0:

            label = "Shut the Box"

        else:

            label = str(tiles)

        print(
            f"{label:30} "
            f"Restpunkte = {score:3d}   "
            f"P = {100 * probability:7.3f} %"
        )

    print()

optimal_strategy = (
    OptimalPenaltyStrategy()
)

many_tiles_strategy = PriorityStrategy(
    name="Möglichst viele Klappen",
    num_dice=MAX_DICE,
    priorities=(
        "max_count",
        "high_tiles"
    )
)

high_tiles_strategy = PriorityStrategy(
    name="Möglichst hohe Klappen",
    num_dice=MAX_DICE,
    priorities=(
        "high_tiles",
        "max_count"
    )
)

many_options_strategy = PriorityStrategy(
    name="Viele Folgeoptionen",
    num_dice=MAX_DICE,
    priorities=(
        "leave_many_options",
        "high_tiles"
    )
)

random_strategy = RandomStrategy(num_dice=MAX_DICE)

print(f"Anzahl Klappen: {N_TILES}")

print(f"Maximal erlaubte Würfelanzahl: {MAX_DICE}")

print(f"Anzahl Klappenzustände: {NUM_STATES}")
print()

print_strategy_summary(optimal_strategy)
print_strategy_summary(high_tiles_strategy)
print_strategy_summary(many_tiles_strategy)
print_strategy_summary(many_options_strategy)
print_strategy_summary(random_strategy)




