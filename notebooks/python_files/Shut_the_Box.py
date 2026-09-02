from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


N_TILES = 9
ALL_OPEN = (1 << N_TILES) - 1

# Wahrscheinlichkeiten der Summen beim Wurf mit zwei fairen Wuerfeln.
DICE_SUM_PROBABILITIES: dict[int, float] = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}

# Zustandsraum erstellen: Die Klappen als (111111111) darstellen (binär)
def tile_to_bit(tile: int) -> int:
    return 1 << (tile - 1)

def mask_to_tiles(mask: int) -> tuple[int, ...]:
    return tuple(tile for tile in range(1, N_TILES + 1) if mask & tile_to_bit(tile))

def mask_sum(mask: int) -> int:
    return sum(mask_to_tiles(mask))

def close_tiles(open_mask: int, move_mask: int) -> int:
    return open_mask & ~move_mask


@lru_cache(maxsize=None)
def legal_moves(open_mask: int, dice_sum: int) -> tuple[int, ...]:
    #Alle Teilmengen der offenen Klappen, deren Summe zur Wuerfelsumme passt.
    moves: list[int] = []
    submask = open_mask
    while submask:
        if mask_sum(submask) == dice_sum:
            moves.append(submask)
        submask = (submask - 1) & open_mask
    return tuple(moves)


class Strategy(Protocol):
    name: str

    def next_states(self, open_mask: int, dice_sum: int) -> dict[int, float]:
            """Folgezustand -> bedingte Wahrscheinlichkeit nach bekanntem Wurf."""


@dataclass(frozen=True)
class PriorityStrategy:
    name: str
    priorities: tuple[str, ...]

    def next_states(self, open_mask: int, dice_sum: int) -> dict[int, float]:
        moves = legal_moves(open_mask, dice_sum)
        if not moves:
            return {}

        best_move = max(moves, key=lambda move: self._score(open_mask, move))
        return {close_tiles(open_mask, best_move): 1.0}

    def _score(self, open_mask: int, move_mask: int) -> tuple:
        score = []
        for priority in self.priorities:
            if priority == "max_count":
                score.append(len(mask_to_tiles(move_mask)))
            elif priority == "high_tiles":
                score.append(tuple(sorted(mask_to_tiles(move_mask), reverse=True)))
            elif priority == "low_tiles":
                score.append(tuple(-tile for tile in sorted(mask_to_tiles(move_mask))))
            elif priority == "leave_many_options":
                next_mask = close_tiles(open_mask, move_mask)
                options = sum(len(legal_moves(next_mask, s)) for s in DICE_SUM_PROBABILITIES)
                score.append(options)
            else:
                raise ValueError(f"Unbekannte Prioritaet: {priority}")
        return tuple(score)


@dataclass(frozen=True)
class RandomStrategy:
    name: str = "zufaellige Strategie"

    def next_states(self, open_mask: int, dice_sum: int) -> dict[int, float]:
        moves = legal_moves(open_mask, dice_sum)
        if not moves:
            return {}

        probability_per_move = 1 / len(moves)
        result: dict[int, float] = {}

        for move in moves:
            next_mask = close_tiles(open_mask, move)
            result[next_mask] = result.get(next_mask, 0.0) + probability_per_move

        return result

        
@dataclass(frozen=True)
class OptimalPenaltyStrategy:
    name: str = "Optimal: minimale erwartete Restpunkte"

    def next_states(self, open_mask: int, dice_sum: int) -> dict[int, float]:
        moves = legal_moves(open_mask, dice_sum)

        if not moves:
            return {}

        best_move = min(
            moves,
            key=lambda move: self.value(close_tiles(open_mask, move))
        )

        next_mask = close_tiles(open_mask, best_move)
        return {next_mask: 1.0}

    @staticmethod
    @lru_cache(maxsize=None)
    def value(open_mask: int) -> float:
        if open_mask == 0:
            return 0.0

        expected_value = 0.0

        for dice_sum, prob in DICE_SUM_PROBABILITIES.items():
            moves = legal_moves(open_mask, dice_sum)

            if not moves:
                expected_value += prob * mask_sum(open_mask)
            else:
                best_future_value = min(
                    OptimalPenaltyStrategy.value(
                        close_tiles(open_mask, move)
                    )
                    for move in moves
                )
                expected_value += prob * best_future_value

        return expected_value

def transition_matrix(strategy: Strategy) -> list[list[float]]:
    
   # Übergangsmatrix mit absorbierendem Verlustzustand.
    #Zustände 0 bis 511: normale Spielzustände
    #Zustand 512: Game Over
  
    n_game_states = 1 << N_TILES
    game_over = n_game_states
    n_states = n_game_states + 1

    matrix = [[0.0 for _ in range(n_states)] for _ in range(n_states)]

    for open_mask in range(n_game_states):

        if open_mask == 0:
            matrix[0][0] = 1.0
            continue

        for dice_sum, dice_probability in DICE_SUM_PROBABILITIES.items():

            next_states = strategy.next_states(
                open_mask,
                dice_sum
            )

            if not next_states:
                matrix[open_mask][game_over] += dice_probability
                continue

            for next_mask, strategy_probability in next_states.items():
                matrix[open_mask][next_mask] += (
                    dice_probability
                    * strategy_probability
                )

    matrix[game_over][game_over] = 1.0

    return matrix

def expected_penalty(strategy: Strategy, start_mask: int = ALL_OPEN) -> float:
    @lru_cache(maxsize=None)
    def value(open_mask: int) -> float:
        result = 0.0

        for dice_sum, dice_probability in DICE_SUM_PROBABILITIES.items():
            next_states = strategy.next_states(open_mask, dice_sum)

            if not next_states:
                result += dice_probability * mask_sum(open_mask)
                continue

            for next_mask, strategy_probability in next_states.items():
                result += dice_probability * strategy_probability * value(next_mask)

        return result

    return value(start_mask)

def expected_rolls(strategy: Strategy, start_mask: int = ALL_OPEN) -> float:
    @lru_cache(maxsize=None)
    def rolls(open_mask: int) -> float:
        result = 0.0

        for dice_sum, dice_probability in DICE_SUM_PROBABILITIES.items():
            next_states = strategy.next_states(open_mask, dice_sum)

            # Es wird trotzdem ein Wurf gemacht, auch wenn das Spiel danach endet
            if not next_states:
                result += dice_probability * 1
                continue

            for next_mask, strategy_probability in next_states.items():
                result += dice_probability * strategy_probability * (
                    1 + rolls(next_mask)
                )

        return result

    return rolls(start_mask)

def terminal_distribution(strategy: Strategy, start_mask: int = ALL_OPEN) -> dict[int, float]:
    @lru_cache(maxsize=None)
    def distribution(open_mask: int) -> tuple[tuple[int, float], ...]:
        result: dict[int, float] = {}

        for dice_sum, dice_probability in DICE_SUM_PROBABILITIES.items():
            next_states = strategy.next_states(open_mask, dice_sum)

            if not next_states:
                result[open_mask] = result.get(open_mask, 0.0) + dice_probability
                continue

            for next_mask, strategy_probability in next_states.items():
                for terminal_mask, terminal_probability in distribution(next_mask):
                    total_probability = dice_probability * strategy_probability * terminal_probability
                    result[terminal_mask] = result.get(terminal_mask, 0.0) + total_probability

        return tuple(sorted(result.items()))

    return dict(distribution(start_mask))


def print_strategy_summary(strategy: Strategy) -> None:
    distribution = terminal_distribution(strategy)
    shut_probability = distribution.get(0, 0.0)

    print(f"\nStrategie: {strategy.name}")
    print(f"Erwartete Minuspunkte: {expected_penalty(strategy):.4f}")
    print(f"Wahrscheinlichkeit fuer shut the box: {100 * shut_probability:.2f}%")
    print(f"Erwartete Anzahl Wuerfe: {expected_rolls(strategy):.4f}")

    print("Haeufigste Endzustaende:")
    for mask, probability in sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:5]:
        tiles = mask_to_tiles(mask)
        tiles_text = "keine" if not tiles else ", ".join(str(tile) for tile in tiles)
        print(f"  offen: {tiles_text:<18} Punkte: {mask_sum(mask):>2}  p={100 * probability:5.2f}%")


def main() -> None:
    strategies: list[Strategy] = [
        PriorityStrategy("moeglichst viele Klappen", ("max_count", "high_tiles")),
        PriorityStrategy("moeglichst hohe Klappen", ("high_tiles", "max_count")),
        PriorityStrategy("viele Folgeoptionen", ("leave_many_options", "high_tiles")),
        RandomStrategy(),
        OptimalPenaltyStrategy()
    ]

    print("Shut the Box Markov-Analyse")
    print(f"Startzustand: {ALL_OPEN} = offene Klappen {mask_to_tiles(ALL_OPEN)}")

    for strategy in strategies:
        print_strategy_summary(strategy)

    matrix = transition_matrix(RandomStrategy())
    print(f"\nMatrixcheck Zufallsstrategie, Startzeile: Summe = {sum(matrix[ALL_OPEN]):.4f}")


if __name__ == "__main__":
    main()





