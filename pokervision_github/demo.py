"""
Tiny, fast demo for PokerVision.

This does NOT require any data files or pre-trained weights.
It just runs a few fixed example hands through a very simple
heuristic "model" and prints the recommended actions plus
explanations, to give reviewers an immediate feel for the project.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DemoHand:
    opponent: str
    pot: float
    bet: float
    hero_stack: float
    spr: float
    position: str
    tendency: str  # "overfolds", "calls_too_much", "aggressive"


def recommend_action(hand: DemoHand) -> str:
    """
    Heuristic policy chosen to mirror the kinds of patterns
    the real model learns (SPR, opponent tendencies).
    """
    # Very low SPR: commit or fold
    if hand.spr < 3:
        if hand.tendency == "overfolds":
            return "JAM: low SPR + over-folder → maximum pressure."
        else:
            return "CALL: low SPR, pot already big → continue."

    # High SPR vs over-folder → bluff more
    if hand.tendency == "overfolds":
        return "RAISE: opponent over-folds; increase bluffing frequency."

    # Calling station → value bet
    if hand.tendency == "calls_too_much":
        return "BET SMALL FOR VALUE: calling station; avoid big bluffs."

    # Very aggressive → trap
    if hand.tendency == "aggressive":
        return "CHECK-CALL / CHECK-RAISE: let aggression work for you."

    return "CHECK: unclear edge; keep pot small."


def main() -> None:
    print("=" * 70)
    print("PokerVision demo – heuristic snapshot (no model file needed)")
    print("=" * 70)
    print()

    hands: List[DemoHand] = [
        DemoHand(
            opponent="seb",
            pot=50,
            bet=30,
            hero_stack=200,
            spr=4.0,
            position="button",
            tendency="overfolds",
        ),
        DemoHand(
            opponent="punter_sausage",
            pot=40,
            bet=18,
            hero_stack=160,
            spr=8.0,
            position="bb",
            tendency="calls_too_much",
        ),
        DemoHand(
            opponent="cursed_pete",
            pot=30,
            bet=25,
            hero_stack=90,
            spr=2.0,
            position="sb",
            tendency="aggressive",
        ),
    ]

    for i, hand in enumerate(hands, start=1):
        print(f"Hand {i}: opponent={hand.opponent}, pot=${hand.pot}, bet=${hand.bet}, SPR={hand.spr}")
        action = recommend_action(hand)
        print(f"  → Recommendation: {action}")
        print()

    print("Done. For full training and real models, see docs/TRAINING.md.")


if __name__ == "__main__":
    main()

