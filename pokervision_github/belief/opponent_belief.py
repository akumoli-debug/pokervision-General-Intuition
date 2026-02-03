from __future__ import annotations

"""
Opponent belief representation for PokerVision.

This module keeps the implementation deliberately minimal and framework‑agnostic:
it provides a small dataclass for storing per‑opponent behavioural statistics,
plus helpers for serialisation and converting to a flat numeric feature vector.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class OpponentBelief:
    """
    Lightweight belief state over a single opponent.

    The goal is to capture a few high‑signal statistics that can be updated
    online and fed into the policy as features. All rates are stored in
    [0, 1] and backed by explicit counts to make confidence weighting easy.
    """

    # Aggregate interaction statistics
    hands_seen: int = 0

    # Behavioural rates in [0, 1]
    fold_to_pressure: float = 0.5  # frequency of folding when facing significant bets/raises
    call_down: float = 0.5        # tendency to call down with marginal hands
    aggression: float = 1.0       # aggression factor proxy (bets + raises vs calls)

    # Supporting counts for confidence / Bayesian updates
    fold_n: int = 0
    call_n: int = 0
    aggro_n: int = 0

    # Optional free‑form metadata (e.g., last_seen_ts, notes)
    meta: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------------- #
    # Serialisation helpers
    # --------------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the belief state to a plain dictionary suitable for JSON.
        """
        return {
            "hands_seen": self.hands_seen,
            "fold_to_pressure": self.fold_to_pressure,
            "call_down": self.call_down,
            "aggression": self.aggression,
            "fold_n": self.fold_n,
            "call_n": self.call_n,
            "aggro_n": self.aggro_n,
            "meta": self.meta,
        }

    # --------------------------------------------------------------------- #
    # Feature vector interface
    # --------------------------------------------------------------------- #

    def as_vector(self) -> List[float]:
        """
        Represent this belief as a flat numeric vector.

        This can be concatenated directly with environment features before
        feeding into a policy network. The exact ordering is intentionally
        simple and documented here so it can be mirrored on the training side:

        [0] hands_seen (log‑scaled proxy)
        [1] fold_to_pressure
        [2] call_down
        [3] aggression
        [4] fold_n (log‑scaled)
        [5] call_n (log‑scaled)
        [6] aggro_n (log‑scaled)
        """

        def _log_scale(count: int) -> float:
            # Simple, bounded log transform to keep counts in a reasonable range.
            if count <= 0:
                return 0.0
            # Using natural log keeps dependencies in stdlib (via math).
            import math

            return math.log(1.0 + float(count))

        return [
            _log_scale(self.hands_seen),
            float(self.fold_to_pressure),
            float(self.call_down),
            float(self.aggression),
            _log_scale(self.fold_n),
            _log_scale(self.call_n),
            _log_scale(self.aggro_n),
        ]

