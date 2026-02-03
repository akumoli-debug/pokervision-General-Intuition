from __future__ import annotations

"""
Opponent belief representation for PokerVision.

This module keeps the implementation deliberately minimal and framework‑agnostic:
it provides a small dataclass for storing per‑opponent behavioural statistics,
plus helpers for serialisation and converting to a flat numeric feature vector.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict


@dataclass
class StreetAction:
    """Action taken on a specific street."""
    street: str  # 'preflop', 'flop', 'turn', 'river'
    action: str  # 'BET', 'CALL', 'RAISE', 'CHECK', 'FOLD'
    bet_size: float = 0.0
    pot_size: float = 0.0
    position: Optional[str] = None


@dataclass
class HandHistory:
    """Complete history of actions in a hand across all streets."""
    hand_id: str = ""
    streets: Dict[str, List[StreetAction]] = field(default_factory=lambda: defaultdict(list))
    
    def add_action(self, street: str, action: str, bet_size: float = 0.0, 
                   pot_size: float = 0.0, position: Optional[str] = None):
        """Add an action to the hand history."""
        self.streets[street].append(StreetAction(
            street=street,
            action=action,
            bet_size=bet_size,
            pot_size=pot_size,
            position=position
        ))
    
    def get_street_actions(self, street: str) -> List[StreetAction]:
        """Get all actions for a specific street."""
        return self.streets.get(street, [])
    
    def get_action_pattern(self) -> Dict[str, Any]:
        """Extract action pattern for prediction."""
        pattern = {
            'preflop_aggressive': False,
            'flop_continuation': False,
            'turn_continuation': False,
            'river_value_bet': False,
            'bluff_frequency': 0.0,
        }
        
        # Check preflop aggression
        preflop_actions = self.get_street_actions('preflop')
        if any(a.action in ('BET', 'RAISE') for a in preflop_actions):
            pattern['preflop_aggressive'] = True
        
        # Check continuation betting
        if preflop_actions and self.get_street_actions('flop'):
            if any(a.action in ('BET', 'RAISE') for a in preflop_actions):
                flop_actions = self.get_street_actions('flop')
                if any(a.action in ('BET', 'RAISE') for a in flop_actions):
                    pattern['flop_continuation'] = True
        
        # Check turn continuation
        if self.get_street_actions('flop') and self.get_street_actions('turn'):
            flop_actions = self.get_street_actions('flop')
            if any(a.action in ('BET', 'RAISE') for a in flop_actions):
                turn_actions = self.get_street_actions('turn')
                if any(a.action in ('BET', 'RAISE') for a in turn_actions):
                    pattern['turn_continuation'] = True
        
        # Check river value betting
        river_actions = self.get_street_actions('river')
        if any(a.action in ('BET', 'RAISE') for a in river_actions):
            pattern['river_value_bet'] = True
        
        return pattern


@dataclass
class OpponentBelief:
    """
    Lightweight belief state over a single opponent.

    The goal is to capture a few high‑signal statistics that can be updated
    online and fed into the policy as features. All rates are stored in
    [0, 1] and backed by explicit counts to make confidence weighting easy.
    
    Now includes action history tracking for street-by-street prediction.
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

    # Action history tracking
    hand_histories: List[HandHistory] = field(default_factory=list)
    max_history: int = 50  # Keep last N hands
    
    # Street-specific patterns
    preflop_raise_freq: float = 0.3  # Frequency of preflop raises
    continuation_bet_freq: float = 0.6  # Frequency of c-betting after raising preflop
    turn_barrel_freq: float = 0.4  # Frequency of betting turn after c-betting flop
    river_bet_freq: float = 0.3  # Frequency of river bets
    
    # Pattern counts
    preflop_raise_n: int = 0
    continuation_bet_n: int = 0
    turn_barrel_n: int = 0
    river_bet_n: int = 0

    # Optional free‑form metadata (e.g., last_seen_ts, notes)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def add_hand_history(self, history: HandHistory):
        """Add a hand history and update patterns."""
        self.hand_histories.append(history)
        if len(self.hand_histories) > self.max_history:
            self.hand_histories.pop(0)
        
        # Update pattern frequencies
        pattern = history.get_action_pattern()
        
        if pattern['preflop_aggressive']:
            self.preflop_raise_n += 1
            lr = 1.0 / (self.preflop_raise_n + 1)
            self.preflop_raise_freq = self.preflop_raise_freq + lr * (1.0 - self.preflop_raise_freq)
        
        if pattern['flop_continuation']:
            self.continuation_bet_n += 1
            lr = 1.0 / (self.continuation_bet_n + 1)
            self.continuation_bet_freq = self.continuation_bet_freq + lr * (1.0 - self.continuation_bet_freq)
        
        if pattern['turn_continuation']:
            self.turn_barrel_n += 1
            lr = 1.0 / (self.turn_barrel_n + 1)
            self.turn_barrel_freq = self.turn_barrel_freq + lr * (1.0 - self.turn_barrel_freq)
        
        if pattern['river_value_bet']:
            self.river_bet_n += 1
            lr = 1.0 / (self.river_bet_n + 1)
            self.river_bet_freq = self.river_bet_freq + lr * (1.0 - self.river_bet_freq)
    
    def predict_action(self, current_street: str, previous_actions: List[StreetAction]) -> Dict[str, float]:
        """
        Predict opponent's likely action based on history and current context.
        Returns probabilities for each action type.
        """
        probs = {
            'BET': 0.0,
            'RAISE': 0.0,
            'CALL': 0.0,
            'CHECK': 0.0,
            'FOLD': 0.0,
        }
        
        # Base probabilities from aggregate stats
        probs['BET'] = self.aggression * 0.3
        probs['RAISE'] = self.aggression * 0.2
        probs['CALL'] = (1.0 - self.fold_to_pressure) * 0.4
        probs['CHECK'] = (1.0 - self.aggression) * 0.3
        probs['FOLD'] = self.fold_to_pressure * 0.3
        
        # Adjust based on street-specific patterns
        if current_street == 'preflop':
            if self.preflop_raise_freq > 0.4:
                probs['RAISE'] += 0.2
                probs['BET'] -= 0.1
        
        elif current_street == 'flop':
            # Check if they raised preflop (continuation bet pattern)
            if previous_actions and any(a.action in ('BET', 'RAISE') for a in previous_actions if a.street == 'preflop'):
                if self.continuation_bet_freq > 0.5:
                    probs['BET'] += 0.3
                    probs['CHECK'] -= 0.2
        
        elif current_street == 'turn':
            # Check if they c-bet flop (barrel pattern)
            if previous_actions and any(a.action in ('BET', 'RAISE') for a in previous_actions if a.street == 'flop'):
                if self.turn_barrel_freq > 0.4:
                    probs['BET'] += 0.25
                    probs['CHECK'] -= 0.2
        
        elif current_street == 'river':
            # River betting frequency
            if self.river_bet_freq > 0.3:
                probs['BET'] += 0.2
                probs['CHECK'] -= 0.15
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            for action in probs:
                probs[action] /= total
        
        return probs

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
            "preflop_raise_freq": self.preflop_raise_freq,
            "continuation_bet_freq": self.continuation_bet_freq,
            "turn_barrel_freq": self.turn_barrel_freq,
            "river_bet_freq": self.river_bet_freq,
            "preflop_raise_n": self.preflop_raise_n,
            "continuation_bet_n": self.continuation_bet_n,
            "turn_barrel_n": self.turn_barrel_n,
            "river_bet_n": self.river_bet_n,
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
        [7] preflop_raise_freq
        [8] continuation_bet_freq
        [9] turn_barrel_freq
        [10] river_bet_freq
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
            float(self.preflop_raise_freq),
            float(self.continuation_bet_freq),
            float(self.turn_barrel_freq),
            float(self.river_bet_freq),
        ]

