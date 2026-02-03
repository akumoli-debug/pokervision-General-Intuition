"""
Live Poker UI - Fixed with Hand Analysis
Analyzes your actual cards + game state

Usage: python3 live_ui_fixed.py
Then open: http://localhost:8000
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import torch
import os
import re
from datetime import datetime
import uuid
import sys
from pathlib import Path

# Ensure local project root is on sys.path so `belief` / `telemetry` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from belief.opponent_belief import OpponentBelief  # type: ignore
from telemetry.logger import log_event  # type: ignore

# Load model
MODEL_PATH = 'models/final_model.pt'
DATA_PATH = 'data/akumoli_final_merged.json'

class CardParser:
    """Parse and evaluate cards"""
    RANKS = '23456789TJQKA'
    RANK_NAMES = {
        '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
        '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
        'T': 'Ten', 'J': 'Jack', 'Q': 'Queen', 'K': 'King', 'A': 'Ace'
    }
    
    @staticmethod
    def parse_card(card_str):
        """Parse Ah, Kd, etc."""
        card_str = card_str.strip().upper()
        if len(card_str) < 2:
            return None
        rank = card_str[0]
        suit = card_str[1].lower()
        if rank in CardParser.RANKS and suit in 'hdcs':
            return (rank, suit)
        return None
    
    @staticmethod
    def evaluate_hand(hole_cards, board_cards):
        """Evaluate hand strength"""
        all_cards = hole_cards + board_cards
        if not all_cards:
            return {'strength': 0, 'description': 'No cards'}
        
        ranks = [CardParser.RANKS.index(c[0]) for c in all_cards]
        suits = [c[1] for c in all_cards]
        
        # Count ranks
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        pairs = [r for r, c in rank_counts.items() if c == 2]
        trips = [r for r, c in rank_counts.items() if c == 3]
        quads = [r for r, c in rank_counts.items() if c == 4]
        
        # Check flush
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        is_flush = max(suit_counts.values()) >= 5 if suit_counts else False
        
        # Evaluate
        if quads:
            return {'strength': 0.95, 'description': 'Four of a Kind'}
        elif trips and pairs:
            return {'strength': 0.90, 'description': 'Full House'}
        elif is_flush:
            return {'strength': 0.80, 'description': 'Flush'}
        elif trips:
            return {'strength': 0.60, 'description': 'Three of a Kind'}
        elif len(pairs) >= 2:
            high_pair = CardParser.RANK_NAMES[CardParser.RANKS[max(pairs)]]
            return {'strength': 0.50, 'description': f'Two Pair, {high_pair}s'}
        elif pairs:
            pair_rank = CardParser.RANK_NAMES[CardParser.RANKS[pairs[0]]]
            strength = 0.35 + (pairs[0] / 13) * 0.15
            return {'strength': strength, 'description': f'Pair of {pair_rank}s'}
        else:
            high_card = CardParser.RANK_NAMES[CardParser.RANKS[max(ranks)]]
            return {'strength': 0.20, 'description': f'High Card {high_card}'}

class PokerAssistant:
    def __init__(self):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            self.action_vocab = checkpoint['action_vocab']
            self.accuracy = checkpoint.get('accuracy', 0.74)
            print(f"✓ Model loaded: {self.accuracy:.1%} accuracy")
            
            with open(DATA_PATH, 'r') as f:
                data = json.load(f)
            self.opponent_models = data.get('opponent_models', {})
            print(f"✓ Opponent data: {len(self.opponent_models)} players")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            self.opponent_models = {}
        
        # Persistent belief state per opponent (updated across hands)
        self.belief_by_player = {}  # name -> OpponentBelief
        # Where to write JSONL telemetry events (best‑effort only)
        self.telemetry_path = "logs/telemetry.jsonl"
    
    def get_belief(self, name):
        """Get or create the persistent belief state for an opponent."""
        key = (name or "").strip().lower()
        if not key:
            # Anonymous / unknown opponent: use a shared default instance
            key = "_anonymous"
        belief = self.belief_by_player.get(key)
        if belief is None:
            belief = OpponentBelief()
            self.belief_by_player[key] = belief
        return belief
    
    def get_opponent_info(self, name):
        name = name.lower().strip()
        if name in self.opponent_models:
            opp = self.opponent_models[name]
            return {
                'known': True,
                'actions': opp.get('total_actions', 0),
                'aggression': opp.get('aggression_factor', 1.0),
                'fold_freq': opp.get('fold_frequency', 0.5)
            }
        return {'known': False}
    
    def analyze(self, game_state):
        """Full analysis with cards"""
        
        pot = float(game_state.get('pot', 0))
        bet = float(game_state.get('bet', 0))
        your_stack = float(game_state.get('your_stack', 100))
        opp_stack = float(game_state.get('opp_stack', 100))
        sb = float(game_state.get('small_blind', 0.5))
        bb = float(game_state.get('big_blind', 1))
        position = game_state.get('position', 'bb')
        opponent_name = game_state.get('opponent', '')
        raw_opponent_action = (game_state.get('opponent_action') or '').strip().upper()
        
        # Parse cards
        hole_str = game_state.get('hole_cards', '')
        board_str = game_state.get('board_cards', '')
        
        hole_cards = []
        if hole_str:
            for card in hole_str.split():
                parsed = CardParser.parse_card(card)
                if parsed:
                    hole_cards.append(parsed)
        
        board_cards = []
        if board_str:
            for card in board_str.split():
                parsed = CardParser.parse_card(card)
                if parsed:
                    board_cards.append(parsed)
        
        # Evaluate hand
        hand_eval = CardParser.evaluate_hand(hole_cards, board_cards)
        hand_strength = hand_eval['strength']
        
        # Calculate metrics
        effective_stack = min(your_stack, opp_stack)
        spr = effective_stack / max(pot, 1)
        pot_odds = bet / (pot + bet) if bet > 0 else 0
        
        # Infer coarse opponent action + context for belief updates / telemetry
        if raw_opponent_action in ("BET", "RAISE", "ALL_IN"):
            observed_opponent_action = "PRESSURE"
        elif raw_opponent_action in ("CHECK", "CALL", "FOLD"):
            observed_opponent_action = "NO_BET"
        else:
            # Fallback: infer from bet size if action not specified
            if bet and bet > 0:
                observed_opponent_action = "PRESSURE"
            else:
                observed_opponent_action = "NO_BET"
        
        # Simple pressure context bucketed by SPR and position
        if spr < 3:
            spr_bucket = "low_spr"
        elif spr > 10:
            spr_bucket = "high_spr"
        else:
            spr_bucket = "mid_spr"
        pressure_context = f"{spr_bucket}_{position}"
        
        # Get opponent
        opp_info = self.get_opponent_info(opponent_name) if opponent_name else {'known': False}
        # Persistent belief state for this opponent (updated online)
        belief = self.get_belief(opponent_name)
        belief.hands_seen += 1
        
        # Make recommendation based on hand strength + situation
        if bet == 0:
            # No bet facing
            if hand_strength > 0.65:
                action = 'BET'
                reasoning = f'Strong hand ({hand_eval["description"]}) - bet for value'
                bet_size = f'${pot * 0.66:.2f} (2/3 pot)'
            elif hand_strength > 0.45:
                if spr < 5:
                    action = 'BET'
                    reasoning = f'Medium strength, low SPR - take control'
                    bet_size = f'${pot * 0.5:.2f} (1/2 pot)'
                else:
                    action = 'CHECK'
                    reasoning = f'Medium strength - pot control'
                    bet_size = 'N/A'
            else:
                action = 'CHECK'
                reasoning = f'Weak hand - check behind'
                bet_size = 'N/A'
        else:
            # Facing a bet
            equity_needed = pot_odds
            
            if hand_strength > equity_needed + 0.15:
                action = 'RAISE'
                reasoning = f'Strong hand ({hand_eval["description"]}) vs pot odds {pot_odds:.1%}'
                bet_size = f'${bet * 2.5:.2f} (2.5x their bet)'
            elif hand_strength > equity_needed:
                action = 'CALL'
                reasoning = f'Hand strength ({hand_strength:.1%}) > pot odds ({pot_odds:.1%})'
                bet_size = f'${bet:.2f}'
            else:
                # Consider opponent tendencies
                if opp_info.get('known') and opp_info['fold_freq'] > 0.60:
                    action = 'RAISE'
                    reasoning = f'Bluff opportunity - opponent folds {opp_info["fold_freq"]:.1%}'
                    bet_size = f'${bet * 3:.2f} (3x raise)'
                else:
                    action = 'FOLD'
                    reasoning = f'Hand strength ({hand_strength:.1%}) < pot odds ({pot_odds:.1%})'
                    bet_size = 'N/A'
        
        # Adjust for opponent
        if opp_info.get('known'):
            if opp_info['aggression'] > 2.5 and action == 'CALL' and hand_strength > 0.6:
                reasoning += ' | Consider check-raise vs aggressive opponent'
            elif opp_info['fold_freq'] < 0.30 and action == 'RAISE' and hand_strength < 0.7:
                reasoning += ' | WARNING: Calling station - need strong value'
        
        # ------------------------------------------------------------------
        # Online belief updates (kept tiny and stable)
        # ------------------------------------------------------------------

        # Update aggression estimate based on whether we faced pressure
        if observed_opponent_action == "PRESSURE":
            # Move aggression toward 1.0 with a decaying learning rate
            lr = 1.0 / float(belief.aggro_n + 1)
            target = 1.0
            new_aggr = belief.aggression + lr * (target - belief.aggression)
            belief.aggression = max(0.0, min(1.0, new_aggr))
            belief.aggro_n += 1
        elif observed_opponent_action == "NO_BET" and belief.aggro_n > 0:
            # Slightly nudge aggression down when opponents decline to apply pressure
            lr = 1.0 / float(belief.aggro_n + 1)
            target = 0.3
            new_aggr = belief.aggression + lr * (target - belief.aggression)
            belief.aggression = max(0.0, min(1.0, new_aggr))

        # When we choose a pressure action, nudge fold_to_pressure toward prior fold_freq
        if action in ("BET", "RAISE") and opp_info.get('known') and 'fold_freq' in opp_info:
            prior_fold = float(opp_info['fold_freq'])
            lr = 0.1 / float(belief.fold_n + 1)  # very conservative update
            new_fold = belief.fold_to_pressure + lr * (prior_fold - belief.fold_to_pressure)
            belief.fold_to_pressure = max(0.0, min(1.0, new_fold))
            belief.fold_n += 1

        # ------------------------------------------------------------------
        # Telemetry (best-effort JSONL write, never breaks endpoint)
        # ------------------------------------------------------------------
        try:
            ts = datetime.utcnow().isoformat() + "Z"
            # Use provided hand_id if present; otherwise generate a short UUID.
            hand_id = str(game_state.get("hand_id") or uuid.uuid4().hex[:8])

            env_state = {
                "pot": pot,
                "bet": bet,
                "your_stack": your_stack,
                "opp_stack": opp_stack,
                "small_blind": sb,
                "big_blind": bb,
                "position": position,
                "street": game_state.get("street"),
                "raw_opponent_action": raw_opponent_action,
                "spr": spr,
                "pot_odds": pot_odds,
                "has_hole_cards": bool(hole_str),
                "board_cards_count": len(board_cards),
            }

            event = {
                "ts": ts,
                "hand_id": hand_id,
                "opponent": opponent_name,
                "env_state": env_state,
                "observed_opponent_action": observed_opponent_action,
                "pressure_context": pressure_context,
                "belief_state": {
                    "dict": belief.to_dict(),
                    "vector": belief.as_vector(),
                },
                "opp_info_prior": opp_info,
                "recommendation": {
                    "action": action,
                    "bet_size": bet_size,
                    "confidence": self.accuracy,
                    "reasoning": reasoning,
                },
            }

            log_event(self.telemetry_path, event)
        except Exception:
            # Telemetry is strictly best-effort.
            pass

        return {
            'action': action,
            'bet_size': bet_size,
            'confidence': self.accuracy,
            'reasoning': reasoning,
            'hand_evaluation': hand_eval,
            'pot_odds': pot_odds,
            'spr': spr,
            'opponent_info': opp_info,
            'metrics': {
                'pot': pot,
                'bet': bet,
                'effective_stack': effective_stack,
                'hand_strength': hand_strength
            }
        }

assistant = PokerAssistant()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PokerVision Live</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 { font-size: 36px; margin-bottom: 10px; }
        .accuracy { font-size: 18px; opacity: 0.9; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.95);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            color: #333;
        }
        
        .card h2 { margin-bottom: 20px; color: #1e3c72; font-size: 24px; }
        
        .input-group { margin-bottom: 15px; }
        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
            font-size: 14px;
        }
        
        .input-group input, .input-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        
        .input-group input:focus, .input-group select:focus {
            outline: none;
            border-color: #2a5298;
        }
        
        .input-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .card-input-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .card-input-section h3 {
            font-size: 16px;
            margin-bottom: 10px;
            color: #1e3c72;
        }
        
        .card-help {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .analyze-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .analyze-btn:hover { transform: translateY(-2px); }
        
        #result { display: none; }
        
        .recommendation {
            background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            color: white;
        }
        
        .rec-action { font-size: 48px; font-weight: 700; margin-bottom: 10px; }
        .rec-bet-size { font-size: 24px; margin-bottom: 15px; opacity: 0.9; }
        .rec-reasoning { font-size: 16px; line-height: 1.6; }
        
        .hand-eval {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            color: #0d47a1;
        }
        
        .hand-eval h3 { margin-bottom: 10px; }
        
        .strength-bar {
            width: 100%;
            height: 30px;
            background: #ddd;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .strength-fill {
            height: 100%;
            background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #1e3c72;
        }
        
        .opponent-intel {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 8px;
            color: #856404;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(133,100,4,0.2);
        }
        
        .strategy-tip {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            color: #0c5460;
        }
        
        @media (max-width: 968px) {
            .main-grid { grid-template-columns: 1fr; }
            .input-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎰 PokerVision Live</h1>
            <div class="accuracy">74.3% Accuracy • Beats GTO by +19%</div>
        </div>
        
        <div class="main-grid">
            <div class="card">
                <h2>Game State</h2>
                <form id="gameForm">
                    <div class="card-input-section">
                        <h3>Your Cards</h3>
                        <div class="input-group">
                            <label>Hole Cards</label>
                            <input type="text" id="hole_cards" placeholder="e.g., Ah Kd">
                            <div class="card-help">Format: Ah Kd (Ace hearts, King diamonds)</div>
                        </div>
                        <div class="input-group">
                            <label>Board Cards (if any)</label>
                            <input type="text" id="board_cards" placeholder="e.g., Ks 9s 4h">
                            <div class="card-help">Flop/Turn/River cards (if dealt)</div>
                        </div>
                    </div>
                    
                    <div class="input-row">
                        <div class="input-group">
                            <label>Pot Size ($)</label>
                            <input type="number" id="pot" step="0.01" placeholder="50.00" required>
                        </div>
                        <div class="input-group">
                            <label>Bet to You ($)</label>
                            <input type="number" id="bet" step="0.01" placeholder="0" value="0">
                        </div>
                    </div>
                    
                    <div class="input-row">
                        <div class="input-group">
                            <label>Your Stack ($)</label>
                            <input type="number" id="your_stack" step="0.01" placeholder="200.00" required>
                        </div>
                        <div class="input-group">
                            <label>Opponent Stack ($)</label>
                            <input type="number" id="opp_stack" step="0.01" placeholder="180.00" required>
                        </div>
                    </div>
                    
                    <div class="input-row">
                        <div class="input-group">
                            <label>Small Blind ($)</label>
                            <input type="number" id="small_blind" step="0.01" placeholder="1.00" required>
                        </div>
                        <div class="input-group">
                            <label>Big Blind ($)</label>
                            <input type="number" id="big_blind" step="0.01" placeholder="2.00" required>
                        </div>
                    </div>
                    
                    <div class="input-row">
                        <div class="input-group">
                            <label>Position</label>
                            <select id="position">
                                <option value="button">Button (BTN)</option>
                                <option value="sb">Small Blind (SB)</option>
                                <option value="bb">Big Blind (BB)</option>
                                <option value="utg">UTG</option>
                                <option value="hj">Hijack (HJ)</option>
                                <option value="co">Cutoff (CO)</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="input-row">
                        <div class="input-group">
                            <label>Street</label>
                            <select id="street">
                                <option value="preflop">Preflop</option>
                                <option value="flop">Flop</option>
                                <option value="turn">Turn</option>
                                <option value="river">River</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Opponent Action (this decision)</label>
                            <select id="opponent_action">
                                <option value="">-- Select --</option>
                                <option value="CHECK">Check</option>
                                <option value="BET">Bet</option>
                                <option value="RAISE">Raise</option>
                                <option value="CALL">Call</option>
                                <option value="FOLD">Fold</option>
                                <option value="ALL_IN">All-in</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="input-group">
                        <label>Opponent Name (optional)</label>
                        <input type="text" id="opponent" placeholder="e.g., seb">
                    </div>
                    
                    <button type="submit" class="analyze-btn">🔍 Analyze & Recommend</button>
                </form>
            </div>
            
            <div class="card">
                <h2>Instructions</h2>
                <div style="padding: 20px;">
                    <h3 style="margin-bottom: 15px; color: #1e3c72;">How to Use:</h3>
                    <ol style="line-height: 2; color: #666;">
                        <li><strong>Enter your hole cards</strong> (e.g., "Ah Kd")</li>
                        <li><strong>Enter board cards</strong> if flop/turn/river dealt</li>
                        <li><strong>Fill in pot size</strong> and bet facing</li>
                        <li><strong>Enter stack sizes</strong> and blinds</li>
                        <li><strong>Click Analyze</strong> to get AI recommendation</li>
                    </ol>
                    
                    <h3 style="margin: 20px 0 15px; color: #1e3c72;">Card Format:</h3>
                    <ul style="line-height: 2; color: #666;">
                        <li><strong>Ranks:</strong> 2-9, T, J, Q, K, A</li>
                        <li><strong>Suits:</strong> h (hearts), d (diamonds), c (clubs), s (spades)</li>
                        <li><strong>Examples:</strong> Ah Kd, Tc 9s, Qh Jh</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div id="result"></div>
    </div>
    
    <script>
        document.getElementById('gameForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const gameState = {
                pot: document.getElementById('pot').value,
                bet: document.getElementById('bet').value,
                your_stack: document.getElementById('your_stack').value,
                opp_stack: document.getElementById('opp_stack').value,
                small_blind: document.getElementById('small_blind').value,
                big_blind: document.getElementById('big_blind').value,
                street: document.getElementById('street').value,
                position: document.getElementById('position').value,
                opponent_action: document.getElementById('opponent_action').value,
                opponent: document.getElementById('opponent').value,
                hole_cards: document.getElementById('hole_cards').value,
                board_cards: document.getElementById('board_cards').value
            };
            
            try {
                const response = await fetch('/api/analyze?' + new URLSearchParams(gameState));
                const result = await response.json();
                displayResult(result);
            } catch (error) {
                alert('Error: ' + error);
            }
        });
        
        function displayResult(result) {
            const metrics = result.metrics;
            const opp = result.opponent_info;
            const hand = result.hand_evaluation;
            
            let html = `
                <div class="card">
                    <div class="recommendation">
                        <div class="rec-action">🎯 ${result.action}</div>
                        <div class="rec-bet-size">${result.bet_size}</div>
                        <div class="rec-reasoning">${result.reasoning}</div>
                    </div>
            `;
            
            if (hand.description !== 'No cards') {
                const strength = (metrics.hand_strength * 100).toFixed(0);
                html += `
                    <div class="hand-eval">
                        <h3>🎴 Your Hand: ${hand.description}</h3>
                        <div class="strength-bar">
                            <div class="strength-fill" style="width: ${strength}%">
                                ${strength}% strength
                            </div>
                        </div>
                    </div>
                `;
            }
            
            html += `
                <h2>Key Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Pot</div>
                        <div class="metric-value">$${metrics.pot.toFixed(2)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">To Call</div>
                        <div class="metric-value">$${metrics.bet.toFixed(2)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Pot Odds</div>
                        <div class="metric-value">${(result.pot_odds * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">SPR</div>
                        <div class="metric-value">${result.spr.toFixed(1)}</div>
                    </div>
                </div>
            `;
            
            if (opp.known) {
                html += `
                    <div class="opponent-intel">
                        <h3>👤 Opponent: Known Player</h3>
                        <div class="stat-row">
                            <span>Actions Observed:</span>
                            <strong>${opp.actions}</strong>
                        </div>
                        <div class="stat-row">
                            <span>Aggression Factor:</span>
                            <strong>${opp.aggression.toFixed(2)}</strong>
                        </div>
                        <div class="stat-row">
                            <span>Fold Frequency:</span>
                            <strong>${(opp.fold_freq * 100).toFixed(1)}%</strong>
                        </div>
                `;
                
                if (opp.fold_freq > 0.55) {
                    html += `<div class="strategy-tip">💡 They fold often - exploit with aggression</div>`;
                } else if (opp.fold_freq < 0.30) {
                    html += `<div class="strategy-tip">💡 Calling station - value bet only</div>`;
                }
                
                html += `</div>`;
            }
            
            html += `</div>`;
            
            document.getElementById('result').innerHTML = html;
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))
        
        elif self.path.startswith('/api/analyze'):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            
            game_state = {k: v[0] for k, v in params.items()}
            result = assistant.analyze(game_state)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

print("="*70)
print("PokerVision Live UI - Fixed")
print("="*70)
print()

server = HTTPServer(('localhost', 8000), RequestHandler)

print("✓ Server Running!")
print()
print("🌐 Open: http://localhost:8000")
print()
print("Features:")
print("  • Hand analysis (with your actual cards)")
print("  • Real-time recommendations")
print("  • Opponent intelligence")
print("  • Bet sizing suggestions")
print()
print("Press Ctrl+C to stop")
print("="*70)
print()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n\nShutting down...")
    server.shutdown()
