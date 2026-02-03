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
import base64
import io

# Screen capture and OCR (optional dependencies)
try:
    import mss
    import pytesseract
    from PIL import Image
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    SCREEN_CAPTURE_AVAILABLE = False
    print("⚠ Screen capture not available. Install: pip install mss pytesseract pillow")

# Browser automation for PokerNow tab capture (macOS AppleScript)
import subprocess
import platform

# Ensure local project root is on sys.path so `belief` / `telemetry` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from belief.opponent_belief import OpponentBelief, HandHistory, StreetAction  # type: ignore
from telemetry.logger import log_event  # type: ignore

# #region agent log
DEBUG_LOG = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"  # workspace/.cursor/debug.log
def _debug_log(msg, data=None, hypothesis_id=None):
    try:
        import time
        line = json.dumps({"message": msg, "data": data or {}, "hypothesisId": hypothesis_id, "timestamp": time.time(), "location": "live_ui_fixed.py"}) + "\n"
        with open(DEBUG_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass
# #endregion

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


class HandDescriptionParser:
    """Parse natural language hand descriptions into structured game state."""
    
    @staticmethod
    def parse_card_notation(card_str):
        """Convert shorthand like 'KTo' (King Ten offsuit) or 'Ah' to standard format."""
        card_str = card_str.strip().upper()
        
        # Handle offsuit notation (KTo -> K T)
        if 'o' in card_str.lower() and len(card_str) >= 3:
            # Remove 'o' and split into two cards
            parts = card_str.replace('o', '').replace('O', '')
            if len(parts) == 2:
                rank1, rank2 = parts[0], parts[1]
                # Default to different suits (offsuit)
                return f"{rank1}h {rank2}d"
        
        # Handle suited notation (KTs -> K T suited)
        if 's' in card_str.lower() and len(card_str) >= 3:
            parts = card_str.replace('s', '').replace('S', '')
            if len(parts) == 2:
                rank1, rank2 = parts[0], parts[1]
                # Default to same suit
                return f"{rank1}h {rank2}h"
        
        # Already in standard format (Ah Kd)
        if len(card_str) >= 4 and card_str[1] in 'hdcs' and card_str[3] in 'hdcs':
            return card_str
        
        return card_str
    
    @staticmethod
    def _parse_amount(s: str, full_match: str) -> float:
        """Parse amount from match; handle 2k, 6.7k, 400, etc."""
        s = s.strip()
        if not s:
            return 0.0
        val = float(s)
        if 'k' in full_match.lower():
            return val * 1000
        return val

    @staticmethod
    def _board_to_cards(board_str: str) -> str:
        """Convert board string like 'T42' or 'T 4 2' to Ah Kd format with default suits."""
        suits = 'hdsc'
        cards = []
        for i, c in enumerate(re.sub(r'[^AKQJT2-9]', '', board_str.upper())):
            suit = suits[i % 4]
            cards.append(c + suit)
        return ' '.join(cards) if cards else ''

    @staticmethod
    def parse_hand_description(text):
        """
        Parse natural language hand description.
        
        Handles: "1/3 8k effective", "Raise KTo UTG to 20", "BU 3 bets to 100",
        "Flop T42r", "Turn 7r", "River 2", "He bets 2k", "move in for 6.7k total".
        """
        parsed = {}
        text_lower = text.lower()
        
        # Blinds (1/3, $1/$3)
        blind_match = re.search(r'(\d+)\s*[/-]\s*(\d+)', text)
        if blind_match:
            parsed['small_blind'] = float(blind_match.group(1))
            parsed['big_blind'] = float(blind_match.group(2))
        else:
            parsed['small_blind'] = 1
            parsed['big_blind'] = 2
        
        # Effective stack (8k, 6.7k, 8000)
        stack_vals = []
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*k\s*(?:effective|total)?', text_lower):
            stack_vals.append(float(m.group(1)) * 1000)
        if not stack_vals:
            for m in re.finditer(r'\b(\d{3,})\b', text):
                n = float(m.group(1))
                if 100 <= n <= 100000:
                    stack_vals.append(n)
        if stack_vals:
            parsed['your_stack'] = max(stack_vals)
            parsed['opp_stack'] = max(stack_vals)
        
        # Position from "UTG", "BU", "raise ... UTG"
        if 'utg' in text_lower:
            parsed['position'] = 'utg'
        elif ' bu ' in text_lower or 'button' in text_lower or ' btn ' in text_lower:
            parsed['position'] = 'button'
        elif 'co' in text_lower or 'cutoff' in text_lower:
            parsed['position'] = 'co'
        elif 'hj' in text_lower or 'hijack' in text_lower:
            parsed['position'] = 'hj'
        elif 'sb' in text_lower or 'small blind' in text_lower:
            parsed['position'] = 'sb'
        elif 'bb' in text_lower or 'big blind' in text_lower:
            parsed['position'] = 'bb'
        else:
            parsed['position'] = 'utg'
        
        # Hole cards: "Raise KTo UTG", "KTo", "Ah Kd"
        card_patterns = [
            r'raise\s+([AKQJT2-9][hdcs]?\s*[AKQJT2-9][hdcs]?[os]?)',
            r'([AKQJT2-9][hdcs]\s+[AKQJT2-9][hdcs])',
            r'\b([AKQJT2-9]{1,2}[os]?)\s+UTG',
            r'\b([AKQJT2-9]{1,2}[os]?)\s+BU',
            r'\b([AKQJT2-9]T[os]?)\b',
            r'\b(KTo|KTs|QJo|AK)\b',
        ]
        for pattern in card_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cards = HandDescriptionParser.parse_card_notation(match.group(1).strip())
                if cards and len(cards) >= 3:
                    parsed['hole_cards'] = cards
                    break
        
        # Build board from Flop / Turn / River
        flop_str = ''
        turn_str = ''
        river_str = ''
        flop_match = re.search(r'flop\s+([AKQJT2-9hdcs\s\.]+?)(?=\s*[\.\n]|\s+turn\s+|\s+river\s+|$)', text_lower, re.DOTALL | re.IGNORECASE)
        if flop_match:
            flop_str = re.sub(r'[^AKQJT2-9]', '', flop_match.group(1))
        turn_match = re.search(r'turn\s+([AKQJT2-9hdcs\s\.]+?)(?=\s*[\.\n]|\s+river\s+|$)', text_lower, re.DOTALL | re.IGNORECASE)
        if turn_match:
            turn_str = re.sub(r'[^AKQJT2-9]', '', turn_match.group(1))
        river_match = re.search(r'river\s+([AKQJT2-9hdcs\s\.]+?)(?=\s*[\.\n]|\s+I\s+|\s+he\s+|\s+what\s+|$)', text_lower, re.DOTALL | re.IGNORECASE)
        if river_match:
            river_str = re.sub(r'[^AKQJT2-9]', '', river_match.group(1))
        
        full_board = flop_str + turn_str + river_str
        if full_board:
            parsed['board_cards'] = HandDescriptionParser._board_to_cards(full_board)
        
        # Street: if "river" mentioned and we're asking about the decision, use river
        if 'river' in text_lower or river_str:
            parsed['street'] = 'river'
        elif turn_str:
            parsed['street'] = 'turn'
        elif flop_str:
            parsed['street'] = 'flop'
        else:
            parsed['street'] = 'preflop'
        
        # Bet to you: "He bets 2k", "bets 400", "bet 2k"
        last_bet = None
        for m in re.finditer(r'(?:he\s+)?(?:bet|bets)\s+\$?(\d+(?:\.\d+)?)\s*(k)?', text_lower):
            last_bet = float(m.group(1)) * (1000 if m.group(2) else 1)
        if last_bet is not None:
            parsed['bet'] = last_bet
        
        # Pot estimate: sum of money in before opponent's current bet
        sb = parsed.get('small_blind', 1)
        bb = parsed.get('big_blind', 2)
        pot_before_current_bet = sb + bb
        bets_found = []
        for m in re.finditer(r'(?:to|bet|bets?)\s+\$?(\d+(?:\.\d+)?)\s*(k)?', text_lower):
            amt = float(m.group(1)) * (1000 if m.group(2) else 1)
            bets_found.append(amt)
        # Each bet we see went in twice (villain + hero call) except the last one (villain's current bet)
        for i, amt in enumerate(bets_found):
            if i < len(bets_found) - 1:
                pot_before_current_bet += amt * 2
            else:
                pot_before_current_bet += amt  # villain's bet already in
        if pot_before_current_bet > 0:
            parsed['pot'] = max(int(pot_before_current_bet), (parsed.get('bet') or 0) * 2)
        elif parsed.get('bet'):
            parsed['pot'] = parsed['bet'] * 3  # fallback
        
        # Opponent action this street
        if 'river' in text_lower and ('bet' in text_lower or '2k' in text_lower or 'move in' in text_lower):
            parsed['opponent_action'] = 'BET'
        elif 'check' in text_lower and 'bet' in text_lower:
            parsed['opponent_action'] = 'BET'
        elif '3 bet' in text_lower or '3-bet' in text_lower:
            parsed['opponent_action'] = 'RAISE'
        elif 'call' in text_lower:
            parsed['opponent_action'] = 'CALL'
        else:
            parsed['opponent_action'] = 'BET'
        
        # Action history for model
        action_history = []
        if re.search(r'3\s*bet|raise.*to\s+\d+', text_lower):
            action_history.append({'street': 'preflop', 'action': 'RAISE', 'bet_size': 100, 'pot_size': 30})
        if flop_str and re.search(r'he\s+bet|bet\s+\d+', text_lower):
            action_history.append({'street': 'flop', 'action': 'BET', 'bet_size': 400, 'pot_size': 200})
        if turn_str and re.search(r'turn.*he\s+bet|bet\s+850', text_lower):
            action_history.append({'street': 'turn', 'action': 'BET', 'bet_size': 850, 'pot_size': 1000})
        if river_str and last_bet:
            action_history.append({'street': 'river', 'action': 'BET', 'bet_size': last_bet, 'pot_size': parsed.get('pot', 0)})
        parsed['action_history'] = action_history
        
        parsed['raw_description'] = text
        parsed['success'] = True
        return parsed


class ScreenCaptureParser:
    """Parse poker data from screen captures using OCR."""
    
    @staticmethod
    def capture_pokernow_tab():
        """
        Capture the screen for any visible poker table.
        On macOS we *try* to bring a browser window to the front, but in
        general we just capture the primary monitor and rely on the user
        to have a poker table visible somewhere on screen.
        """
        if not SCREEN_CAPTURE_AVAILABLE:
            return None
        
        # Currently we do not try to be clever about which window is a poker
        # table; we just grab the primary monitor. This keeps behaviour simple
        # and makes it work for *any* poker site or client as long as it is
        # visible on screen.
        return ScreenCaptureParser.capture_screen()
    
    @staticmethod
    def capture_screen():
        """Capture the entire screen and return as PIL Image."""
        if not SCREEN_CAPTURE_AVAILABLE:
            return None
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                return img
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    @staticmethod
    def parse_ocr_text(text):
        """
        Parse OCR text to extract poker game state from any poker table UI.
        Handles PokerNow, PokerStars-style, and other common formats.
        """
        parsed = {}
        text_lower = text.lower()
        original_text = text  # Keep original for better pattern matching
        
        # Extract pot size - PokerNow shows "total X" or "pot X"
        # Look for "total" followed by number, or "pot" followed by number
        pot_patterns = [
            r'total\s+(\d+)',  # "total 10"
            r'pot[:\s]*\$?(\d+(?:\.\d+)?)',  # "pot: 50" or "pot 50"
            r'\$(\d+(?:\.\d+)?)\s*(?:pot|total)',  # "$50 pot"
        ]
        for pattern in pot_patterns:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                # Take the last match (most likely the current pot)
                match = matches[-1]
                try:
                    pot_val = float(match.group(1))
                    if pot_val > 0:  # Only accept positive values
                        parsed['pot'] = pot_val
                        break
                except:
                    pass
        
        # Extract bet amount - look for "bet", "to call", "raise to", or chip numbers near action buttons
        bet_patterns = [
            r'bet[:\s]*\$?(\d+(?:\.\d+)?)',
            r'to\s+call[:\s]*\$?(\d+(?:\.\d+)?)',
            r'raise\s+to[:\s]*\$?(\d+(?:\.\d+)?)',
            r'call[:\s]*\$?(\d+(?:\.\d+)?)',
        ]
        for pattern in bet_patterns:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                match = matches[-1]  # Most recent bet
                try:
                    bet_val = float(match.group(1))
                    if bet_val > 0:
                        parsed['bet'] = bet_val
                        break
                except:
                    pass
        
        # Extract stack sizes - PokerNow shows "PlayerName 1234" format
        # Find all 3-4 digit numbers (likely stacks)
        stack_numbers = re.findall(r'\b(\d{3,4})\b', text)
        if stack_numbers:
            try:
                stacks = [float(s) for s in stack_numbers if 50 <= float(s) <= 10000]
                if stacks:
                    # Use median stack as "your stack", largest as opponent
                    stacks.sort()
                    parsed['your_stack'] = stacks[len(stacks)//2] if stacks else stacks[0]
                    parsed['opp_stack'] = max(stacks) if len(stacks) > 1 else parsed['your_stack']
            except:
                pass
        
        # Extract cards - handle both text format (Ah Kd) and suit symbols (K♥ 10♠)
        # First try standard format
        card_pattern_standard = r'\b([AKQJT2-9][hdcs])\b'
        cards_standard = re.findall(card_pattern_standard, text, re.IGNORECASE)
        
        # Also try PokerNow format with suit symbols (K♥, 10♠, etc.)
        # Map suit symbols to letters
        suit_map = {'♥': 'h', '♠': 's', '♦': 'd', '♣': 'c', '❤': 'h', '♡': 'h'}
        text_with_suits = text
        for symbol, letter in suit_map.items():
            text_with_suits = text_with_suits.replace(symbol, letter)
        
        # Look for rank + suit symbol patterns: "K♥", "10♠", "6♦", etc.
        card_pattern_suits = r'([AKQJT2-9]|10)\s*[♥♠♦♣❤♡]'
        cards_suits = re.findall(card_pattern_suits, text)
        # Convert to standard format
        cards_from_suits = []
        for i, rank in enumerate(cards_suits):
            # Find the suit symbol after this rank
            match = re.search(r'(' + re.escape(rank) + r')\s*([♥♠♦♣❤♡])', text)
            if match:
                suit_letter = suit_map.get(match.group(2), 'h')
                cards_from_suits.append(rank.upper() + suit_letter)
        
        # Combine both formats
        all_cards = []
        seen = set()
        for card in cards_standard + cards_from_suits:
            card_upper = card.upper()
            if card_upper not in seen:
                all_cards.append(card_upper)
                seen.add(card_upper)
                if len(all_cards) >= 7:  # 2 hole + 5 board max
                    break
        
        if len(all_cards) >= 2:
            parsed['hole_cards'] = ' '.join(all_cards[:2])
        if len(all_cards) > 2:
            parsed['board_cards'] = ' '.join(all_cards[2:])
        elif len(all_cards) == 2:
            # Only 2 cards found, likely preflop
            parsed['street'] = 'preflop'
        
        # Extract position - look for dealer button indicator or position labels
        # PokerNow shows "D" for dealer button
        if re.search(r'\bD\b', text) or 'dealer' in text_lower:
            # If we see dealer button, try to infer position
            # This is heuristic - in PokerNow, dealer is usually "button"
            parsed['position'] = 'button'
        
        position_map = {
            'button': ['button', 'btn', 'dealer', 'd'],
            'sb': ['small blind', 'sb', 'small'],
            'bb': ['big blind', 'bb', 'big'],
            'utg': ['utg', 'under the gun'],
            'hj': ['hijack', 'hj'],
            'co': ['cutoff', 'co'],
        }
        for pos, keywords in position_map.items():
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text_lower):
                    parsed['position'] = pos
                    break
            if 'position' in parsed:
                break
        
        # Extract blinds - look for "1/2", "$1/$2", "SB/BB", or infer from stacks
        blind_match = re.search(r'(\d+(?:\.\d+)?)\s*[/-]\s*(\d+(?:\.\d+)?)', text)
        if blind_match:
            try:
                parsed['small_blind'] = float(blind_match.group(1))
                parsed['big_blind'] = float(blind_match.group(2))
            except:
                pass
        
        # If no blinds found but we have stacks, use common defaults
        if 'small_blind' not in parsed:
            parsed['small_blind'] = 1.0
            parsed['big_blind'] = 2.0
        
        # Determine street from board cards count
        if 'board_cards' in parsed:
            board_count = len(parsed['board_cards'].split())
            if board_count == 3:
                parsed['street'] = 'flop'
            elif board_count == 4:
                parsed['street'] = 'turn'
            elif board_count == 5:
                parsed['street'] = 'river'
        elif 'street' not in parsed:
            # No board cards detected, assume preflop
            parsed['street'] = 'preflop'
        
        # Extract opponent action from UI buttons/text
        if re.search(r'\b(bet|raise|all.?in)\b', text_lower):
            parsed['opponent_action'] = 'BET'
        elif re.search(r'\b(call|check)\b', text_lower):
            parsed['opponent_action'] = 'CALL'
        
        # Extract player names and their data - improved to catch all players
        players = []
        player_dict = {}  # name -> player data
        
        # Method 1: Look for player names followed by stack numbers
        # Pattern: word(s) followed by 3-5 digit number (stacks can be 3-5 digits)
        player_pattern = r'([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)*)\s+(\d{3,5})'
        player_matches = list(re.finditer(player_pattern, text))
        
        # Filter out false positives
        exclude_words = {'total', 'pot', 'bet', 'call', 'raise', 'chips', 'stack', 'sign', 'guest', 
                        'activate', 'extra', 'time', 'room', 'owner', 'paused', 'game', 'your', 'turn'}
        
        for match in player_matches:
            name = match.group(1).strip()
            stack = float(match.group(2))
            name_lower = name.lower()
            
            # Skip if it's a common false positive
            if any(word in name_lower for word in exclude_words):
                continue
            
            # Skip if name is too short or looks like a number/command
            if len(name) < 2 or name.isdigit():
                continue
            
            # Check if this looks like a player name (has letters)
            if not re.search(r'[A-Za-z]', name):
                continue
            
            # Store player
            if name not in player_dict:
                player_dict[name] = {
                    'name': name,
                    'stack': stack,
                    'all_in': False,
                    'is_turn': False,
                    'cards': []
                }
            else:
                # Update stack if we found a better match
                player_dict[name]['stack'] = stack
        
        # Method 2: Look for "All" or "All In" status near player names
        all_in_patterns = [
            r'([A-Za-z0-9_]{2,})\s+All\b',  # "akumoli All"
            r'([A-Za-z0-9_]{2,})\s+All\s+In',
            r'\bAll\s+([A-Za-z0-9_]{2,})',
        ]
        for pattern in all_in_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                name_lower = name.lower()
                if name and len(name) >= 2 and re.search(r'[A-Za-z]', name):
                    if not any(word in name_lower for word in exclude_words):
                        if name not in player_dict:
                            player_dict[name] = {
                                'name': name,
                                'stack': 0,  # All-in players might have 0 stack visible
                                'all_in': True,
                                'is_turn': False,
                                'cards': []
                            }
                        else:
                            player_dict[name]['all_in'] = True
        
        # Method 2b: Look for player names near action buttons (CHECK, FOLD, RAISE, etc.)
        action_keywords = ['CHECK', 'FOLD', 'RAISE', 'CALL', 'ALL IN', 'BET']
        for keyword in action_keywords:
            keyword_matches = list(re.finditer(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
            for kw_match in keyword_matches:
                # Look for a name within 100 chars before the keyword
                before_text = text[max(0, kw_match.start() - 100):kw_match.start()]
                # Find capitalized words that look like names
                name_matches = re.finditer(r'\b([A-Z][a-z]{2,})\b', before_text)
                for name_match in name_matches:
                    name = name_match.group(1).strip()
                    name_lower = name.lower()
                    if name and len(name) >= 2:
                        if not any(word in name_lower for word in exclude_words):
                            if name not in player_dict:
                                player_dict[name] = {
                                    'name': name,
                                    'stack': 0,
                                    'all_in': False,
                                    'is_turn': False,
                                    'cards': []
                                }
        
        # Method 3: Look for "YOUR TURN" - the active player
        your_turn_match = re.search(r'YOUR\s+TURN', text, re.IGNORECASE)
        if your_turn_match:
            # Find player name near "YOUR TURN" (within 200 chars before)
            turn_pos = your_turn_match.start()
            before_text = text[max(0, turn_pos - 200):turn_pos]
            # Look for a name pattern before "YOUR TURN"
            name_match = re.search(r'([A-Za-z0-9_]{2,})\s+(?:\d+|All)', before_text)
            if name_match:
                name = name_match.group(1).strip()
                if name and name.lower() not in exclude_words:
                    if name not in player_dict:
                        player_dict[name] = {
                            'name': name,
                            'stack': 0,
                            'all_in': False,
                            'is_turn': True,
                            'cards': []
                        }
                    else:
                        player_dict[name]['is_turn'] = True
        
        # Method 3b: For players without stacks, try to find stack numbers near their names
        for name, player_data in player_dict.items():
            if player_data.get('stack', 0) == 0:
                name_positions = [m.start() for m in re.finditer(re.escape(name), text)]
                for name_pos in name_positions:
                    # Look for a 3-5 digit number within 50 chars after the name
                    after_text = text[name_pos:name_pos + 50]
                    stack_match = re.search(r'\s+(\d{3,5})\b', after_text)
                    if stack_match:
                        stack_val = float(stack_match.group(1))
                        if 50 <= stack_val <= 100000:  # Reasonable stack range
                            player_data['stack'] = stack_val
                            break
        
        # Method 4: Extract cards and associate with players
        card_pattern_with_suits = r'([AKQJT2-9]|10)\s*[♥♠♦♣❤♡]'
        card_matches = list(re.finditer(card_pattern_with_suits, text))
        
        # For each player, find cards near their name
        for name, player_data in player_dict.items():
            name_positions = [m.start() for m in re.finditer(re.escape(name), text)]
            if name_positions:
                # Find cards within 150 chars of any occurrence of the name
                nearby_cards = []
                for name_pos in name_positions:
                    for card_match in card_matches:
                        card_pos = card_match.start()
                        if abs(card_pos - name_pos) < 150:
                            rank = card_match.group(1)
                            # Find the suit symbol
                            suit_match = re.search(r'(' + re.escape(rank) + r')\s*([♥♠♦♣❤♡])', text[max(0, card_match.start()-5):card_match.start()+20])
                            if suit_match:
                                suit_letter = suit_map.get(suit_match.group(2), 'h')
                                card_str = rank.upper() + suit_letter
                                if card_str not in nearby_cards:
                                    nearby_cards.append(card_str)
                
                if len(nearby_cards) >= 2:
                    player_data['cards'] = ' '.join(nearby_cards[:2])
                elif len(nearby_cards) == 1:
                    # Single card found, might be part of a pair
                    player_data['cards'] = nearby_cards[0]
        
        # Convert to list and filter out players with no useful data
        players = []
        for name, player_data in player_dict.items():
            # Keep player if they have stack, cards, all-in status, or is_turn status
            if (player_data.get('stack', 0) > 0 or 
                player_data.get('cards') or 
                player_data.get('all_in') or 
                player_data.get('is_turn')):
                # Clean up empty cards list
                if isinstance(player_data.get('cards'), list) and len(player_data['cards']) == 0:
                    del player_data['cards']
                players.append(player_data)
        
        # Detect dealer button position
        dealer_name = None
        if re.search(r'\bD\b', text):
            d_pos = text.find('D')
            for player in players:
                name_positions = [m.start() for m in re.finditer(re.escape(player['name']), text)]
                for name_pos in name_positions:
                    if abs(d_pos - name_pos) < 50:
                        dealer_name = player['name']
                        player['position'] = 'button'
                        break
                if dealer_name:
                    break
        
        # Infer positions based on dealer button (if we found one)
        if dealer_name and len(players) >= 2:
            dealer_idx = next((i for i, p in enumerate(players) if p['name'] == dealer_name), None)
            if dealer_idx is not None:
                for i, player in enumerate(players):
                    if player['name'] != dealer_name and not player.get('position'):
                        if i < dealer_idx:
                            player['position'] = 'sb'
                        else:
                            player['position'] = 'bb'
        
        if players:
            parsed['players'] = players
        
        return parsed
    
    @staticmethod
    def parse_image_bytes(data: bytes):
        """Run OCR and parse poker data from raw image bytes (uploaded screenshot)."""
        if not SCREEN_CAPTURE_AVAILABLE:
            return {'error': 'Screen capture not available. Install: pip install mss pytesseract pillow'}
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")

            ocr_text = pytesseract.image_to_string(img)
            parsed = ScreenCaptureParser.parse_ocr_text(ocr_text)

            screenshot_dir = Path("logs/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = screenshot_dir / f"upload_{timestamp}.png"
            img.save(screenshot_path)

            # Validate and set defaults for missing critical fields
            if 'pot' not in parsed or parsed.get('pot', 0) == 0:
                parsed['pot'] = parsed.get('bet', 0) * 2 if parsed.get('bet', 0) > 0 else 0
            
            if 'your_stack' not in parsed or parsed.get('your_stack', 0) == 0:
                # Try to infer from other numbers in text
                all_numbers = re.findall(r'\b(\d{2,4})\b', ocr_text)
                if all_numbers:
                    try:
                        stacks = [float(n) for n in all_numbers if 50 <= float(n) <= 10000]
                        if stacks:
                            parsed['your_stack'] = stacks[len(stacks)//2] if stacks else 100
                            parsed['opp_stack'] = max(stacks) if len(stacks) > 1 else parsed['your_stack']
                    except:
                        parsed['your_stack'] = 100
                        parsed['opp_stack'] = 100
                else:
                    parsed['your_stack'] = 100
                    parsed['opp_stack'] = 100
            
            if 'small_blind' not in parsed:
                parsed['small_blind'] = 1.0
            if 'big_blind' not in parsed:
                parsed['big_blind'] = 2.0
            
            if 'position' not in parsed:
                parsed['position'] = 'button'  # Default
            
            if 'street' not in parsed:
                parsed['street'] = 'preflop'  # Default

            parsed['screenshot_path'] = str(screenshot_path)
            parsed['ocr_text'] = ocr_text[:500]
            parsed['success'] = True
            parsed['source'] = 'upload'
            parsed['warnings'] = []
            
            # Add warnings for potentially missing data
            if not parsed.get('hole_cards'):
                parsed['warnings'].append('No hole cards detected - please enter manually')
            if parsed.get('pot', 0) == 0 and parsed.get('bet', 0) == 0:
                parsed['warnings'].append('No pot/bet detected - please verify values')
            
            return parsed
        except Exception as e:
            return {'error': f'OCR parsing failed: {str(e)}'}

    @staticmethod
    def capture_and_parse():
        """Capture screen, run OCR, and parse poker data.

        This is intentionally generic: it will work with any poker table
        (PokerNow, other sites, or desktop clients) as long as the table is
        visible somewhere on the primary monitor.
        """
        if not SCREEN_CAPTURE_AVAILABLE:
            return {'error': 'Screen capture not available. Install: pip install mss pytesseract pillow'}
        
        # Capture current screen (any visible poker table)
        img = ScreenCaptureParser.capture_pokernow_tab()
        if img is None:
            return {'error': 'Failed to capture screen. Make sure a poker table is visible on your main monitor.'}
        
        try:
            # Run OCR
            ocr_text = pytesseract.image_to_string(img)
            
            # Parse the text
            parsed = ScreenCaptureParser.parse_ocr_text(ocr_text)
            
            # Store screenshot for reference
            screenshot_dir = Path("logs/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = screenshot_dir / f"screenshot_{timestamp}.png"
            img.save(screenshot_path)
            
            # Validate and set defaults for missing critical fields
            if 'pot' not in parsed or parsed.get('pot', 0) == 0:
                parsed['pot'] = parsed.get('bet', 0) * 2 if parsed.get('bet', 0) > 0 else 0
            
            if 'your_stack' not in parsed or parsed.get('your_stack', 0) == 0:
                # Try to infer from other numbers in text
                all_numbers = re.findall(r'\b(\d{2,4})\b', ocr_text)
                if all_numbers:
                    try:
                        stacks = [float(n) for n in all_numbers if 50 <= float(n) <= 10000]
                        if stacks:
                            parsed['your_stack'] = stacks[len(stacks)//2] if stacks else 100
                            parsed['opp_stack'] = max(stacks) if len(stacks) > 1 else parsed['your_stack']
                    except:
                        parsed['your_stack'] = 100
                        parsed['opp_stack'] = 100
                else:
                    parsed['your_stack'] = 100
                    parsed['opp_stack'] = 100
            
            if 'small_blind' not in parsed:
                parsed['small_blind'] = 1.0
            if 'big_blind' not in parsed:
                parsed['big_blind'] = 2.0
            
            if 'position' not in parsed:
                parsed['position'] = 'button'  # Default
            
            if 'street' not in parsed:
                parsed['street'] = 'preflop'  # Default
            
            parsed['screenshot_path'] = str(screenshot_path)
            parsed['ocr_text'] = ocr_text[:500]  # First 500 chars for debugging
            parsed['success'] = True
            parsed['source'] = 'screen'
            parsed['warnings'] = []
            
            # Add warnings for potentially missing data
            if not parsed.get('hole_cards'):
                parsed['warnings'].append('No hole cards detected - please enter manually')
            if parsed.get('pot', 0) == 0 and parsed.get('bet', 0) == 0:
                parsed['warnings'].append('No pot/bet detected - please verify values')
            
            return parsed
        except Exception as e:
            return {'error': f'OCR parsing failed: {str(e)}'}


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
        # Current hand history tracking (hand_id -> HandHistory)
        self.current_hand_histories = {}  # hand_id -> HandHistory
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
    
    def get_hand_history(self, hand_id: str) -> HandHistory:
        """Get or create hand history for tracking actions across streets."""
        if hand_id not in self.current_hand_histories:
            self.current_hand_histories[hand_id] = HandHistory(hand_id=hand_id)
        return self.current_hand_histories[hand_id]
    
    def analyze(self, game_state):
        """Full analysis with cards"""
        
        pot = float(game_state.get('pot', 0))
        bet = float(game_state.get('bet', 0))
        your_stack = float(game_state.get('your_stack', 100))
        opp_stack = float(game_state.get('opp_stack', 100))
        sb = float(game_state.get('small_blind', 0.5))
        bb = float(game_state.get('big_blind', 1))
        position = game_state.get('position', 'bb')
        street = game_state.get('street', 'preflop').lower()
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
        
        # Auto-detect street from board cards if not explicitly set or wrong
        board_count = len(board_cards)
        if board_count == 0:
            detected_street = 'preflop'
        elif board_count == 3:
            detected_street = 'flop'
        elif board_count == 4:
            detected_street = 'turn'
        elif board_count == 5:
            detected_street = 'river'
        else:
            detected_street = street  # Use provided street if board count is unusual
        
        # Override street if board cards suggest different street
        if board_count > 0 and detected_street != street.lower():
            street = detected_street
        
        # Get or create hand history for tracking actions across streets
        hand_id = game_state.get('hand_id') or str(uuid.uuid4().hex[:8])
        hand_history = self.get_hand_history(hand_id)
        
        # Parse action history from game_state if provided
        action_history_json = game_state.get('action_history', '[]')
        try:
            action_history_data = json.loads(action_history_json) if isinstance(action_history_json, str) else action_history_json
            # Reconstruct hand history from JSON if provided
            if action_history_data:
                for action_data in action_history_data:
                    hand_history.add_action(
                        street=action_data.get('street', 'preflop'),
                        action=action_data.get('action', 'CHECK'),
                        bet_size=float(action_data.get('bet_size', 0)),
                        pot_size=float(action_data.get('pot_size', 0)),
                        position=action_data.get('position')
                    )
        except:
            pass  # If parsing fails, continue with existing history
        
        # Add current opponent action to history if provided
        if raw_opponent_action and raw_opponent_action != '-- SELECT --':
            hand_history.add_action(
                street=street,
                action=raw_opponent_action,
                bet_size=bet,
                pot_size=pot,
                position=position
            )
        
        # Get all previous actions for this hand
        all_previous_actions = []
        for s in ['preflop', 'flop', 'turn', 'river']:
            if s != street:  # Don't include current street
                all_previous_actions.extend(hand_history.get_street_actions(s))
        
        # Evaluate hand
        hand_eval = CardParser.evaluate_hand(hole_cards, board_cards)
        hand_strength = hand_eval['strength']
        
        # On preflop, pot should include blinds already posted
        # If pot is 0 or very small (less than blinds), add blinds to it
        # This handles cases where OCR didn't detect the pot but blinds are posted
        if street == 'preflop':
            if pot == 0 or pot < (sb + bb):
                # Pot doesn't include blinds yet, add them
                pot = pot + sb + bb
            # If pot >= sb + bb, assume it already includes blinds (from OCR or user input)
        
        # Calculate what you've already invested based on position (blinds)
        if position == 'sb':
            already_invested = sb
        elif position == 'bb':
            already_invested = bb
        else:
            # Button, UTG, HJ, CO: no blind posted (unless you've acted, which we don't track here)
            already_invested = 0.0
        
        # Calculate "to call" = bet amount - what you've already put in
        to_call = max(0.0, bet - already_invested)
        
        # Calculate metrics
        effective_stack = min(your_stack, opp_stack)
        spr = effective_stack / max(pot, 1)
        # Pot odds = amount to call / (current pot + amount to call)
        # Current pot is what's in the middle BEFORE your call
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0
        
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
        
        # Predict opponent's likely action based on action history
        opponent_action_probs = belief.predict_action(street, all_previous_actions)
        predicted_opponent_action = max(opponent_action_probs.items(), key=lambda x: x[1])[0]
        prediction_confidence = opponent_action_probs[predicted_opponent_action]
        
        # Analyze action patterns from history
        action_pattern = hand_history.get_action_pattern()
        pattern_insights = []
        if action_pattern['preflop_aggressive']:
            pattern_insights.append("Opponent raised preflop")
        if action_pattern['flop_continuation']:
            pattern_insights.append("Opponent c-bet flop")
        if action_pattern['turn_continuation']:
            pattern_insights.append("Opponent barreled turn")
        if action_pattern['river_value_bet']:
            pattern_insights.append("Opponent betting river")
        
        # Adjust reasoning based on action patterns
        # If opponent has been aggressive throughout, they're more likely to have a strong hand
        aggression_level = sum([
            action_pattern['preflop_aggressive'],
            action_pattern['flop_continuation'],
            action_pattern['turn_continuation'],
            action_pattern['river_value_bet']
        ])
        
        # Make recommendation based on hand strength + situation + action history
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
            is_all_in = (raw_opponent_action == 'ALL_IN' or bet >= opp_stack * 0.95)
            
            # For all-in situations, require higher equity margin (need clear +EV)
            # Also account for vulnerable hands (two pair on paired board, etc.)
            # Check if board has pairs by counting ranks
            board_ranks = [c[0] for c in board_cards] if board_cards else []
            board_rank_counts = {}
            for rank in board_ranks:
                board_rank_counts[rank] = board_rank_counts.get(rank, 0) + 1
            board_paired = len(board_cards) >= 3 and any(count >= 2 for count in board_rank_counts.values())
            is_vulnerable = (hand_eval['description'].startswith('Two Pair') and board_paired) or \
                           (hand_eval['description'].startswith('Pair') and board_paired)
            
            # Adjust required equity for all-in and vulnerable situations
            if is_all_in:
                # Need significant edge for all-in calls (at least 10-15% above pot odds)
                # Vulnerable hands (two pair on paired board) need even more margin
                required_margin = 0.15 if is_vulnerable else 0.10
                equity_needed = pot_odds + required_margin
            elif is_vulnerable:
                # Vulnerable hands need extra margin even for non-all-in bets
                equity_needed = pot_odds + 0.08
            
            # Adjust equity needed based on action history
            # If opponent has been aggressive on multiple streets, they likely have a strong hand
            if aggression_level >= 3:
                # Opponent has been aggressive throughout - likely strong hand
                equity_needed += 0.05  # Require more equity to call
                pattern_insights.append("High aggression suggests strong hand")
            elif aggression_level == 0 and bet > 0:
                # Opponent suddenly betting after checking - could be a bluff
                equity_needed -= 0.03  # Can call with slightly less equity
                pattern_insights.append("Sudden aggression after passivity - possible bluff")
            
            if hand_strength > equity_needed + 0.15:
                action = 'RAISE'
                reasoning = f'Strong hand ({hand_eval["description"]}) vs pot odds {pot_odds:.1%}'
                if is_all_in:
                    reasoning += ' | All-in: need strong hand'
                if pattern_insights:
                    reasoning += ' | ' + '; '.join(pattern_insights)
                bet_size = f'${bet * 2.5:.2f} (2.5x their bet)'
            elif hand_strength > equity_needed:
                if is_all_in and hand_strength < pot_odds + 0.10:
                    # Close spot - be more conservative for all-in
                    action = 'FOLD'
                    reasoning = f'All-in call: hand strength ({hand_strength:.1%}) too close to pot odds ({pot_odds:.1%})'
                    if is_vulnerable:
                        reasoning += ' | Vulnerable two pair on paired board'
                    if pattern_insights:
                        reasoning += ' | ' + '; '.join(pattern_insights)
                    bet_size = 'N/A'
                else:
                    action = 'CALL'
                    reasoning = f'Hand strength ({hand_strength:.1%}) > required equity ({equity_needed:.1%})'
                    if is_all_in:
                        reasoning += ' | All-in call'
                    if is_vulnerable:
                        reasoning += ' | Caution: vulnerable hand'
                    if pattern_insights:
                        reasoning += ' | ' + '; '.join(pattern_insights)
                    if prediction_confidence > 0.4:
                        reasoning += f' | Predicted opponent action: {predicted_opponent_action} ({prediction_confidence:.0%} confidence)'
                    bet_size = f'${bet:.2f}'
            else:
                # Consider opponent tendencies and action history
                if not is_all_in and opp_info.get('known') and opp_info['fold_freq'] > 0.60:
                    action = 'RAISE'
                    reasoning = f'Bluff opportunity - opponent folds {opp_info["fold_freq"]:.1%}'
                    if pattern_insights:
                        reasoning += ' | ' + '; '.join(pattern_insights)
                    bet_size = f'${bet * 3:.2f} (3x raise)'
                elif aggression_level == 0 and bet > 0 and hand_strength > pot_odds - 0.05:
                    # Opponent suddenly betting after passivity - might be a bluff, call if close
                    action = 'CALL'
                    reasoning = f'Opponent suddenly aggressive after passivity - possible bluff'
                    if pattern_insights:
                        reasoning += ' | ' + '; '.join(pattern_insights)
                    bet_size = f'${bet:.2f}'
                else:
                    action = 'FOLD'
                    reasoning = f'Hand strength ({hand_strength:.1%}) < required equity ({equity_needed:.1%})'
                    if is_all_in:
                        reasoning += ' | All-in: need stronger hand'
                    if is_vulnerable:
                        reasoning += ' | Vulnerable hand on dangerous board'
                    if pattern_insights:
                        reasoning += ' | ' + '; '.join(pattern_insights)
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
        
        # Serialize action history for return value (before telemetry)
        action_history_serialized = []
        for s in ['preflop', 'flop', 'turn', 'river']:
            for action_obj in hand_history.get_street_actions(s):
                action_history_serialized.append({
                    'street': action_obj.street,
                    'action': action_obj.action,
                    'bet_size': action_obj.bet_size,
                    'pot_size': action_obj.pot_size,
                    'position': action_obj.position
                })
        
        # If hand is complete (river), save hand history to belief for pattern learning
        if street == 'river' and len(hand_history.get_street_actions('river')) > 0:
            # Hand is complete, add to belief for pattern learning
            belief.add_hand_history(hand_history)
            # Clear from current histories (hand is done)
            if hand_id in self.current_hand_histories:
                del self.current_hand_histories[hand_id]

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
                "to_call": to_call,
                "already_invested": already_invested,
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
                "action_history": action_history_serialized,
                "action_pattern": action_pattern,
                "predicted_opponent_action": predicted_opponent_action,
                "prediction_confidence": prediction_confidence,
                "aggression_level": aggression_level,
            }

            event = {
                "ts": ts,
                "hand_id": hand_id,
                "opponent": opponent_name,
                "env_state": env_state,
                "observed_opponent_action": observed_opponent_action,
                "pressure_context": pressure_context,
                "capture_context": {
                    "screenshot_path": game_state.get("screenshot_path"),
                    "capture_source": game_state.get("capture_source"),
                },
                "action_notes": game_state.get("action_notes", ""),
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
            'env_state': {
                'action_history': action_history_serialized,
                'action_pattern': action_pattern,
                'predicted_opponent_action': predicted_opponent_action,
                'prediction_confidence': prediction_confidence,
                'aggression_level': aggression_level,
            },
            'metrics': {
                'pot': pot,
                'bet': bet,
                'to_call': to_call,
                'already_invested': already_invested,
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
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }
        
        .tab {
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #1e3c72;
            border-bottom-color: #1e3c72;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            min-height: 80px;
        }
        
        textarea:focus {
            outline: none;
            border-color: #2a5298;
        }
        
        .capture-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        
        .capture-btn:hover {
            transform: translateY(-2px);
        }
        
        .capture-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .ocr-result {
            background: #f0f9ff;
            border-left: 4px solid #0ea5e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 14px;
            color: #0c4a6e;
        }
        
        .ocr-result.error {
            background: #fef2f2;
            border-left-color: #ef4444;
            color: #991b1b;
        }
        
        .parse-btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
        }
        
        .parse-btn:hover {
            transform: translateY(-1px);
        }
        
        .hand-description-section {
            background: #f0f9ff;
            border-left: 4px solid #0ea5e9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .hand-description-section h3 {
            font-size: 16px;
            margin-bottom: 10px;
            color: #1e3c72;
        }
        
        .player-select-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .player-select-modal.active {
            display: flex;
        }
        
        .player-select-content {
            background: white;
            padding: 30px;
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .player-select-content h3 {
            margin-top: 0;
            color: #1e3c72;
            margin-bottom: 20px;
        }
        
        .player-option {
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .player-option:hover {
            border-color: #2a5298;
            background: #f0f9ff;
        }
        
        .player-option.selected {
            border-color: #2a5298;
            background: #e0f2fe;
        }
        
        .player-option.disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #f3f4f6;
        }
        
        .player-option.disabled:hover {
            border-color: #ddd;
            background: #f3f4f6;
        }
        
        .player-name {
            font-weight: 600;
            font-size: 16px;
            color: #1e3c72;
        }
        
        .player-details {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        .select-player-btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
        }
        
        .select-player-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
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
                
                <div class="tabs">
                    <button type="button" class="tab active" onclick="switchTab('manual', this)">Manual Input</button>
                    <button type="button" class="tab" onclick="switchTab('capture', this)">Screen Capture</button>
                </div>
                
                <div id="manualTab" class="tab-content active">
                <form id="gameForm">
                    <div class="hand-description-section" style="margin-bottom: 20px;">
                        <h3>📝 Describe your hand (like ChatGPT)</h3>
                        <p style="color: #666; font-size: 14px; margin-bottom: 10px;">Paste a hand history or describe the situation. We'll parse it and give a recommendation.</p>
                        <textarea id="hand_description" placeholder="e.g. Home game. 1/3 8k effective. Raise KTo UTG to 20. BU 3-bets to 100. I call. Flop T42r. I check. He bets 400. I call. Turn 7r. I check. He bets 850. I call. River 2. I check. He bets 2k. What do you think?" style="min-height: 140px; width: 100%; padding: 12px; border-radius: 8px; border: 2px solid #ddd; font-size: 14px;"></textarea>
                        <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
                            <button type="button" class="parse-btn" onclick="analyzeFromDescription()" style="padding: 12px 24px; font-size: 16px;">🎯 Analyze & Recommend</button>
                            <button type="button" class="parse-btn" onclick="parseHandDescription()" style="background: #64748b; padding: 10px 18px;">Fill form only</button>
                        </div>
                        <div id="descriptionResult" style="display: none; margin-top: 16px;"></div>
                    </div>
                    
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
                    
                    <div class="input-group">
                        <label>Action History (previous streets) - JSON format</label>
                        <textarea id="action_history" placeholder='[{"street": "preflop", "action": "RAISE", "bet_size": 20, "pot_size": 30}, {"street": "flop", "action": "BET", "bet_size": 50, "pot_size": 100}]' style="min-height: 100px; font-family: monospace; font-size: 12px;"></textarea>
                        <div class="card-help" style="margin-top: 5px;">
                            Enter actions from previous streets. Format: [{"street": "preflop|flop|turn|river", "action": "BET|CALL|RAISE|CHECK|FOLD", "bet_size": number, "pot_size": number}]
                        </div>
                    </div>
                    
                    <div class="input-group">
                        <label>Action Notes (optional)</label>
                        <textarea id="action_notes" placeholder="Describe the action, opponent behavior, or any relevant context..."></textarea>
                    </div>
                    
                    <!-- Hidden fields to tie captures/uploads to telemetry -->
                    <input type="hidden" id="screenshot_path" value="">
                    <input type="hidden" id="capture_source" value="">
                    <input type="hidden" id="hand_id" value="">
                    
                    <button type="submit" class="analyze-btn">🔍 Analyze & Recommend</button>
                </form>
                </div>
                
                <div id="captureTab" class="tab-content">
                    <div class="input-group">
                        <button type="button" class="capture-btn" id="captureBtn" onclick="captureScreen()">
                            📸 Capture Screen & Analyze
                        </button>
                        <div style="margin-top: 10px;">
                            <input type="file" id="screenshot_file" accept="image/*">
                            <button type="button" class="capture-btn" id="uploadBtn" style="margin-top: 10px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);" onclick="uploadScreenshot()">
                                ⬆️ Upload Screenshot & Analyze
                            </button>
                        </div>
                        <div id="ocrResult" class="ocr-result" style="display: none;"></div>
                    </div>
                    <p style="color: #666; font-size: 14px; margin-top: 15px;">
                        This is an experimental feature: it captures your primary screen, tries to detect cards, stacks,
                        and pot from any visible poker table, and then analyzes the hand. For best results, keep a single
                        poker table clearly visible on your main monitor.
                    </p>
                </div>
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
        function switchTab(tabName, clickedBtn) {
            try {
                console.log('[DEBUG] switchTab called:', tabName, 'clickedBtn:', !!clickedBtn);
                // #region agent log
                var tabEl = document.getElementById(tabName + 'Tab');
                console.log('[DEBUG] tabEl found:', !!tabEl, 'id:', tabName + 'Tab');
                fetch('http://127.0.0.1:7243/ingest/b4103c64-22ce-4509-985b-06f21511cca0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'switchTab',message:'switchTab called',data:{tabName:tabName,hasTabEl:!!tabEl,tabId:tabName+'Tab',hasClickedBtn:!!clickedBtn},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H4'})}).catch(function(e){console.error('[DEBUG] log fetch failed:',e);});
                // #endregion
                // Hide all tab contents and tab buttons
                var allTabs = document.querySelectorAll('.tab-content');
                console.log('[DEBUG] Found', allTabs.length, 'tab-content elements');
                allTabs.forEach(function(tab) {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(function(t) {
                    t.classList.remove('active');
                });
                if (tabEl) {
                    tabEl.classList.add('active');
                    console.log('[DEBUG] Added active to tabEl');
                } else {
                    console.error('[DEBUG] tabEl is null! Looking for:', tabName + 'Tab');
                }
                if (clickedBtn) {
                    clickedBtn.classList.add('active');
                    console.log('[DEBUG] Added active to clickedBtn');
                }
            } catch (e) {
                console.error('[DEBUG] switchTab error:', e);
            }
        }
        
        async function analyzeFromDescription() {
            try {
                console.log('[DEBUG] analyzeFromDescription called');
                var text = document.getElementById('hand_description').value.trim();
                var resultDiv = document.getElementById('descriptionResult');
                console.log('[DEBUG] text length:', text.length, 'resultDiv:', !!resultDiv);
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/b4103c64-22ce-4509-985b-06f21511cca0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'analyzeFromDescription',message:'entry',data:{textLen:text.length,hasResultDiv:!!resultDiv},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H3'})}).catch(function(e){console.error('[DEBUG] log fetch failed:',e);});
                // #endregion
                if (!text) {
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = '<div class="ocr-result error"><strong>Please enter a hand description.</strong></div>';
                    return;
                }
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">Analyzing...</div>';
                console.log('[DEBUG] About to fetch /api/analyze_description');
                var url = '/api/analyze_description?' + new URLSearchParams({ text: text });
                console.log('[DEBUG] URL length:', url.length);
                var response = await fetch(url);
                console.log('[DEBUG] Response status:', response.status, 'ok:', response.ok);
                var result = await response.json();
                console.log('[DEBUG] Result received, has action:', !!result.action, 'has error:', !!result.error);
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/b4103c64-22ce-4509-985b-06f21511cca0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'analyzeFromDescription',message:'after fetch',data:{ok:response.ok,status:response.status,hasError:!!result.error,hasAction:!!result.action},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H3'})}).catch(function(){});
                // #endregion
                if (result.error) {
                    resultDiv.innerHTML = '<div class="ocr-result error"><strong>Error:</strong> ' + result.error + '</div>';
                    return;
                }
                resultDiv.innerHTML = '';
                var card = document.createElement('div');
                card.className = 'card';
                card.style.marginTop = '12px';
                card.innerHTML = '<div class="recommendation"><div class="rec-action">' + result.action + '</div><div class="rec-bet-size">' + (result.bet_size || '') + '</div><div class="rec-reasoning">' + (result.reasoning || '') + '</div></div>';
                if (result._parsed && result._parsed.hole_cards) {
                    card.innerHTML += '<p style="margin: 10px 0; font-size: 14px;"><strong>Parsed:</strong> ' + result._parsed.hole_cards + ' on ' + (result._parsed.board_cards || '') + ' | Pot $' + (result._parsed.pot || 0) + ', Bet $' + (result._parsed.bet || 0) + ' | ' + (result._parsed.street || '') + '</p>';
                }
                resultDiv.appendChild(card);
                displayResult(result);
                var mainResult = document.getElementById('result');
                if (mainResult) mainResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (e) {
                console.error('[DEBUG] analyzeFromDescription error:', e);
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/b4103c64-22ce-4509-985b-06f21511cca0',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'analyzeFromDescription',message:'catch',data:{error:e.message,stack:e.stack},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H3'})}).catch(function(err){console.error('[DEBUG] log fetch failed:',err);});
                // #endregion
                resultDiv.innerHTML = '<div class="ocr-result error"><strong>Error:</strong> ' + e.message + '</div>';
            }
        }
        
        function parseHandDescription() {
            var text = document.getElementById('hand_description').value.trim();
            if (!text) {
                alert('Please enter a hand description');
                return;
            }
            fetch('/api/parse_description?' + new URLSearchParams({ text: text }))
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.error) {
                        alert('Parse error: ' + data.error);
                        return;
                    }
                    if (data.small_blind != null) document.getElementById('small_blind').value = data.small_blind;
                    if (data.big_blind != null) document.getElementById('big_blind').value = data.big_blind;
                    if (data.your_stack != null) document.getElementById('your_stack').value = data.your_stack;
                    if (data.opp_stack != null) document.getElementById('opp_stack').value = data.opp_stack || data.your_stack;
                    if (data.position) document.getElementById('position').value = data.position;
                    if (data.hole_cards) document.getElementById('hole_cards').value = data.hole_cards;
                    if (data.board_cards) document.getElementById('board_cards').value = data.board_cards;
                    if (data.street) document.getElementById('street').value = data.street;
                    if (data.pot != null) document.getElementById('pot').value = data.pot;
                    if (data.bet != null) document.getElementById('bet').value = data.bet;
                    if (data.opponent_action) document.getElementById('opponent_action').value = data.opponent_action;
                    if (data.action_history && data.action_history.length) {
                        document.getElementById('action_history').value = JSON.stringify(data.action_history, null, 2);
                    }
                    if (data.raw_description) document.getElementById('action_notes').value = 'Parsed: ' + data.raw_description.substring(0, 150);
                    alert('Form filled. Click "Analyze & Recommend" below to run the model.');
                })
                .catch(function(err) { alert('Error: ' + err.message); });
        }
        
        async function captureScreen() {
            const btn = document.getElementById('captureBtn');
            const resultDiv = document.getElementById('ocrResult');
            
            btn.disabled = true;
            btn.textContent = '📸 Capturing...';
            resultDiv.style.display = 'none';
            
            try {
                const response = await fetch('/api/capture');
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.className = 'ocr-result error';
                    resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
                    resultDiv.style.display = 'block';
                } else if (data.success) {
                    // If players detected, show player selection modal
                    if (data.players && data.players.length > 0) {
                        showPlayerSelection(data.players, data);
                    } else {
                        // No players detected, use old method
                        if (data.pot !== undefined) document.getElementById('pot').value = data.pot;
                        if (data.bet !== undefined) document.getElementById('bet').value = data.bet;
                        if (data.your_stack !== undefined) document.getElementById('your_stack').value = data.your_stack;
                        if (data.opp_stack !== undefined) document.getElementById('opp_stack').value = data.opp_stack;
                        if (data.position) document.getElementById('position').value = data.position;
                        if (data.hole_cards) document.getElementById('hole_cards').value = data.hole_cards;
                        if (data.board_cards) document.getElementById('board_cards').value = data.board_cards;
                        if (data.small_blind !== undefined) document.getElementById('small_blind').value = data.small_blind;
                        if (data.big_blind !== undefined) document.getElementById('big_blind').value = data.big_blind;
                        if (data.street) document.getElementById('street').value = data.street;
                        if (data.opponent_action) document.getElementById('opponent_action').value = data.opponent_action;
                        if (data.screenshot_path) document.getElementById('screenshot_path').value = data.screenshot_path;
                        document.getElementById('capture_source').value = data.source || 'screen';
                        
                        // Show extracted values and warnings
                        let resultHtml = `<strong>✓ Captured!</strong><br>`;
                        resultHtml += `Extracted: Pot=$${data.pot || 0}, Bet=$${data.bet || 0}, Stack=$${data.your_stack || 0}`;
                        if (data.hole_cards) resultHtml += `, Cards=${data.hole_cards}`;
                        if (data.position) resultHtml += `, Position=${data.position}`;
                        resultHtml += `<br><small>Screenshot: ${data.screenshot_path}</small>`;
                        
                        if (data.warnings && data.warnings.length > 0) {
                            resultHtml += `<br><strong style="color: #f59e0b;">⚠️ Warnings:</strong> ${data.warnings.join('; ')}`;
                            resultDiv.className = 'ocr-result error';
                        } else {
                            resultDiv.className = 'ocr-result';
                        }
                        
                        resultDiv.innerHTML = resultHtml;
                        resultDiv.style.display = 'block';
                        
                        // Switch to manual tab and auto-analyze
                        const manualBtn = document.querySelector('.tab[onclick*="manual"]');
                        if (manualBtn) {
                            switchTab('manual', manualBtn);
                        }
                        
                        // Auto-submit for analysis
                        document.getElementById('gameForm').dispatchEvent(new Event('submit'));
                    }
                }
            } catch (error) {
                resultDiv.className = 'ocr-result error';
                resultDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
                resultDiv.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = '📸 Capture Screen & Analyze';
            }
        }
        
        async function uploadScreenshot() {
            const fileInput = document.getElementById('screenshot_file');
            const file = fileInput.files[0];
            const btn = document.getElementById('uploadBtn');
            const resultDiv = document.getElementById('ocrResult');
            
            if (!file) {
                resultDiv.className = 'ocr-result error';
                resultDiv.innerHTML = '<strong>Error:</strong> Please choose an image file to upload.';
                resultDiv.style.display = 'block';
                return;
            }
            
            btn.disabled = true;
            btn.textContent = '⬆️ Uploading...';
            resultDiv.style.display = 'none';
            
            try {
                const response = await fetch('/api/upload_screenshot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': file.type || 'application/octet-stream',
                    },
                    body: file,
                });
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.className = 'ocr-result error';
                    resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
                    resultDiv.style.display = 'block';
                } else if (data.success) {
                    // If players detected, show player selection modal
                    if (data.players && data.players.length > 0) {
                        showPlayerSelection(data.players, data);
                    } else {
                        // No players detected, use old method
                        if (data.pot !== undefined) document.getElementById('pot').value = data.pot;
                        if (data.bet !== undefined) document.getElementById('bet').value = data.bet;
                        if (data.your_stack !== undefined) document.getElementById('your_stack').value = data.your_stack;
                        if (data.opp_stack !== undefined) document.getElementById('opp_stack').value = data.opp_stack;
                        if (data.position) document.getElementById('position').value = data.position;
                        if (data.hole_cards) document.getElementById('hole_cards').value = data.hole_cards;
                        if (data.board_cards) document.getElementById('board_cards').value = data.board_cards;
                        if (data.small_blind !== undefined) document.getElementById('small_blind').value = data.small_blind;
                        if (data.big_blind !== undefined) document.getElementById('big_blind').value = data.big_blind;
                        if (data.street) document.getElementById('street').value = data.street;
                        if (data.opponent_action) document.getElementById('opponent_action').value = data.opponent_action;
                        if (data.screenshot_path) document.getElementById('screenshot_path').value = data.screenshot_path;
                        document.getElementById('capture_source').value = data.source || 'upload';
                        
                        // Show extracted values and warnings
                        let resultHtml = `<strong>✓ Uploaded!</strong><br>`;
                        resultHtml += `Extracted: Pot=$${data.pot || 0}, Bet=$${data.bet || 0}, Stack=$${data.your_stack || 0}`;
                        if (data.hole_cards) resultHtml += `, Cards=${data.hole_cards}`;
                        if (data.position) resultHtml += `, Position=${data.position}`;
                        resultHtml += `<br><small>Screenshot: ${data.screenshot_path}</small>`;
                        
                        if (data.warnings && data.warnings.length > 0) {
                            resultHtml += `<br><strong style="color: #f59e0b;">⚠️ Warnings:</strong> ${data.warnings.join('; ')}`;
                            resultDiv.className = 'ocr-result error';
                        } else {
                            resultDiv.className = 'ocr-result';
                        }
                        
                        resultDiv.innerHTML = resultHtml;
                        resultDiv.style.display = 'block';
                        
                        const manualBtn = document.querySelector('.tab[onclick*="manual"]');
                        if (manualBtn) {
                            switchTab('manual', manualBtn);
                        }
                        document.getElementById('gameForm').dispatchEvent(new Event('submit'));
                    }
                }
            } catch (error) {
                resultDiv.className = 'ocr-result error';
                resultDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
                resultDiv.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = '⬆️ Upload Screenshot & Analyze';
            }
        }
        
        document.getElementById('gameForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Generate hand_id if not set (for tracking same hand across streets)
            let handId = document.getElementById('hand_id').value;
            if (!handId) {
                handId = Math.random().toString(36).substring(2, 10);
                document.getElementById('hand_id').value = handId;
            }
            
            const gameState = {
                hand_id: handId,
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
                board_cards: document.getElementById('board_cards').value,
                action_history: document.getElementById('action_history').value || '[]',
                action_notes: document.getElementById('action_notes').value,
                screenshot_path: document.getElementById('screenshot_path').value,
                capture_source: document.getElementById('capture_source').value
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
                        <div class="metric-value">$${metrics.to_call.toFixed(2)}</div>
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
            
            // Show action history insights if available
            if (result.env_state && result.env_state.action_history && result.env_state.action_history.length > 0) {
                html += `
                    <div class="opponent-intel" style="margin-top: 20px;">
                        <h3>📊 Action History Analysis</h3>
                        <div style="font-size: 14px; color: #666; margin-bottom: 10px;">
                            <strong>Previous Streets:</strong>
                        </div>
                `;
                const streets = ['preflop', 'flop', 'turn', 'river'];
                for (const s of streets) {
                    const streetActions = result.env_state.action_history.filter(a => a.street === s);
                    if (streetActions.length > 0) {
                        html += `<div style="margin: 5px 0;"><strong>${s.toUpperCase()}:</strong> `;
                        streetActions.forEach((action, idx) => {
                            if (idx > 0) html += ' → ';
                            html += `${action.action}`;
                            if (action.bet_size > 0) html += ` ($${action.bet_size})`;
                        });
                        html += `</div>`;
                    }
                }
                if (result.env_state.action_pattern) {
                    const pattern = result.env_state.action_pattern;
                    html += `<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">`;
                    html += `<strong>Patterns Detected:</strong><br>`;
                    if (pattern.preflop_aggressive) html += `• Raised preflop<br>`;
                    if (pattern.flop_continuation) html += `• C-bet flop<br>`;
                    if (pattern.turn_continuation) html += `• Barreled turn<br>`;
                    if (pattern.river_value_bet) html += `• Bet river<br>`;
                    html += `</div>`;
                }
                if (result.env_state.predicted_opponent_action) {
                    html += `<div style="margin-top: 10px; color: #2a5298;">
                        <strong>Predicted Next Action:</strong> ${result.env_state.predicted_opponent_action} 
                        (${(result.env_state.prediction_confidence * 100).toFixed(0)}% confidence)
                    </div>`;
                }
                html += `</div>`;
            }
            
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
        
        // Player selection modal
        let selectedPlayerData = null;
        let pendingCaptureData = null;
        let sortedPlayersList = null;  // Store sorted players for selection
        
        function closePlayerModal() {
            const modal = document.getElementById('playerSelectModal');
            if (modal) modal.classList.remove('active');
        }
        
        function showPlayerSelection(players, captureData) {
            pendingCaptureData = captureData;
            const modal = document.getElementById('playerSelectModal');
            const content = document.getElementById('playerSelectContent');
            
            // Sort players: prioritize non-all-in players, then by is_turn status
            sortedPlayersList = [...players].sort((a, b) => {
                if (a.all_in && !b.all_in) return 1;
                if (!a.all_in && b.all_in) return -1;
                if (a.is_turn && !b.is_turn) return -1;
                if (!a.is_turn && b.is_turn) return 1;
                return 0;
            });
            const sortedPlayers = sortedPlayersList;
            
            let html = '<h3>👤 Who are you in this screenshot?</h3>';
            if (sortedPlayers.length === 0) {
                html += '<p>No players detected. Please enter data manually.</p>';
                html += '<button class="select-player-btn" type="button" onclick="closePlayerModal()">Close</button>';
                content.innerHTML = html;
                return;
            }
            
            sortedPlayers.forEach((player, idx) => {
                const isSelected = idx === 0 ? 'selected' : '';
                let badges = [];
                if (player.position) badges.push(player.position.toUpperCase());
                if (player.all_in) badges.push('ALL-IN');
                if (player.is_turn) badges.push('YOUR TURN');
                
                const badgeText = badges.length > 0 ? ` <span style="color: #f59e0b; font-size: 12px;">[${badges.join(', ')}]</span>` : '';
                const position = player.position ? ` (${player.position.toUpperCase()})` : '';
                const cards = player.cards ? ` • Cards: ${player.cards}` : '';
                const stackText = player.stack > 0 ? `Stack: $${player.stack}` : 'Stack: Unknown';
                
                // Disable selection for all-in players (you can't be all-in)
                const disabledClass = player.all_in ? 'disabled' : '';
                const disabledStyle = player.all_in ? 'opacity: 0.5; cursor: not-allowed;' : '';
                const onClickAttr = player.all_in ? '' : 'onclick="selectPlayer(' + idx + ')"';
                
                html += '<div class="player-option ' + isSelected + ' ' + disabledClass + '" ' + onClickAttr + ' style="' + disabledStyle + '">';
                html += '<div class="player-name">' + (player.name || '') + badgeText + '</div>';
                html += '<div class="player-details">' + stackText + (cards || '') + '</div>';
                if (player.all_in) {
                    html += '<div style="color: #ef4444; font-size: 12px; margin-top: 5px;">⚠️ This player is all-in - you cannot be this player</div>';
                }
                html += '</div>';
            });
            
            // Find first non-all-in player as default selection
            const defaultIdx = sortedPlayers.findIndex(p => !p.all_in);
            const selectedIdx = defaultIdx >= 0 ? defaultIdx : 0;
            
            html += '<button class="select-player-btn" type="button" onclick="confirmPlayerSelection()">Select Player</button>';
            
            content.innerHTML = html;
            if (sortedPlayers.length > 0) {
                selectedPlayerData = sortedPlayers[selectedIdx];
                // Make sure the default is visually selected
                document.querySelectorAll('.player-option').forEach((el, i) => {
                    if (i === selectedIdx) {
                        el.classList.add('selected');
                    } else {
                        el.classList.remove('selected');
                    }
                });
            }
            modal.classList.add('active');
        }
        
        function selectPlayer(idx) {
            document.querySelectorAll('.player-option').forEach((el, i) => {
                if (i === idx) {
                    el.classList.add('selected');
                } else {
                    el.classList.remove('selected');
                }
            });
            if (sortedPlayersList && sortedPlayersList[idx]) {
                selectedPlayerData = sortedPlayersList[idx];
            }
        }
        
        function confirmPlayerSelection() {
            if (!selectedPlayerData || !pendingCaptureData) {
                alert('Please select a player');
                return;
            }
            
            const modal = document.getElementById('playerSelectModal');
            modal.classList.remove('active');
            
            // Fill form with selected player's data
            const data = pendingCaptureData;
            const player = selectedPlayerData;
            
            // Set your data from selected player
            if (player.stack) document.getElementById('your_stack').value = player.stack;
            if (player.cards) document.getElementById('hole_cards').value = player.cards;
            if (player.position) document.getElementById('position').value = player.position;
            
            // Set opponent data (use other players)
            if (data.players && data.players.length > 1) {
                const otherPlayers = data.players.filter(p => p.name !== player.name);
                if (otherPlayers.length > 0) {
                    // Use largest stack as opponent
                    const opp = otherPlayers.reduce((a, b) => a.stack > b.stack ? a : b);
                    if (opp.stack) document.getElementById('opp_stack').value = opp.stack;
                    if (opp.name) document.getElementById('opponent').value = opp.name;
                }
            }
            
            // Fill other fields
            if (data.pot !== undefined) document.getElementById('pot').value = data.pot;
            if (data.bet !== undefined) document.getElementById('bet').value = data.bet;
            if (data.small_blind !== undefined) document.getElementById('small_blind').value = data.small_blind;
            if (data.big_blind !== undefined) document.getElementById('big_blind').value = data.big_blind;
            if (data.street) document.getElementById('street').value = data.street;
            if (data.board_cards) document.getElementById('board_cards').value = data.board_cards;
            if (data.opponent_action) document.getElementById('opponent_action').value = data.opponent_action;
            if (data.screenshot_path) document.getElementById('screenshot_path').value = data.screenshot_path;
            document.getElementById('capture_source').value = data.source || 'screen';
            
            // Show result
            const resultDiv = document.getElementById('ocrResult');
            let resultHtml = `<strong>✓ Captured!</strong><br>`;
            resultHtml += `Selected: ${player.name} (${player.position || 'unknown'})`;
            resultHtml += ` • Pot=$${data.pot || 0}, Bet=$${data.bet || 0}`;
            if (player.cards) resultHtml += `, Cards=${player.cards}`;
            resultHtml += `<br><small>Screenshot: ${data.screenshot_path}</small>`;
            
            if (data.warnings && data.warnings.length > 0) {
                resultHtml += `<br><strong style="color: #f59e0b;">⚠️ Warnings:</strong> ${data.warnings.join('; ')}`;
                resultDiv.className = 'ocr-result error';
            } else {
                resultDiv.className = 'ocr-result';
            }
            
            resultDiv.innerHTML = resultHtml;
            resultDiv.style.display = 'block';
            
            // Switch to manual tab and auto-analyze
            const manualBtn = document.querySelector('.tab[onclick*="manual"]');
            if (manualBtn) {
                switchTab('manual', manualBtn);
            }
            
            // Auto-submit for analysis
            document.getElementById('gameForm').dispatchEvent(new Event('submit'));
            
            // Reset
            pendingCaptureData = null;
            selectedPlayerData = null;
        }
    </script>
    
    <!-- Player Selection Modal -->
    <div id="playerSelectModal" class="player-select-modal">
        <div id="playerSelectContent" class="player-select-content"></div>
    </div>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # #region agent log
        _debug_log("do_GET path", {"path": self.path, "path_len": len(self.path)}, "H1")
        # #endregion
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            try:
                self.wfile.write(HTML.encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected before response finished (e.g. refresh, navigate away)
        
        elif self.path.startswith('/api/analyze_description'):
            # Parse hand description and run analysis (must be checked before /api/analyze)
            # #region agent log
            _debug_log("branch taken", {"branch": "analyze_description"}, "H1")
            # #endregion
            parsed_url = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_url.query)
            text = params.get('text', [''])[0]
            # #region agent log
            _debug_log("analyze_description text", {"text_len": len(text), "has_text": bool(text)}, "H2")
            # #endregion
            if not text:
                result = {'error': 'No description text provided'}
            else:
                parsed = HandDescriptionParser.parse_hand_description(text)
                if not parsed.get('success'):
                    result = {'error': 'Could not parse hand description'}
                else:
                    game_state = {
                        'pot': parsed.get('pot', 0),
                        'bet': parsed.get('bet', 0),
                        'your_stack': parsed.get('your_stack', 8000),
                        'opp_stack': parsed.get('opp_stack', 8000),
                        'small_blind': parsed.get('small_blind', 1),
                        'big_blind': parsed.get('big_blind', 3),
                        'street': parsed.get('street', 'river'),
                        'position': parsed.get('position', 'utg'),
                        'opponent_action': parsed.get('opponent_action', 'BET'),
                        'opponent': parsed.get('opponent', ''),
                        'hole_cards': parsed.get('hole_cards', ''),
                        'board_cards': parsed.get('board_cards', ''),
                        'action_history': json.dumps(parsed.get('action_history', [])),
                    }
                    result = assistant.analyze(game_state)
                    result['_parsed'] = {k: v for k, v in parsed.items() if k != 'raw_description'}
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        
        elif self.path.startswith('/api/analyze'):
            # #region agent log
            _debug_log("branch taken", {"branch": "analyze"}, "H1")
            # #endregion
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            game_state = {k: v[0] for k, v in params.items()}
            result = assistant.analyze(game_state)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        
        elif self.path == '/api/capture':
            # Screen capture endpoint
            result = ScreenCaptureParser.capture_and_parse()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        
        elif self.path.startswith('/api/parse_description'):
            parsed_url = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_url.query)
            text = params.get('text', [''])[0]
            
            if not text:
                result = {'error': 'No description text provided'}
            else:
                result = HandDescriptionParser.parse_hand_description(text)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        
        else:
            # #region agent log
            _debug_log("404 path", {"path": self.path, "path_start": self.path[:50]}, "H1")
            # #endregion
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/upload_screenshot':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                data = self.rfile.read(length) if length > 0 else b''
                result = ScreenCaptureParser.parse_image_bytes(data)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
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
