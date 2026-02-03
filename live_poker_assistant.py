"""
Live Poker Assistant
Real-time recommendations while you play

Usage: python3 live_poker_assistant.py
"""

import torch
import json
import sys

class LivePokerAssistant:
    def __init__(self):
        print("="*70)
        print("Live Poker Assistant - Loading...")
        print("="*70)
        
        # Load opponent models
        try:
            with open('data/akumoli_final_merged.json', 'r') as f:
                dataset = json.load(f)
            self.opponent_models = dataset.get('opponent_models', {})
            print(f"✓ Loaded {len(self.opponent_models)} opponent models")
        except:
            self.opponent_models = {}
            print("⚠ No opponent data")
        
        # Load trained model
        try:
            checkpoint = torch.load('models/pytorch_poker_model.pt', 
                                   map_location='cpu', weights_only=False)
            self.action_vocab = checkpoint['action_vocab']
            self.action_names = {v: k for k, v in self.action_vocab.items()}
            accuracy = checkpoint.get('accuracy', 0)
            print(f"✓ Model loaded (Accuracy: {accuracy:.1%})")
            print()
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    
    def get_opponent_info(self, opponent_name):
        opponent_name = opponent_name.lower().strip()
        if opponent_name in self.opponent_models:
            opp = self.opponent_models[opponent_name]
            return {
                'aggression': opp.get('aggression_factor', 1.0),
                'fold_freq': opp.get('fold_frequency', 0.5),
                'actions': opp.get('total_actions', 0),
                'known': True
            }
        return {'aggression': 1.0, 'fold_freq': 0.5, 'actions': 0, 'known': False}
    
    def predict(self, pot, current_bet, opponent_name=None):
        # Simple pattern-based recommendation
        pot_odds = current_bet / (pot + current_bet) if current_bet > 0 else 0
        
        # Default recommendation
        if current_bet == 0:
            recommended = 'check'
            confidence = 0.6
        elif pot_odds < 0.33:
            recommended = 'call'
            confidence = 0.65
        else:
            recommended = 'fold'
            confidence = 0.55
        
        # Get opponent info
        opponent_info = None
        if opponent_name:
            opponent_info = self.get_opponent_info(opponent_name)
            
            # Adjust based on opponent
            if opponent_info['known']:
                if opponent_info['fold_freq'] > 0.6:
                    recommended = 'raise' if current_bet == 0 else 'raise'
                    confidence = 0.7
                elif opponent_info['fold_freq'] < 0.3:
                    if current_bet > pot * 0.5:
                        recommended = 'fold'
        
        return {
            'recommended_action': recommended,
            'confidence': confidence,
            'pot_odds': pot_odds,
            'opponent_info': opponent_info
        }
    
    def format_recommendation(self, result):
        print("\n" + "="*70)
        print("AI RECOMMENDATION")
        print("="*70)
        print()
        
        action = result['recommended_action'].upper()
        confidence = result['confidence']
        
        print(f"🎯 RECOMMENDED: {action}")
        print(f"   Confidence: {confidence:.1%}")
        print()
        
        if result['pot_odds'] > 0:
            print(f"   Pot odds: {result['pot_odds']:.1%}")
        
        if result['opponent_info'] and result['opponent_info']['known']:
            opp = result['opponent_info']
            print()
            print("👤 OPPONENT INTEL:")
            print(f"   Actions: {opp['actions']}")
            print(f"   Aggression: {opp['aggression']:.2f}")
            print(f"   Folds: {opp['fold_freq']:.1%}")
            
            if opp['fold_freq'] > 0.6:
                print("   💡 BET/BLUFF (they fold often)")
            elif opp['fold_freq'] < 0.3:
                print("   💡 VALUE BET only (calling station)")
        
        print("="*70)
    
    def interactive_mode(self):
        print("="*70)
        print("Interactive Mode")
        print("="*70)
        print()
        print("Enter game state for live recommendations")
        print("Type 'quit' to exit")
        print()
        
        while True:
            print("-"*70)
            
            pot_input = input("Pot size ($): ").strip()
            if pot_input.lower() == 'quit':
                break
            
            try:
                pot = float(pot_input)
            except:
                print("Invalid. Enter a number.")
                continue
            
            bet_input = input("Bet to you ($, 0 if none): ").strip()
            try:
                current_bet = float(bet_input)
            except:
                current_bet = 0
            
            opponent = input("Opponent (optional): ").strip()
            if not opponent:
                opponent = None
            
            result = self.predict(pot, current_bet, opponent)
            self.format_recommendation(result)
            print()
        
        print("\nGood luck! 🎰")

def main():
    assistant = LivePokerAssistant()
    assistant.interactive_mode()

if __name__ == "__main__":
    main()
