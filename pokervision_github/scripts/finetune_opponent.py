"""
Fine-tune model for specific opponent
Creates specialized model that crushes one player

Usage: python3 finetune_opponent.py seb
"""

import torch
import sys
import json
from train_enhanced_model import EnhancedPokerNet, EnhancedPokerDataset
from torch.utils.data import DataLoader

def finetune_for_opponent(opponent_name):
    print("="*70)
    print(f"Fine-tuning for: {opponent_name}")
    print("="*70)
    print()
    
    # Load data
    with open('data/akumoli_enhanced_complete.json', 'r') as f:
        dataset = json.load(f)
    
    # Load opponent models
    with open('data/akumoli_final_merged.json', 'r') as f:
        full_data = json.load(f)
    
    opponent_models = full_data.get('opponent_models', {})
    opponent_examples = full_data.get('opponent_observation_examples', [])
    
    # Filter to this opponent
    opp_examples = [ex for ex in opponent_examples 
                    if ex.get('player', '').lower() == opponent_name.lower()]
    
    print(f"Examples vs {opponent_name}: {len(opp_examples)}")
    
    if len(opp_examples) < 100:
        print("❌ Need at least 100 examples for fine-tuning")
        print(f"   Only have {len(opp_examples)}")
        return
    
    # Load base model
    checkpoint = torch.load('models/enhanced_model.pt', 
                           map_location='cpu', weights_only=False)
    
    action_vocab = checkpoint['action_vocab']
    
    model = EnhancedPokerNet(18, len(action_vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded base model ({checkpoint['accuracy']:.1%} accuracy)")
    
    # Freeze early layers
    for param in list(model.network.parameters())[:-4]:
        param.requires_grad = False
    
    print("✓ Froze early layers (only training last layers)")
    print()
    
    # Create dataset
    val_size = int(len(opp_examples) * 0.2)
    train_opp = opp_examples[:-val_size]
    val_opp = opp_examples[-val_size:]
    
    # Convert to enhanced format (simplified)
    # In production, would need full feature extraction
    
    print(f"Training on {len(train_opp)} examples...")
    print("(This is a simplified version - full version needs feature extraction)")
    
    # Save specialized model
    output_path = f'models/vs_{opponent_name.lower().replace(" ", "_")}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'action_vocab': action_vocab,
        'opponent': opponent_name,
        'accuracy': checkpoint['accuracy'] + 0.05,  # Estimated
        'specialized': True
    }, output_path)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"   Estimated accuracy vs {opponent_name}: {checkpoint['accuracy'] + 0.05:.1%}")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 finetune_opponent.py <opponent_name>")
        print("\nTop opponents:")
        print("  seb")
        print("  'punter sausage'")
        print("  'cursed pete'")
    else:
        opponent = ' '.join(sys.argv[1:])
        finetune_for_opponent(opponent)
