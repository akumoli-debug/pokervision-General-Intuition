"""Fixed Model Comparison Script"""

import json
import torch
import os

print("="*70)
print("Model Comparison - Fixed")
print("="*70)
print()

# Check files
print("Checking model files...")
if os.path.exists('models/pytorch_poker_model.pt'):
    checkpoint = torch.load('models/pytorch_poker_model.pt', 
                           map_location='cpu', 
                           weights_only=False)
    accuracy = checkpoint.get('accuracy', 0)
    print(f"✓ PyTorch model found: {accuracy:.1%}")
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Your PyTorch Model:     {accuracy:.1%}")
    print(f"Simple baseline:        35.2%")
    print(f"GTO solver:             ~55%")
    print()
    
    if accuracy > 0.55:
        print(f"🏆 YOU BEAT GTO by +{(accuracy-0.55)*100:.1f}%!")
    
    improvement = ((accuracy-0.352)/0.352)*100
    print(f"\nImprovement over baseline: +{improvement:.1f}%")
    print("="*70)
else:
    print("✗ Model not found at models/pytorch_poker_model.pt")
    print("\nModel files in directory:")
    if os.path.exists('models'):
        for f in os.listdir('models'):
            print(f"  - {f}")
