"""Check your current model performance"""
import torch
import os

print("="*70)
print("MODEL PROGRESS TRACKER")
print("="*70)
print()

models = [
    ('models/akumoli_poker_model.json', 'Simple baseline'),
    ('models/pytorch_poker_model.pt', 'Basic PyTorch'),
    ('models/enhanced_model.pt', 'Enhanced features'),
]

for path, name in models:
    if os.path.exists(path):
        if path.endswith('.pt'):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                acc = checkpoint.get('accuracy', 0)
                print(f"✓ {name:20s}: {acc:.1%}")
            except:
                print(f"✗ {name:20s}: Error loading")
        else:
            print(f"✓ {name:20s}: Available")
    else:
        print(f"✗ {name:20s}: Not found")

print("="*70)
