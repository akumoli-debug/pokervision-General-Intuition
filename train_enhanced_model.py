"""Train with Enhanced Features (Blinds, SPR, Preflop action, etc.)"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

class EnhancedPokerDataset(Dataset):
    def __init__(self, examples, action_vocab):
        self.examples = examples
        self.action_vocab = action_vocab
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # 18 enhanced features
        features = torch.FloatTensor([
            ex['small_blind'] / 10,
            ex['big_blind'] / 10,
            ex['bb_denomination'] / 10,
            ex['hero_stack_bb'] / 100,
            ex['opponent_stack'] / 100,
            ex['effective_stack_bb'] / 100,
            ex['stack_to_pot_ratio'] / 100,  # SPR!
            float(ex['position_encoded']),
            ex['pot'] / 100,
            ex['pot_bb'] / 10,
            ex['action_amount'] / 100,
            ex['action_amount_bb'] / 10 if ex['action_amount_bb'] else 0,
            float(ex['street_encoded']) / 3,
            1.0 if ex['preflop_raised'] else 0.0,
            1.0 if ex['preflop_reraise'] else 0.0,
            float(ex['preflop_summary']['num_raises']) / 4,
            ex['reward'] / 100,
            ex['reward_bb'] / 10
        ])
        
        action_idx = self.action_vocab.get(ex['action_type'], 0)
        
        return {
            'features': features,
            'action': torch.LongTensor([action_idx]),
            'reward': torch.FloatTensor([ex['reward']])
        }

class EnhancedPokerNet(nn.Module):
    def __init__(self, input_dim=18, num_actions=5):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.action_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        features = self.network(x)
        return self.action_head(features), self.value_head(features)

print("="*70)
print("Training with Enhanced Features")
print("Includes: Blinds, SPR, Preflop Action, Position")
print("="*70)
print()

# Load
with open('data/akumoli_enhanced_complete.json', 'r') as f:
    dataset = json.load(f)

examples = dataset['training_examples']
val_indices = set(dataset.get('validation_indices', []))

train_ex = [ex for i, ex in enumerate(examples) if i not in val_indices]
val_ex = [examples[i] for i in val_indices if i < len(examples)]

print(f"✓ Train: {len(train_ex)}, Val: {len(val_ex)}")

# Action vocab
action_vocab = {}
for ex in examples:
    if ex['action_type'] not in action_vocab:
        action_vocab[ex['action_type']] = len(action_vocab)

print(f"Actions: {list(action_vocab.keys())}\n")

# Datasets
train_data = EnhancedPokerDataset(train_ex, action_vocab)
val_data = EnhancedPokerDataset(val_ex, action_vocab)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EnhancedPokerNet(18, len(action_vocab)).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

print("Training...")
print("="*70)

best_acc = 0
patience_counter = 0

for epoch in range(50):
    # Train
    model.train()
    correct, total = 0, 0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        actions = batch['action'].to(device).squeeze()
        rewards = batch['reward'].to(device).squeeze()
        
        action_logits, values = model(features)
        
        loss = nn.functional.cross_entropy(action_logits, actions)
        loss += 0.5 * nn.functional.mse_loss(values.squeeze(), rewards / 100)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        correct += (action_logits.argmax(1) == actions).sum().item()
        total += actions.size(0)
    
    train_acc = correct / total
    
    # Validate
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            actions = batch['action'].to(device).squeeze()
            
            action_logits, _ = model(features)
            correct += (action_logits.argmax(1) == actions).sum().item()
            total += actions.size(0)
    
    val_acc = correct / total
    scheduler.step(1 - val_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/50 | Train: {train_acc:.1%} | Val: {val_acc:.1%}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'action_vocab': action_vocab,
            'accuracy': val_acc
        }, 'models/enhanced_model.pt')
    else:
        patience_counter += 1
    
    if patience_counter >= 10:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Best accuracy: {best_acc:.1%}")
print(f"Previous (basic): 63.4%")
print(f"Improvement: +{(best_acc - 0.634)*100:.1f}%")
print()
print("New features used:")
print("  ✓ Blinds/Stakes")
print("  ✓ Stack-to-Pot Ratio (SPR)")
print("  ✓ Position")
print("  ✓ Preflop action")
print()
print("✓ Saved to: models/enhanced_model.pt")
print("="*70)
