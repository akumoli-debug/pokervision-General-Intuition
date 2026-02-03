"""Train card-aware model on 4x augmented data"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_with_cards import CardAwareDataset, CardAwarePokerNet

print("="*70)
print("Training with 4x AUGMENTED Data + Cards")
print("="*70)
print()

# Load augmented data
with open('data/akumoli_augmented.json', 'r') as f:
    dataset = json.load(f)

examples = dataset['training_examples']
val_indices = set(dataset.get('validation_indices', []))

train_ex = [ex for i, ex in enumerate(examples) if i not in val_indices]
val_ex = [examples[i] for i in val_indices if i < len(examples)]

print(f"✓ Train: {len(train_ex)}, Val: {len(val_ex)}")
print(f"  (4x more than before!)")

# Action vocab
action_vocab = {}
for ex in examples:
    if ex['action_type'] not in action_vocab:
        action_vocab[ex['action_type']] = len(action_vocab)

print(f"Actions: {list(action_vocab.keys())}\n")

# Datasets
train_data = CardAwareDataset(train_ex, action_vocab)
val_data = CardAwareDataset(val_ex, action_vocab)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CardAwarePokerNet(len(action_vocab)).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

print("Training...")
print("="*70)

best_acc = 0
patience_counter = 0

for epoch in range(100):
    # Train
    model.train()
    correct, total = 0, 0
    
    for batch in train_loader:
        scalar = batch['scalar_features'].to(device)
        cards = batch['card_indices'].to(device)
        actions = batch['action'].to(device).squeeze()
        rewards = batch['reward'].to(device).squeeze()
        
        action_logits, values = model(scalar, cards)
        
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
            scalar = batch['scalar_features'].to(device)
            cards = batch['card_indices'].to(device)
            actions = batch['action'].to(device).squeeze()
            
            action_logits, _ = model(scalar, cards)
            correct += (action_logits.argmax(1) == actions).sum().item()
            total += actions.size(0)
    
    val_acc = correct / total
    scheduler.step(1 - val_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/100 | Train: {train_acc:.1%} | Val: {val_acc:.1%}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'action_vocab': action_vocab,
            'accuracy': val_acc,
            'augmented': True
        }, 'models/final_model.pt')
    else:
        patience_counter += 1
    
    if patience_counter >= 15:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Best accuracy: {best_acc:.1%}")
print()
print("Progression:")
print(f"  Basic model:          63.4%")
print(f"  Enhanced features:    71.6%")
print(f"  With cards:           75.1%")
print(f"  Augmented + cards:    {best_acc:.1%}")
print()
print("✓ Saved to: models/final_model.pt")
print("="*70)
