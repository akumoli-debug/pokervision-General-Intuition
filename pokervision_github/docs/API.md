# API Documentation

## Live Assistant
```python
from scripts.live_ui_fixed import PokerAssistant

assistant = PokerAssistant()

result = assistant.analyze({
    'pot': 50,
    'bet': 30,
    'your_stack': 200,
    'opp_stack': 180,
    'big_blind': 2,
    'position': 'button',
    'opponent': 'seb',
    'hole_cards': 'Ah Kd',
    'board_cards': 'Ks 9s 4h'
})

print(result['action'])  # 'RAISE'
print(result['reasoning'])  # Strategic explanation
```

## Model Interface
```python
import torch

# Load model
checkpoint = torch.load('models/final_model.pt')
model = CardAwarePokerNet()
model.load_state_dict(checkpoint['model_state_dict'])

# Make prediction
action_logits, value = model(scalar_features, card_indices)
```
