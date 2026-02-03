# Data Directory

## Format

Training data should be in JSON format with the following structure:
```json
{
  "metadata": {
    "hero_name": "akumoli",
    "total_hands": 3991,
    "total_examples": 2492
  },
  "training_examples": [
    {
      "hand_num": 1,
      "small_blind": 1.0,
      "big_blind": 2.0,
      "hero_stack_bb": 100.0,
      "stack_to_pot_ratio": 25.5,
      "position": "button",
      "preflop_raised": true,
      "action_type": "raise"
    }
  ]
}
```

## Generating Data

Place PokerNow CSV logs here and run:
```bash
python scripts/enhanced_complete_parser.py
```
