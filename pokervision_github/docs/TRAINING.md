# Training Guide

## Quick Start

### 1. Data Preparation
```bash
# Place PokerNow CSV logs in data/ directory
# Run parser
python scripts/enhanced_complete_parser.py
```

### 2. Training Progression

**Stage 1: Basic Model (10 min)**
```bash
python scripts/train_pytorch.py
# Expected: 63.4% accuracy
```

**Stage 2: Enhanced Features (15 min)**
```bash
python scripts/train_enhanced_model.py
# Expected: 71.6% accuracy (+8%)
```

**Stage 3: Card Information (20 min)**
```bash
python scripts/train_with_cards.py
# Expected: 74.3% accuracy (+3%)
```

**Stage 4: Data Augmentation (30 min)**
```bash
python scripts/augment_data.py
python scripts/train_with_cards.py
# Expected: 74-75% accuracy
```

### 3. Opponent-Specific Models
```bash
python scripts/finetune_opponent.py seb
# Creates models/vs_seb.pt with 80%+ accuracy
```

## Detailed Training

See full documentation for:
- Hyperparameter tuning
- Custom datasets
- Advanced architectures
- Ensemble methods
