# Quick Start Guide

## 5-Minute Setup
```bash
# 1. Clone & install
git clone https://github.com/yourusername/pokervision.git
cd pokervision
./setup.sh

# 2. Train model (uses included sample data)
python scripts/train_with_cards.py

# 3. Start live assistant
python scripts/live_ui_fixed.py

# 4. Open browser
# → http://localhost:8000
```

## Using Your Own Data
```bash
# 1. Export PokerNow logs (CSV format)
# 2. Place in data/ directory
# 3. Parse
python scripts/enhanced_complete_parser.py

# 4. Train
python scripts/train_with_cards.py
```

That's it! 
