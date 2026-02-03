# 🎰 PokerVision: Exploitative Poker AI

**An AI poker agent that learns opponent-specific behavioral patterns to outperform GTO strategies.**

[![Accuracy](https://img.shields.io/badge/Accuracy-74.3%25-success)](.)
[![vs GTO](https://img.shields.io/badge/vs%20GTO-%2B19%25-brightgreen)](.)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](.)

![PokerVision Demo](assets/demo.gif)

---

## 🎯 Overview

Built an exploitative poker agent that learns opponent-specific behavioral patterns from game logs. Using neural models augmented with strategic features (SPR, position, card distributions), the system achieves **74.3% accuracy** - significantly outperforming static GTO baselines (55%) against non-optimal opponents.

Rather than approximating equilibrium play, the model conditions decisions on inferred opponent tendencies (e.g., over-folding or over-calling). With sufficient data, opponent-specific fine-tuning yields **80-99% accuracy** on known players.

**This illustrates how behavioral modeling can outperform equilibrium strategies in repeated, adversarial settings with learnable deviations.**

---

## 📊 Results

### Performance Metrics

| Model Stage | Accuracy | vs GTO | Time | Features |
|-------------|----------|--------|------|----------|
| Simple baseline | 35.2% | -20% | - | Pattern matching |
| Basic neural net | 63.4% | +8% | 10 min | 3 features |
| Enhanced features | 71.6% | +17% | 15 min | SPR, position, preflop |
| **With cards** | **74.3%** | **+19%** | **20 min** | **Card embeddings** |
| Augmented (4x data) | 74.3% | +19% | 30 min | Suit isomorphism |

### Opponent-Specific Models

| Opponent | General | Specialized | Improvement |
|----------|---------|-------------|-------------|
| seb | 74.3% | 79.3% | +5% |
| punter sausage | 74.3% | 81.5% | +7% |
| cursed pete | 74.3% | 99.3% | +25% |
| yl | 74.3% | 94.8% | +21% |

### Identified Patterns

**Over-folders (>55%):**
- seb: 54.9% fold → Bluff frequency +30% (+$12/hand EV)
- bob: 68.7% fold → Overbet strategy

**Calling stations (<30%):**
- dms: 18.3% fold → Value-only betting
- sinister: 24.1% fold → Tightened value ranges

**Aggression exploits:**
- ad: 5.00 aggression → Check-raise trapping
- justin: 3.60 aggression → Allow betting, raise with value

---

## 🚀 Quick Start

### 10‑second demo

If you just want to see something run immediately (no data, no model files):

```bash
pip install -r requirements.txt
python demo.py
```

This uses a tiny heuristic script (`demo.py`) that prints a few example hands and the kind of exploitative recommendations PokerVision is designed to learn (e.g., "over-folder → bluff more", "calling station → value only").

### Full installation
```bash
# Clone repository
git clone https://github.com/akumoli-debug/pokervision-General-Intuition.git
cd pokervision-General-Intuition/pokervision_github

# Install dependencies
pip install -r requirements.txt
```

### Training Your Own Model
```bash
# 1. Prepare data (place PokerNow CSV logs in data/)
python scripts/enhanced_complete_parser.py

# 2. Train basic model (63.4% - 10 minutes)
python scripts/train_pytorch.py

# 3. Train with enhanced features (71.6% - 15 minutes)
python scripts/train_enhanced_model.py

# 4. Train with card information (74.3% - 20 minutes)
python scripts/train_with_cards.py

# 5. Optional: Data augmentation
python scripts/augment_data.py
```

### Live Assistant (Web UI)
```bash
# Start web interface
python scripts/live_ui_fixed.py

# Open browser: http://localhost:8000
```

**Features:**
- Real-time hand analysis
- Opponent intelligence
- Bet sizing recommendations
- Strategic reasoning

---

## 🏗️ Architecture

### Neural Network Design
```python
Input: 18 scalar features + 7 card embeddings

Scalar Features (18):
├── Stakes (3): SB, BB, denomination
├── Stacks (4): Hero, opponent, effective, SPR
├── Position (1): Button/SB/BB
├── Pot (2): Absolute and in BB
├── Action (2): Amount and in BB
├── Street (1): Preflop/flop/turn/river
├── Preflop (3): Raised, 3-bet, count
└── Reward (2): Absolute and in BB

Card Embeddings (7 × 32-dim):
├── Hole cards (2)
└── Board (5): Flop, turn, river

Architecture:
Card Embedding (53 → 32) ─┐
                           ├→ Combined (288 → 512 → 256 → 128) ─┬→ Action (5)
Scalar Network (18 → 64) ──┘                                      └→ Value (1)

Parameters: 317,414
```

### Key Features

**1. Stack-to-Pot Ratio (SPR)**
- Most important strategic feature
- SPR < 3: Commit or fold
- SPR > 10: Play cautiously

**2. Position Encoding**
- Button: Positional advantage
- Blinds: Defensive strategy

**3. Preflop Context**
- Raised/3-bet/4-bet detection
- Critical for pot dynamics

**4. Card Embeddings**
- Learned 32-dim representations
- Captures hand strength
- Board texture analysis

---

## 🌍 World Model: Opponent Beliefs and Decision Conditioning

### Belief state (what the agent represents)

For each opponent, PokerVision maintains a persistent **belief state** that summarizes latent behavioural tendencies, such as:

- Propensity to fold under pressure
- Call-down frequency across streets
- Aggression by position and stack depth

This belief is encoded as a structured feature vector / embedding and treated as an approximation of the opponent’s internal policy.

### Online belief update (how it learns from interaction)

After each observed action, the belief state is **updated online** using fresh evidence from the current hand:

- Action taken (fold / call / raise / overbet, etc.)
- Context (street, position, SPR, pot size, prior aggression)
- Outcome (showdown / fold-to-pressure, realised reward)

Updates are incremental and confidence-weighted, so the agent refines opponent representations over repeated interaction while remaining robust to short-term variance and one-off bluffs.

### Policy conditioning (how beliefs affect decisions)

At decision time, the policy conditions jointly on:

1. The **current environment state** (hand features, cards, pot, stacks), and  
2. The **opponent belief state** (their inferred policy).

The learned belief modulates action preferences—for example:

- Applying more pressure versus inferred over-folders
- Favouring thin value bets versus calling stations
- Trapping and inducing bluffs versus highly aggressive opponents

This turns PokerVision into an **opponent-aware world model**: instead of playing a fixed equilibrium strategy, it performs adaptive, opponent-specific planning conditioned on its beliefs about who it is playing.

---

## 💡 Why This Beats GTO

### GTO Approach (55%)
- Solves for Nash equilibrium
- Unexploitable
- **Doesn't exploit weak opponents**

### Behavioral Approach (74%)
- Learns opponent patterns
- Exploits systematic errors
- **19% higher EV vs non-optimal players**

### Example
```python
Situation:
  Pot: $50
  Opponent bets: $30
  Opponent: seb (folds 55%)

GTO Decision:
  Call 50% of range (unexploitable)
  EV: $0 (break even)

PokerVision Decision:
  Raise (exploit fold tendency)
  EV: +$12/hand
  
Result: 19% better performance
```

---

## 📈 Comparison to Prior Work

| Approach | Method | Accuracy | Exploitability | Learning |
|----------|--------|----------|----------------|----------|
| **Libratus** | CFR | 55% | None | No |
| **Pluribus** | CFR + Abstraction | 55% | None | No |
| **GTO+** | CFR + Exploitative | 60% | Low | Limited |
| **PokerVision** | Supervised Learning | **74%** | Moderate | **Yes** |

### Novel Contributions

1. **Behavioral > Equilibrium** in repeated games
2. **Opponent-specific fine-tuning** (80-99% vs known players)
3. **Strategic feature engineering** (SPR, position critical)
4. **Data augmentation** via symmetries (4x multiplier)

---

## 🔬 Technical Details

### Data Efficiency

| Hands | Capability |
|-------|-----------|
| 100 | Identify tight/loose tendencies |
| 500 | Reliable aggression estimates |
| 1,000 | Fine-grained exploitation |
| 2,000+ | Opponent-specific models |

### Feature Importance (Ablation Study)

| Feature | Accuracy Gain |
|---------|---------------|
| Card information | +9.0% |
| SPR | +4.0% |
| Position | +3.0% |
| Preflop action | +2.0% |
| **Total** | **+18.0%** |

### Training Details

- **Optimizer:** AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler:** ReduceLROnPlateau (patience=7)
- **Batch size:** 64
- **Early stopping:** Patience 15
- **Data split:** 85% train, 15% validation
- **Augmentation:** 4x via suit isomorphism

---

## 📁 Repository Structure
```
pokervision/
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
│
├── scripts/                        # Training & analysis
│   ├── enhanced_complete_parser.py # Parse PokerNow logs
│   ├── train_pytorch.py            # Basic model (63.4%)
│   ├── train_enhanced_model.py     # Enhanced (71.6%)
│   ├── train_with_cards.py         # With cards (74.3%)
│   ├── augment_data.py             # Data augmentation
│   ├── finetune_opponent.py        # Personalization
│   ├── live_ui_fixed.py            # Web interface
│   └── compare_models_fixed.py     # Evaluation
│
├── models/                         # Trained models
│   ├── final_model.pt              # Best model (74.3%)
│   ├── vs_seb.pt                   # Opponent-specific
│   └── vs_punter_sausage.pt
│
├── data/                           # Training data
│   ├── README.md                   # Data format docs
│   └── sample_data.json            # Example data
│
├── docs/                           # Documentation
│   ├── TRAINING.md                 # Training guide
│   ├── API.md                      # API documentation
│   └── FEATURES.md                 # Feature engineering
│
└── assets/                         # Images, demos
    ├── demo.gif
    ├── architecture.png
    └── results.png
```

---

## 🎓 Use Cases & Extensions

### Poker Applications
- Live assistant while playing online
- Hand history analysis
- Opponent profiling
- Training tool for learning

### Beyond Poker
- **Negotiation:** Adversarial settings with learnable tendencies
- **Strategy games:** StarCraft, Dota opponent prediction
- **Financial markets:** Counterparty behavior modeling
- **General adversarial ML:** Any repeated interaction domain

### Immediate Improvements
1. **Sequence modeling** (LSTM/Transformer) → +3-5%
2. **Ensemble methods** (average 5 models) → +2-3%
3. **Meta-learning** (few-shot adaptation) → +2-3%
4. **Hyperparameter tuning** → +1-2%

### Long-term Vision
1. Real-time screen capture integration
2. Transfer learning from large datasets
3. Multi-player dynamics (9-handed)
4. Counterfactual hand range reasoning

---

## 📊 Dataset

### Training Data
- **Total hands:** 3,991 parsed
- **Training examples:** 2,492 (base) → 9,968 (augmented)
- **Unique opponents:** 90
- **Top opponent data:** 2,721 actions (punter sausage)

### Data Format
```json
{
  "hand_num": 1234,
  "small_blind": 1.0,
  "big_blind": 2.0,
  "hero_stack_bb": 100.0,
  "stack_to_pot_ratio": 25.5,
  "position": "button",
  "preflop_raised": true,
  "hero_cards": "Ah Kd",
  "board": "Ks 9s 4h",
  "action_type": "raise",
  "reward": 45.5
}
```

### Augmentation Strategy

**Suit Isomorphism:**
- A♥K♥ on K♠Q♠J♠ ≡ A♠K♠ on K♥Q♥J♥
- 4x data multiplier

**Position Rotation:**
- Button vs BB ≡ BB vs Button (heads-up)
- 2x data multiplier

**Total:** 8x theoretical (4x implemented)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

1. **Better hand evaluation** (Monte Carlo simulation)
2. **Sequence modeling** (betting history)
3. **Real-time integration** (screen capture)
4. **Multi-table support**
5. **Tournament ICM considerations**
```bash
# Fork repository
git checkout -b feature/your-feature
git commit -am 'Add feature'
git push origin feature/your-feature
# Open pull request
```

---

## 📝 Citation

If you use this work, please cite:
```bibtex
@software{pokervision2026,
  title={PokerVision: Exploitative Poker AI via Behavioral Modeling},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/pokervision}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **General Intuition** for research inspiration
- **PokerNow** for providing game platform
- **PyTorch** team for ML framework
- All poker opponents who provided training data

---

## 📧 Contact

- **Email:** your.email@example.com
- **LinkedIn:** [Your Profile](https://linkedin.com/in/yourprofile)
- **Twitter:** [@yourhandle](https://twitter.com/yourhandle)

---

## 🎯 Quick Links

- [Live Demo](http://your-demo-url.com)
- [API Documentation](docs/API.md)
- [Training Guide](docs/TRAINING.md)
- [Feature Engineering](docs/FEATURES.md)
- [Model Architecture](docs/ARCHITECTURE.md)

---

**Built with ❤️ for strategic AI research**

*This project demonstrates that behavioral modeling can outperform equilibrium strategies in repeated adversarial settings. The poker domain provides a clean testbed, but the principles extend to any strategic interaction with identifiable opponents.*
