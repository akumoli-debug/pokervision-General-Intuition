# General Intuition Application

## Project Summary

**PokerVision: Exploitative Poker AI via Behavioral Modeling**

Built an exploitative poker agent that learns opponent-specific behavioral patterns from game logs. Using neural models augmented with strategic features (SPR, position, card distributions), the system achieves 74.3% accuracy - significantly outperforming static GTO baselines (55%) against non-optimal opponents.

## Key Results

- **74.3% accuracy** vs 55% GTO (+19% improvement)
- **80-99% accuracy** vs specific opponents with fine-tuning
- **Trained on 10K examples** from 90 unique opponents
- **Real-time inference** (<100ms per decision)

## Technical Approach

### 1. Strategic Feature Engineering
- Stack-to-Pot Ratio (SPR): Critical for commitment decisions
- Position encoding: Asymmetric advantages
- Preflop context: Raised/3-bet/4-bet dynamics
- Card embeddings: Learned 32-dim representations

### 2. Opponent Modeling
- Per-player statistics (aggression, fold frequency)
- Behavioral pattern recognition
- Exploitable tendency identification

### 3. Data Augmentation
- Suit isomorphism: 4x data multiplier
- Position symmetry: 2x multiplier
- Total: 8x theoretical increase

## Alignment with General Intuition

### World Models
- Poker as world model: Predict opponent actions + state evolution
- Spatial reasoning: Position dynamics
- Temporal reasoning: Multi-street strategy

### Behavioral Learning
- Learn opponent policies from observation
- Infer latent tendencies
- Adapt strategies dynamically

### Strategic Reasoning
- Decision-making under uncertainty
- Adversarial optimization
- Meta-reasoning (when to deviate from equilibrium)

## Extensions & Future Work

1. Sequence modeling (LSTM/Transformer over betting history)
2. Real-time screen capture integration
3. Transfer learning from large datasets
4. Multi-player dynamics (9-handed)

## Broader Applications

- Negotiation (adversarial settings)
- Strategy games (opponent prediction)
- Financial markets (counterparty modeling)
- General adversarial ML

---

**Contact:** your.email@example.com
**Demo:** [Live UI](http://your-demo-url.com)
**Repository:** [GitHub](https://github.com/yourusername/pokervision)
