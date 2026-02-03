PokerVision: Exploitative Poker AI
==================================

An AI poker assistant that learns opponent-specific behaviour from hand histories and recommends exploitative actions in real time.  
Built as a lightweight research project to demonstrate how behavioural modelling can outperform static “play-perfect” solvers against real, non‑optimal opponents.

Demo
----

The repo includes a simple web UI:

- Run the live assistant (see Quickstart below), then open `http://localhost:8000`.
- A demo GIF (`assets/demo.gif`) in the project shows the assistant analysing hands and suggesting actions.

Quickstart
----------

```bash
# 1. Clone and enter the repo
git clone https://github.com/akumoli-debug/pokervision-General-Intuition.git
cd pokervision-General-Intuition/pokervision_github

# 2. Set up the environment
./setup.sh           # installs Python dependencies via pip

# 3. (Optional) Prepare your own data
#    Put PokerNow CSV logs into data/ and run:
python scripts/enhanced_complete_parser.py

# 4. Train the main model (card-aware, ~74% accuracy in our runs)
python scripts/train_with_cards.py

# 5. Launch the live assistant UI
python scripts/live_ui_fixed.py
# Then open http://localhost:8000 in your browser
```

How it Works
------------

Conceptually, each decision goes through a three-step loop:

1. **Per-hand inference**
   - Input: current hand state (stacks, pot, position, street, actions, cards) + opponent features.
   - A neural network with scalar features and card embeddings predicts action logits and value.
2. **Opponent memory update**
   - After each observed action, opponent statistics (fold frequency, aggression, etc.) are updated.
   - These summary stats become features for future decisions against the same player.
3. **Action selection**
   - The assistant converts logits into a discrete recommendation (fold / call / small bet / big bet / shove).
   - Explanations are based on SPR, position and opponent tendencies (e.g., “over-folds vs pressure”).

In short:

```text
Hand state + opponent stats
          └──> Neural model (features + cards)
                      └──> Action logits + value
                                └──> Recommended action + updated opponent memory
```

Results (Snapshot)
------------------

Offline accuracy on held-out data from PokerNow logs:

| Model                  | Accuracy | Notes                          |
|------------------------|----------|--------------------------------|
| GTO-style baseline     | ~55%     | Unexploitable, not personalised |
| Basic neural net       | ~63%     | Limited strategic features      |
| Enhanced + cards       | ~74%     | Uses SPR, position, cards      |
| Opponent fine-tuning   | 80–99%   | Per-opponent models on enough data |

These numbers are approximate and depend on the exact dataset split and hyperparameters; see the scripts in `scripts/` for the full training pipeline.

Reproducibility
---------------

- **Data**: Results were obtained on ~4K PokerNow hands, augmented via suit symmetry.  
  To reproduce, you will need similar hand histories with comparable stakes and formats.
- **Scripts**: All training scripts live in `scripts/` and can be run end‑to‑end with the commands in `docs/TRAINING.md`.  
- **Randomness**: Training is stochastic (PyTorch initialisation, shuffling, etc.), so expect small variations around the reported accuracies.  
  For more deterministic runs, fix seeds in Python / NumPy / PyTorch and keep hardware and library versions constant.
- **“Beats GTO” definition**: Here, “beats GTO” means the exploitative model achieves higher offline action‑matching accuracy and higher expected value against recorded opponents than a static GTO‑style policy, when evaluated on the same held‑out hands.

Limitations
-----------

- **Not a solver**: This is not a full-game equilibrium solver; it focuses on pattern recognition from historical data.
- **Data requirements**: Performance depends heavily on having enough clean hand histories per opponent; sparse data can lead to noisy estimates.
- **Domain assumptions**: The current feature set and parsing logic target PokerNow-style No-Limit Hold’em logs; other formats may require custom parsing.
- **Evaluation gap**: Offline accuracy and EV on historical data are only proxies for live win‑rate; real‑world performance will depend on table dynamics and opponent adaptation.
- **Ethical use**: This code is for research and educational purposes; many poker sites restrict or forbid real‑time assistance tools—check and follow the rules of any platform you use.

