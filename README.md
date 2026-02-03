Core idea:
A stateful agent that builds persistent internal models of other agents
from interaction, and conditions its decisions on those learned beliefs.
Poker is used as a controlled testbed for repeated, adversarial interaction.

PokerVision: Exploitative Poker AI
==================================
PokerVision is a stateful agent operating in a multi-agent environment that learns persistent, opponent-specific behavioral models from interaction logs. Rather than optimizing for equilibrium play, it infers how other agents deviate from idealized assumptions and conditions its decisions on those learned internal models. Poker is used as a controlled testbed for studying behavioral world modeling under uncertainty.

A stateful agent that builds internal models of other agents and conditions decisions on them.
Built as a lightweight research project to demonstrate how behavioural modelling can outperform static “play-perfect” solvers against real, non‑optimal opponents.

Demo
----

The core project lives in the `pokervision_github/` folder and includes a simple web UI:

- Run the live assistant (see Quickstart below), then open `http://localhost:8000`.
- A demo GIF (`pokervision_github/assets/demo.gif`) in the project shows the assistant analysing hands and suggesting actions.

Architecture
------------

![PokerVision architecture](pokervision_github/assets/architecture.png)

High-level flow:

- **Environment state** (hand, pot, position, cards)  
  ↓  
- **Opponent belief state** (persistent memory over tendencies)  
  ↓  
- **Policy network** (conditions on state + beliefs)  
  ↓  
- **Action recommendation** (bet / call / fold, with explanation)  
  ↓  
- **Observed opponent action** → **belief update loop** back into the opponent state

Why Poker?
----------

Poker provides a minimal, well-defined environment for studying multi-agent interaction under uncertainty. Agents have hidden state, partial observability, repeated interaction, and incentives to exploit systematic deviations—making it a useful testbed for behavioural world modeling without complex physics or perception.

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

Key Entry Points
----------------

Inside `pokervision_github/` the most useful scripts and docs are:

- **Training**
  - `scripts/train_pytorch.py` – basic model training.
  - `scripts/train_enhanced_model.py` – adds richer strategic features.
  - `scripts/train_with_cards.py` – full card-aware model used for the main results.
  - `scripts/augment_data.py` – data augmentation via suit/position symmetry.
  - `scripts/finetune_opponent.py` – opponent-specific fine-tuning.

- **Evaluation & analysis**
  - `scripts/compare_models_fixed.py` – compare different checkpoints.
  - `docs/TRAINING.md` – full training instructions and tips.

- **Live play / demo**
  - `scripts/live_ui_fixed.py` – launches the browser-based assistant.
  - `docs/API.md` – programmatic API usage examples.

- **Context for General Intuition**
  - `docs/GI_APPLICATION.md` – one-pager framing this project for General Intuition.

How it Works
------------

### World model: opponent beliefs and conditioning

PokerVision maintains an internal **belief state** over other agents and updates it through interaction.

1. **Belief state (what is represented)**  
   For each opponent, the agent stores a persistent summary of latent behavioural tendencies (propensity to fold under pressure, call-down frequency, aggression by position and stack depth). This is encoded as a feature vector / embedding that serves as an approximation of the opponent’s policy.

2. **Online belief update (how it learns)**  
   After each observed action, the belief state is updated online using new evidence from the hand (action taken, context, outcome). Updates are incremental and confidence-weighted, so representations refine over repeated interaction while remaining robust to short-term variance.

3. **Policy conditioning (how it decides)**  
   At decision time, the policy conditions jointly on the current environment state and the opponent belief state. The belief modulates action preferences—for example, applying more pressure to inferred over-folders or favouring thin value bets versus calling stations—so behaviour adapts to specific opponents instead of playing a static equilibrium strategy.

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

Results are included to validate the learning loop; the primary contribution is the agent architecture and belief‑update mechanism rather than absolute performance.

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
 - **Behavioural stationarity**: Opponent models assume behavioural stationarity over short horizons; long‑term adaptation and strategic deception are not yet modelled.

Design Choices
--------------

- **Frozen policy backbone with online belief updates**: keeps the main decision network stable while beliefs adapt over time.
- **Confidence-weighted opponent updates**: down-weights noisy, low-signal hands to reduce overfitting to variance.
- **Explicit separation between environment state and agent beliefs**: makes it clear which information comes from the shared world vs. inferred opponent models, and how each influences decisions.

