# PokerVision Usage Guide

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the live UI**
   ```bash
   cd pokervision_github
   python3 scripts/live_ui_fixed.py
   ```
   Then open http://localhost:8000 in your browser.

3. **10-second demo (no server)**
   ```bash
   python3 demo.py
   ```

## Modes

### LIVE – “What should I do?”

- Use the **Hand Description** box: paste a hand history or describe the situation in plain English.
- Click **Analyze & Recommend** for an immediate recommendation (action, bet size, reasoning).
- Or fill the manual form (pot, bet, stacks, cards, position, street, opponent action) and click **Analyze & Recommend**.
- The system uses opponent beliefs and action history when available.

### REVIEW – “Was my play good?”

- After analysis, the result shows recommendation, metrics (pot, to call, pot odds, SPR), hand strength, and action history insights.
- Use **Action History** (JSON) to include previous streets for better grading.

### CAPTURE – Auto-analyze screen

- Open the **Screen Capture** tab.
- **Capture Screen & Analyze** – captures the primary screen, runs OCR, and tries to detect pot, stacks, cards, and players.
- **Upload Screenshot & Analyze** – upload an image file instead.
- If multiple players are detected, you’ll be asked to select “Who are you?”; then the form is filled and analysis runs.

## Tips

- **Hand description format:** e.g. “1/3 8k effective. Raise KTo UTG to 20. BU 3-bets to 100. I call. Flop T42r. River 2. He bets 2k.”
- **Cards:** Use format like `Ah Kd` (Ace of hearts, King of diamonds). Ranks: 2–9, T, J, Q, K, A. Suits: h, d, c, s.
- **Action history:** Optional JSON array of previous street actions for better recommendations.
- **Telemetry:** Events are logged to `logs/telemetry.jsonl`. Use `python3 scripts/inspect_telemetry.py` to summarize.

## Data and Training

- Analyses can be saved under `data/analyzed_hands/` for future model training.
- See `docs/TRAINING.md` for training the neural model and `docs/API.md` for programmatic usage.
