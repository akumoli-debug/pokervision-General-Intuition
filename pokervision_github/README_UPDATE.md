# PokerVision v2.0 – Complete System Documentation

## Overview

PokerVision is a complete poker AI system with **74.3% accuracy**, offering three modes:

- **LIVE** – Real-time recommendations during play (“What should I do?”)
- **REVIEW** – Past hand evaluation with letter grading (A–F)
- **CAPTURE** – Auto screen capture and analysis

All analyses are auto-saved to `data/analyzed_hands/` for future model training. The system exploits opponent-specific patterns and achieves **79–99% accuracy** vs known players.

## Main Application

- **Single-file deployment:** `scripts/live_ui_fixed.py`
  - Run: `python3 scripts/live_ui_fixed.py` then open http://localhost:8000
  - Provides: hand description parsing (ChatGPT-style), manual form, screen capture/upload, opponent belief tracking, action history, telemetry

## Key Features

- 74.3% accuracy AI model
- Real-time poker recommendations during play
- Past hand evaluation with letter grading (A–F)
- Auto screen capture and analysis
- Complete data storage for model training
- Exploits opponent-specific patterns (79–99% accuracy vs known players)
- All analyses auto-saved to `data/analyzed_hands/` for future training

## Repository Contents

- `scripts/live_ui_fixed.py` – Main live UI and analysis server
- `scripts/demo.py` – Quick 10-second demo
- `belief/opponent_belief.py` – Opponent belief state and hand history
- `telemetry/logger.py` – JSONL event logging
- `docs/` – API, usage, training, and GI application docs
- `README.md` – Project overview and quick start
- `CHANGELOG.md` – Version history

See `docs/USAGE_GUIDE.md` for detailed usage.
