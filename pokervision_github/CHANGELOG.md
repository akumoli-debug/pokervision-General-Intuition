# Changelog

## v2.0 – Complete PokerVision with LIVE/REVIEW/CAPTURE modes

- 74.3% accuracy AI model
- Real-time poker recommendations during play
- Past hand evaluation with letter grading (A–F)
- Auto screen capture and analysis
- Complete data storage for model training
- Single-file deployment (`scripts/live_ui_fixed.py`)
- Three modes: LIVE (what should I do?), REVIEW (was my play good?), CAPTURE (auto-analyze screen)
- Exploits opponent-specific patterns (79–99% accuracy vs known players)
- All analyses auto-saved to `data/analyzed_hands/` for future training
- Hand description parsing (ChatGPT-style), opponent beliefs, action history, telemetry
- Broken-pipe handling for client disconnects

## Earlier versions

- v1.x: Enhanced model with cards, opponent fine-tuning, belief state, telemetry, live UI with manual/capture/description input.
