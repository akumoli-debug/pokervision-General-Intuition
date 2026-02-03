#!/bin/bash

echo "======================================================================"
echo "PokerVision Setup"
echo "======================================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "✓ Setup complete!"
echo
echo "Next steps:"
echo "  1. Place PokerNow CSV logs in data/"
echo "  2. Run: python scripts/enhanced_complete_parser.py"
echo "  3. Run: python scripts/train_with_cards.py"
echo "  4. Run: python scripts/live_ui_fixed.py"
echo
