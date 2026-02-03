#!/bin/bash
# Complete Setup and Training Script
# Run this after downloading all files

echo "======================================================================"
echo "PokerVision - Complete Setup and Training"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "data" ] || [ ! -d "models" ]; then
    echo "✗ Error: Not in pokervision directory"
    echo "  Please cd to ~/pokervision first"
    exit 1
fi

echo "✓ In correct directory"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found"
    echo "  Please install from python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python found: $PYTHON_VERSION"
echo ""

# Check data file
if [ ! -f "data/akumoli_final_merged.json" ]; then
    echo "✗ Training data not found: data/akumoli_final_merged.json"
    echo "  Please download and place in data/ folder"
    exit 1
fi

echo "✓ Training data found"
echo ""

# Install requirements
echo "Installing requirements..."
echo "────────────────────────────────────────────────────────────────────"

pip3 install torch numpy --quiet

if [ $? -eq 0 ]; then
    echo "✓ Requirements installed"
else
    echo "✗ Installation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Setup Complete - Ready to Train"
echo "======================================================================"
echo ""
echo "Choose training stage:"
echo ""
echo "  1. Quick test (Stage 1 only, 10 min)   → 55-60% accuracy"
echo "  2. Good demo (Stages 1-2, 1 hour)      → 60-65% accuracy"  
echo "  3. Best demo (All stages, 2-3 hours)   → 65-70% accuracy"
echo "  4. Just compare existing models"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running Stage 1: Basic PyTorch"
        echo "────────────────────────────────────────────────────────────────────"
        python3 train_all_stages.py --stage 1
        ;;
    2)
        echo ""
        echo "Running Stages 1-2: Enhanced Features"
        echo "────────────────────────────────────────────────────────────────────"
        python3 train_all_stages.py --stage 2
        ;;
    3)
        echo ""
        echo "Running All Stages: Complete Pipeline"
        echo "────────────────────────────────────────────────────────────────────"
        python3 train_all_stages.py --all
        ;;
    4)
        echo ""
        echo "Comparing Models"
        echo "────────────────────────────────────────────────────────────────────"
        python3 compare_models.py
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Run comparison after training
echo ""
echo "======================================================================"
echo "Running Model Comparison"
echo "======================================================================"
python3 compare_models.py

echo ""
echo "======================================================================"
echo "Next Steps"
echo "======================================================================"
echo ""
echo "1. Analyze results:"
echo "   python3 analyze_model.py"
echo ""
echo "2. Check specific opponents:"
echo "   python3 analyze_model.py seb"
echo "   python3 analyze_model.py 'punter sausage'"
echo ""
echo "3. Compare all models:"
echo "   python3 compare_models.py"
echo ""
echo "4. Create demo video and apply to General Intuition!"
echo ""
echo "======================================================================"
