#!/usr/bin/env python3
"""
Master Training Pipeline
Runs all improvement stages automatically

Usage:
    python3 train_all_stages.py --stage 1    # Basic PyTorch only
    python3 train_all_stages.py --stage 2    # Through enhanced features
    python3 train_all_stages.py --stage 3    # Full pipeline
    python3 train_all_stages.py --all        # Everything
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_header(text):
    """Print fancy header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def print_stage(stage_num, title):
    """Print stage header"""
    print("\n" + "─"*70)
    print(f"STAGE {stage_num}: {title}")
    print("─"*70 + "\n")

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"→ {description}")
    print(f"  Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"✗ Failed (exit code {result.returncode})")
        return False
    else:
        print(f"✓ Success ({elapsed:.1f}s)\n")
        return True

def check_requirements():
    """Check if required packages are installed"""
    print_header("Checking Requirements")
    
    required = {
        'torch': 'pip3 install torch',
        'numpy': 'pip3 install numpy',
    }
    
    missing = []
    
    for package, install_cmd in required.items():
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not found")
            missing.append((package, install_cmd))
    
    if missing:
        print("\nMissing packages. Install with:")
        for package, cmd in missing:
            print(f"  {cmd}")
        
        response = input("\nInstall now? (y/n): ").strip().lower()
        if response == 'y':
            for package, cmd in missing:
                run_command(cmd, f"Installing {package}")
        else:
            print("Please install manually and try again.")
            sys.exit(1)
    
    print("\n✓ All requirements satisfied")

def stage_1_basic_pytorch():
    """Stage 1: Basic PyTorch Neural Network"""
    print_stage(1, "Basic PyTorch Neural Network (→ 55-60%)")
    
    if not os.path.exists('train_pytorch.py'):
        print("✗ train_pytorch.py not found")
        print("  Please download from Claude")
        return False
    
    success = run_command(
        'python3 train_pytorch.py',
        'Training basic PyTorch model'
    )
    
    if success:
        print("✓ Stage 1 Complete")
        print("  Expected accuracy: 55-60%")
        print("  Model saved: models/pytorch_poker_model.pt")
    
    return success

def stage_2_enhanced_features():
    """Stage 2: Enhanced Feature Engineering"""
    print_stage(2, "Enhanced Feature Engineering (→ 60-65%)")
    
    # Check files
    if not os.path.exists('enhance_features.py'):
        print("✗ enhance_features.py not found")
        return False
    
    if not os.path.exists('train_enhanced.py'):
        print("✗ train_enhanced.py not found")
        return False
    
    # Step 1: Extract features
    print("Step 1: Extracting enhanced features...")
    success = run_command(
        'python3 enhance_features.py',
        'Creating 35-dimensional feature vectors'
    )
    
    if not success:
        return False
    
    # Step 2: Train
    print("Step 2: Training on enhanced features...")
    success = run_command(
        'python3 train_enhanced.py',
        'Training with rich features'
    )
    
    if success:
        print("✓ Stage 2 Complete")
        print("  Expected accuracy: 60-65%")
        print("  Model saved: models/enhanced_poker_model.pt")
    
    return success

def stage_3_transformer():
    """Stage 3: Transformer Architecture"""
    print_stage(3, "Transformer Architecture (→ 65-70%)")
    
    if not os.path.exists('advanced_world_model.py'):
        print("✗ advanced_world_model.py not found")
        print("  Please download from Claude")
        return False
    
    success = run_command(
        'python3 advanced_world_model.py '
        '--data data/akumoli_final_merged.json '
        '--epochs 50 '
        '--batch-size 16 '
        '--hidden-dim 512 '
        '--num-layers 6 '
        '--save-dir models/',
        'Training Transformer model'
    )
    
    if success:
        print("✓ Stage 3 Complete")
        print("  Expected accuracy: 65-70%")
        print("  Model saved: models/best_model.pt")
    
    return success

def generate_report(stages_completed):
    """Generate final report"""
    print_header("Training Complete - Summary Report")
    
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stages completed: {stages_completed}")
    print()
    
    # Check what models exist
    models = {
        'pytorch_poker_model.pt': 'Stage 1 (Basic PyTorch)',
        'enhanced_poker_model.pt': 'Stage 2 (Enhanced Features)',
        'best_model.pt': 'Stage 3 (Transformer)'
    }
    
    print("Models created:")
    for model_file, description in models.items():
        path = f'models/{model_file}'
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"  ✓ {model_file:30s} ({size:.1f} MB) - {description}")
        else:
            print(f"  ✗ {model_file:30s} - Not found")
    
    print("\nNext steps:")
    print("  1. Run: python3 analyze_model.py")
    print("  2. Test specific opponents: python3 analyze_model.py seb")
    print("  3. Create demo materials")
    print("  4. Apply to General Intuition!")
    print()
    print("="*70)

def main():
    """Main pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PokerVision Training Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                       help='Run up to this stage')
    parser.add_argument('--all', action='store_true',
                       help='Run all stages')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip requirement checks')
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.all:
        max_stage = 3
    elif args.stage:
        max_stage = args.stage
    else:
        print("Usage: python3 train_all_stages.py --stage N  or  --all")
        print("\nStages:")
        print("  1: Basic PyTorch (55-60% accuracy)")
        print("  2: Enhanced features (60-65% accuracy)")
        print("  3: Transformer (65-70% accuracy)")
        sys.exit(1)
    
    print_header("PokerVision Training Pipeline")
    print(f"Target: Complete through Stage {max_stage}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check requirements
    if not args.skip_checks:
        check_requirements()
    
    # Check data exists
    if not os.path.exists('data/akumoli_final_merged.json'):
        print("\n✗ Error: data/akumoli_final_merged.json not found")
        print("  Please ensure training data is in place")
        sys.exit(1)
    
    # Run stages
    stages_completed = 0
    
    if max_stage >= 1:
        if stage_1_basic_pytorch():
            stages_completed = 1
        else:
            print("\n✗ Stage 1 failed. Stopping.")
            sys.exit(1)
    
    if max_stage >= 2:
        if stage_2_enhanced_features():
            stages_completed = 2
        else:
            print("\n✗ Stage 2 failed. Stopping.")
            sys.exit(1)
    
    if max_stage >= 3:
        if stage_3_transformer():
            stages_completed = 3
        else:
            print("\n✗ Stage 3 failed. Check logs above.")
            sys.exit(1)
    
    # Generate report
    generate_report(stages_completed)

if __name__ == "__main__":
    main()
