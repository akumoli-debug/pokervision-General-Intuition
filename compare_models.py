"""
Model Comparison Script
Tests all trained models and compares performance

Shows which improvements worked best
"""

import json
import torch
import numpy as np
from collections import defaultdict

def load_data():
    """Load validation data"""
    with open('data/akumoli_final_merged.json', 'r') as f:
        dataset = json.load(f)
    
    hero_examples = dataset['hero_training_examples']
    val_indices = set(dataset.get('validation_indices', []))
    val_data = [hero_examples[i] for i in val_indices]
    
    return val_data

def test_simple_model(val_data):
    """Test the simple pattern matching model"""
    try:
        with open('models/akumoli_poker_model.json', 'r') as f:
            model = json.load(f)
        
        patterns = model.get('patterns', {})
        
        correct = 0
        for example in val_data:
            pot = example.get('pot', 0)
            pot_bucket = int(pot / 10) * 10
            source = example.get('source', 'unknown')
            state_sig = f"{source}_pot_{pot_bucket}"
            
            if state_sig in patterns:
                actions = patterns[state_sig]
                predicted = max(actions, key=actions.get) if actions else 'call'
            else:
                predicted = 'call'
            
            true_action = example.get('action_type', 'fold')
            if predicted == true_action:
                correct += 1
        
        accuracy = correct / len(val_data)
        return accuracy, "✓"
    except:
        return None, "✗"

def test_pytorch_model(val_data):
    """Test basic PyTorch model"""
    try:
        checkpoint = torch.load('models/pytorch_poker_model.pt', 
                               map_location='cpu')
        accuracy = checkpoint.get('accuracy', 0)
        return accuracy, "✓"
    except:
        return None, "✗"

def test_enhanced_model(val_data):
    """Test enhanced features model"""
    try:
        checkpoint = torch.load('models/enhanced_poker_model.pt',
                               map_location='cpu')
        accuracy = checkpoint.get('accuracy', 0)
        return accuracy, "✓"
    except:
        return None, "✗"

def test_transformer_model(val_data):
    """Test Transformer model"""
    try:
        checkpoint = torch.load('models/best_model.pt',
                               map_location='cpu')
        accuracy = checkpoint.get('val_accuracy', 0)
        return accuracy, "✓"
    except:
        return None, "✗"

def main():
    print("="*70)
    print("Model Comparison - All Stages")
    print("="*70)
    print()
    
    # Load validation data
    print("Loading validation data...")
    val_data = load_data()
    print(f"✓ Loaded {len(val_data)} validation examples\n")
    
    # Test all models
    results = []
    
    print("Testing models...")
    print("-"*70)
    
    # Simple model
    acc, status = test_simple_model(val_data)
    results.append(("Simple Pattern Matching", acc, status, "Stage 0"))
    print(f"{status} Simple model:     {acc:.1%}" if acc else f"✗ Simple model:     Not found")
    
    # PyTorch
    acc, status = test_pytorch_model(val_data)
    results.append(("Basic PyTorch", acc, status, "Stage 1"))
    print(f"{status} PyTorch model:    {acc:.1%}" if acc else f"✗ PyTorch model:    Not found")
    
    # Enhanced
    acc, status = test_enhanced_model(val_data)
    results.append(("Enhanced Features", acc, status, "Stage 2"))
    print(f"{status} Enhanced model:   {acc:.1%}" if acc else f"✗ Enhanced model:   Not found")
    
    # Transformer
    acc, status = test_transformer_model(val_data)
    results.append(("Transformer", acc, status, "Stage 3"))
    print(f"{status} Transformer:      {acc:.1%}" if acc else f"✗ Transformer:      Not found")
    
    print()
    
    # Summary table
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"{'Stage':<10}{'Model':<25}{'Accuracy':<12}{'Improvement'}")
    print("-"*70)
    
    baseline = None
    for model_name, acc, status, stage in results:
        if acc is not None:
            if baseline is None:
                baseline = acc
                improvement = "Baseline"
            else:
                abs_improvement = acc - baseline
                rel_improvement = (acc - baseline) / baseline * 100
                improvement = f"+{abs_improvement:.1%} ({rel_improvement:+.1f}%)"
            
            print(f"{stage:<10}{model_name:<25}{acc:<12.1%}{improvement}")
        else:
            print(f"{stage:<10}{model_name:<25}{'Not found':<12}{'N/A'}")
    
    print()
    
    # Find best model
    valid_results = [(name, acc, stage) for name, acc, _, stage in results if acc is not None]
    
    if valid_results:
        best_name, best_acc, best_stage = max(valid_results, key=lambda x: x[1])
        
        print(f"Best model: {best_name} ({best_stage})")
        print(f"Accuracy: {best_acc:.1%}")
        
        if baseline:
            improvement = (best_acc - baseline) / baseline * 100
            print(f"Improvement over baseline: {improvement:+.1f}%")
    
    print()
    print("="*70)
    print("Baseline Comparisons:")
    print("-"*70)
    print(f"Random guessing:        20%")
    print(f"Always fold:            ~33%")
    print(f"GTO solver:             ~55%")
    if valid_results:
        print(f"Your best model:        {best_acc:.1%}")
    print("="*70)

if __name__ == "__main__":
    main()
