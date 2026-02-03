"""
PokerVision Training Script
Simplified version that works on Mac (CPU or GPU)

This trains a basic model on your poker data.
For production PyTorch version, see advanced_world_model.py
"""

import json
import random
from collections import defaultdict
import math

class SimplePokerModel:
    """
    Simplified poker model for demo purposes
    
    For production: Replace with PyTorch Transformer from advanced_world_model.py
    This version: Pattern matching with statistical learning
    """
    
    def __init__(self):
        self.patterns = defaultdict(lambda: defaultdict(int))
        self.outcomes = defaultdict(list)
        self.trained = False
        
    def _get_state_signature(self, state):
        """Create pattern signature from state"""
        # Simplify state to key features
        pot = state.get('pot', 0)
        pot_bucket = int(pot / 10) * 10  # Round to nearest 10
        
        source = state.get('source', 'unknown')
        
        return f"{source}_pot_{pot_bucket}"
    
    def train(self, training_data, epochs=20):
        """Train on your poker data"""
        
        print(f"\nTraining for {epochs} epochs...")
        print("="*60)
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            
            correct = 0
            total = 0
            
            for example in training_data:
                # Get state signature
                state_sig = self._get_state_signature(example)
                
                # Record action frequency
                action = example.get('action_type', 'unknown')
                self.patterns[state_sig][action] += 1
                
                # Record outcomes
                reward = example.get('reward', 0)
                self.outcomes[(state_sig, action)].append(reward)
                
                # Check prediction
                prediction = self.predict_action(example)
                if prediction == action:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Accuracy = {accuracy:.1%}")
        
        self.trained = True
        print("="*60)
        print("✓ Training complete!\n")
    
    def predict_action(self, state):
        """Predict best action for state"""
        if not self.trained:
            return 'call'  # Default
        
        state_sig = self._get_state_signature(state)
        
        if state_sig in self.patterns:
            actions = self.patterns[state_sig]
            if actions:
                return max(actions, key=actions.get)
        
        return 'call'  # Default fallback
    
    def predict_outcome(self, state, action):
        """Predict expected value"""
        state_sig = self._get_state_signature(state)
        key = (state_sig, action)
        
        if key in self.outcomes and self.outcomes[key]:
            return sum(self.outcomes[key]) / len(self.outcomes[key])
        
        return 0.0
    
    def evaluate(self, test_data):
        """Evaluate on test set"""
        correct = 0
        total = 0
        total_error = 0
        
        for example in test_data:
            # Action prediction
            predicted_action = self.predict_action(example)
            true_action = example.get('action_type', 'unknown')
            
            if predicted_action == true_action:
                correct += 1
            total += 1
            
            # Outcome prediction
            predicted_ev = self.predict_outcome(example, predicted_action)
            true_reward = example.get('reward', 0)
            error = (predicted_ev - true_reward) ** 2
            total_error += error
        
        accuracy = correct / total if total > 0 else 0
        rmse = math.sqrt(total_error / total) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'rmse': rmse,
            'total_examples': total
        }
    
    def save(self, filepath):
        """Save model"""
        model_data = {
            'patterns': dict(self.patterns),
            'outcomes': {str(k): v for k, v in self.outcomes.items()},
            'trained': self.trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✓ Model saved to: {filepath}")


def load_data(filepath):
    """Load training data"""
    print(f"Loading data from: {filepath}")
    
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    
    metadata = dataset['metadata']
    hero_examples = dataset['hero_training_examples']
    val_indices = set(dataset.get('validation_indices', []))
    
    # Split train/val
    train_data = [ex for i, ex in enumerate(hero_examples) if i not in val_indices]
    val_data = [hero_examples[i] for i in val_indices]
    
    print(f"✓ Loaded dataset")
    print(f"  Total examples: {len(hero_examples)}")
    print(f"  Training: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Opponents modeled: {metadata.get('unique_opponents', 0)}")
    
    return train_data, val_data, metadata


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("PokerVision Training Pipeline")
    print("="*60)
    print()
    
    # Load data
    data_path = 'data/akumoli_final_merged.json'
    
    try:
        train_data, val_data, metadata = load_data(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_path}")
        print("\nMake sure you:")
        print("  1. Downloaded akumoli_final_merged.json")
        print("  2. Placed it in: ~/pokervision/data/")
        return
    
    # Create model
    print("\nInitializing model...")
    model = SimplePokerModel()
    
    # Train
    model.train(train_data, epochs=20)
    
    # Evaluate
    print("Evaluating on validation set...")
    print("="*60)
    
    val_results = model.evaluate(val_data)
    
    print(f"Validation Accuracy:  {val_results['accuracy']:.1%}")
    print(f"Outcome RMSE:         ${val_results['rmse']:.2f}")
    print(f"Test examples:        {val_results['total_examples']}")
    
    # Save
    print("\n" + "="*60)
    model.save('models/akumoli_poker_model.json')
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print()
    print("Your model is ready!")
    print(f"  Accuracy: {val_results['accuracy']:.1%}")
    print(f"  Trained on: {len(train_data)} examples")
    print(f"  Opponents: {metadata.get('unique_opponents', 0)}")
    print()
    print("Next steps:")
    print("  1. Run: python analyze_model.py")
    print("  2. Or use interactive analyzer")
    print("="*60)


if __name__ == "__main__":
    main()
