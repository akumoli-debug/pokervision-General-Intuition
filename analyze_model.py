"""
Model Analyzer - Shows insights from trained model

Analyzes:
1. Your playing patterns
2. Opponent exploitability  
3. Recommended strategies
"""

import json
from collections import defaultdict

def analyze_dataset(data_path):
    """Analyze the training dataset"""
    
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    metadata = dataset['metadata']
    hero_examples = dataset['hero_training_examples']
    opponent_models = dataset.get('opponent_models', {})
    
    print("="*70)
    print("POKERVISION - DATASET ANALYSIS")
    print("="*70)
    print()
    
    # Your play patterns
    print("🎮 YOUR PLAYING PATTERNS")
    print("-"*70)
    
    action_counts = defaultdict(int)
    for example in hero_examples:
        action = example.get('action_type', 'unknown')
        action_counts[action] += 1
    
    total_actions = sum(action_counts.values())
    
    print(f"Total decisions: {total_actions}")
    print("\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_actions) * 100
        bar = "█" * int(pct / 2)
        print(f"  {action:10s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Opponent analysis
    print("\n" + "="*70)
    print("👥 OPPONENT ANALYSIS")
    print("-"*70)
    
    # Sort by exploitability
    exploitable = []
    
    for player, stats in opponent_models.items():
        aggr = stats.get('aggression_factor', 0)
        fold = stats.get('fold_frequency', 0)
        actions = stats.get('total_actions', 0)
        
        # Calculate exploitability score
        # High fold rate OR very low fold rate = exploitable
        exploit_score = 0
        strategy = ""
        
        if fold > 0.60:
            exploit_score = fold * 100
            strategy = "BET/BLUFF (they fold often)"
        elif fold < 0.30:
            exploit_score = (1 - fold) * 100
            strategy = "VALUE BET (they call everything)"
        elif aggr > 2.5:
            exploit_score = aggr * 20
            strategy = "CHECK-RAISE (they're aggressive)"
        
        if exploit_score > 0 and actions >= 100:
            exploitable.append({
                'player': player,
                'score': exploit_score,
                'strategy': strategy,
                'actions': actions,
                'aggr': aggr,
                'fold': fold
            })
    
    exploitable.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nTop 10 Most Exploitable Opponents:")
    print(f"{'Rank':<6}{'Player':<20}{'Actions':<10}{'Strategy'}")
    print("-"*70)
    
    for i, opp in enumerate(exploitable[:10], 1):
        print(f"{i:<6}{opp['player']:<20}{opp['actions']:<10}{opp['strategy']}")
    
    # Detailed recommendations
    print("\n" + "="*70)
    print("💡 STRATEGIC RECOMMENDATIONS")
    print("-"*70)
    
    if exploitable:
        top = exploitable[0]
        print(f"\n🎯 #1 TARGET: {top['player']}")
        print(f"   Actions observed: {top['actions']}")
        print(f"   Aggression: {top['aggr']:.2f}")
        print(f"   Fold frequency: {top['fold']:.1%}")
        print(f"   → {top['strategy']}")
        print(f"   Expected EV gain: +${top['score']/10:.2f} per hand")
    
    # Statistics summary
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("-"*70)
    print(f"Total data points:    {metadata.get('total_examples', 0):,}")
    print(f"Your decisions:       {metadata.get('hero_examples', 0):,}")
    print(f"Opponent observations: {metadata.get('opponent_observations', 0):,}")
    print(f"Unique opponents:     {metadata.get('unique_opponents', 0)}")
    print(f"Opponents modeled:    {len(opponent_models)}")
    
    # Top opponents by data
    print("\n📈 Most Data on These Opponents:")
    sorted_opponents = sorted(opponent_models.items(), 
                             key=lambda x: x[1].get('total_actions', 0), 
                             reverse=True)
    
    for i, (player, stats) in enumerate(sorted_opponents[:5], 1):
        actions = stats.get('total_actions', 0)
        aggr = stats.get('aggression_factor', 0)
        fold = stats.get('fold_frequency', 0)
        print(f"  {i}. {player:<20} {actions:4d} actions (Aggr: {aggr:.2f}, Fold: {fold:.1%})")
    
    print("\n" + "="*70)


def show_opponent_details(data_path, opponent_name):
    """Show detailed analysis of specific opponent"""
    
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    opponent_models = dataset.get('opponent_models', {})
    
    # Find opponent (case-insensitive)
    opponent_name_lower = opponent_name.lower()
    opponent_data = None
    actual_name = None
    
    for name, data in opponent_models.items():
        if name.lower() == opponent_name_lower:
            opponent_data = data
            actual_name = name
            break
    
    if not opponent_data:
        print(f"❌ Opponent '{opponent_name}' not found")
        print(f"\nAvailable opponents: {', '.join(list(opponent_models.keys())[:10])}")
        return
    
    print("="*70)
    print(f"OPPONENT PROFILE: {actual_name}")
    print("="*70)
    print()
    
    actions = opponent_data.get('total_actions', 0)
    aggr = opponent_data.get('aggression_factor', 0)
    fold = opponent_data.get('fold_frequency', 0)
    aliases = opponent_data.get('aliases_used', [])
    
    print(f"Actions observed:     {actions}")
    print(f"Aggression factor:    {aggr:.2f}")
    print(f"Fold frequency:       {fold:.1%}")
    
    if aliases:
        print(f"Known aliases:        {', '.join(aliases)}")
    
    print()
    print("PLAYER TYPE:")
    
    if fold > 0.60:
        print(f"  🎯 TIGHT/FOLDY ({fold:.1%} fold rate)")
        print(f"  → Strategy: Bet and bluff more often")
        print(f"  → They fold too much, exploit with aggression")
    elif fold < 0.30:
        print(f"  💰 CALLING STATION ({fold:.1%} fold rate)")
        print(f"  → Strategy: Value bet, never bluff")
        print(f"  → They call everything, only bet with strong hands")
    else:
        print(f"  📊 BALANCED ({fold:.1%} fold rate)")
    
    print()
    
    if aggr > 2.0:
        print(f"  🚀 AGGRESSIVE (aggression {aggr:.2f})")
        print(f"  → Strategy: Check-raise, trap them")
        print(f"  → Let them bet, then raise")
    elif aggr < 0.8:
        print(f"  🐌 PASSIVE (aggression {aggr:.2f})")
        print(f"  → Strategy: Bet when they check")
        print(f"  → Take control, they won't fight back")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    data_path = 'data/akumoli_final_merged.json'
    
    if len(sys.argv) > 1:
        # Show specific opponent
        opponent_name = ' '.join(sys.argv[1:])
        show_opponent_details(data_path, opponent_name)
    else:
        # Show overall analysis
        analyze_dataset(data_path)
        
        print("\nTip: To see details on specific opponent:")
        print("  python analyze_model.py seb")
        print("  python analyze_model.py 'punter sausage'")
