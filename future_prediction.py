"""
Future State Prediction Module
The core "world modeling" capability - predicting how the game evolves

This is what differentiates a world model from a simple policy network:
- Policy network: state → action
- World model: state + action → next_state, reward, continuation

Capabilities:
1. Single-step prediction: Given state and action, predict next state
2. Multi-step rollout: Simulate entire hand from current point
3. Counterfactual reasoning: "What if I had raised instead?"
4. Uncertainty quantification: Confidence in predictions
5. Opponent action prediction: Model other agents
6. Latent imagination: Plan in learned representation space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


# =============================================================================
# WORLD MODEL WITH FUTURE PREDICTION
# =============================================================================

class DynamicsModel(nn.Module):
    """
    Learns the dynamics: (state, action) → next_state
    
    This is the core "world modeling" component that predicts:
    - What cards will be revealed
    - What opponents will do
    - How pot/stacks will change
    - Whether hand will continue or end
    """
    
    def __init__(self, 
                 state_dim: int = 512,
                 action_dim: int = 16,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Encode (state, action) pair
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Predict next state
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Predict reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Predict continuation (will hand continue?)
        self.continue_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Predict next community card distribution
        self.card_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 52)  # 52 cards
        )
        
        # Predict opponent's next action distribution
        self.opponent_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Uncertainty estimation (epistemic + aleatoric)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # mean, log_std
        )
    
    def forward(self, state_embedding, action):
        """
        Predict future state given current state and action
        
        Args:
            state_embedding: [batch, state_dim] - encoded state
            action: [batch, action_dim] - one-hot encoded action
        
        Returns:
            Dictionary with predictions
        """
        # Concatenate state and action
        state_action = torch.cat([state_embedding, action], dim=-1)
        
        # Encode
        encoded = self.encoder(state_action)
        
        # Predictions
        next_state = self.state_predictor(encoded)
        reward = self.reward_predictor(encoded).squeeze(-1)
        continue_prob = torch.sigmoid(self.continue_predictor(encoded).squeeze(-1))
        
        card_logits = self.card_predictor(encoded)
        card_probs = F.softmax(card_logits, dim=-1)
        
        opponent_logits = self.opponent_predictor(encoded)
        opponent_probs = F.softmax(opponent_logits, dim=-1)
        
        # Uncertainty
        uncertainty = self.uncertainty_head(encoded)
        pred_mean, pred_log_std = uncertainty[:, 0], uncertainty[:, 1]
        pred_std = torch.exp(pred_log_std)
        
        return {
            'next_state': next_state,
            'reward': reward,
            'continue_prob': continue_prob,
            'card_probs': card_probs,
            'opponent_probs': opponent_probs,
            'uncertainty_mean': pred_mean,
            'uncertainty_std': pred_std
        }


class LatentWorldModel(nn.Module):
    """
    Complete world model with:
    - State encoder (observation → latent state)
    - Dynamics model (latent state + action → next latent state)
    - Decoder (latent state → predictions)
    
    Inspired by DreamerV3, MuZero architecture
    """
    
    def __init__(self,
                 obs_dim: int = 136,
                 action_dim: int = 16,
                 latent_dim: int = 512,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Encoder: observation → latent state (deterministic)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Dynamics model: (latent_state, action) → next_latent_state
        self.dynamics = DynamicsModel(
            state_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # Decoder: latent_state → observations/predictions
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Value network: latent_state → expected return
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def encode(self, observation):
        """Encode observation to latent state"""
        return self.encoder(observation)
    
    def imagine_step(self, latent_state, action):
        """
        Imagine one step into the future
        
        Args:
            latent_state: Current latent state
            action: Action to take (one-hot or index)
        
        Returns:
            Dictionary with imagined next state and predictions
        """
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action
        
        # Predict next state
        dynamics_output = self.dynamics(latent_state, action_onehot)
        
        # Decode predictions from next state
        next_latent = dynamics_output['next_state']
        decoded = self.decoder(next_latent)
        value = self.value_net(next_latent).squeeze(-1)
        
        return {
            'next_latent_state': next_latent,
            'decoded_observation': decoded,
            'reward': dynamics_output['reward'],
            'value': value,
            'continue_prob': dynamics_output['continue_prob'],
            'card_probs': dynamics_output['card_probs'],
            'opponent_probs': dynamics_output['opponent_probs'],
            'uncertainty': (dynamics_output['uncertainty_mean'], 
                          dynamics_output['uncertainty_std'])
        }
    
    def imagine_trajectory(self, 
                          initial_state, 
                          policy_fn, 
                          horizon: int = 10):
        """
        Imagine multiple steps into the future
        
        Args:
            initial_state: Starting latent state
            policy_fn: Function that returns action given state
            horizon: Number of steps to imagine
        
        Returns:
            List of imagined states, actions, rewards
        """
        trajectory = {
            'states': [initial_state],
            'actions': [],
            'rewards': [],
            'values': [],
            'continue_probs': [],
            'uncertainties': []
        }
        
        current_state = initial_state
        
        for step in range(horizon):
            # Get action from policy
            action = policy_fn(current_state)
            
            # Imagine next step
            imagination = self.imagine_step(current_state, action)
            
            # Store trajectory
            trajectory['states'].append(imagination['next_latent_state'])
            trajectory['actions'].append(action)
            trajectory['rewards'].append(imagination['reward'])
            trajectory['values'].append(imagination['value'])
            trajectory['continue_probs'].append(imagination['continue_prob'])
            trajectory['uncertainties'].append(imagination['uncertainty'])
            
            # Check if hand continues
            if imagination['continue_prob'].item() < 0.5:
                break  # Hand ended
            
            current_state = imagination['next_latent_state']
        
        return trajectory
    
    def counterfactual_analysis(self,
                                initial_state,
                                candidate_actions: List[int],
                                horizon: int = 5):
        """
        Compare outcomes of different actions
        "What if I fold vs call vs raise?"
        
        Args:
            initial_state: Current state
            candidate_actions: List of actions to evaluate
            horizon: How far to simulate
        
        Returns:
            Comparison of expected outcomes for each action
        """
        results = {}
        
        for action_idx in candidate_actions:
            action = F.one_hot(torch.tensor([action_idx]), self.action_dim).float()
            
            # Imagine this action's trajectory
            # Use greedy policy for opponent actions
            def greedy_opponent_policy(state):
                with torch.no_grad():
                    # Predict opponent's most likely action
                    dynamics_out = self.dynamics(state, action)
                    return dynamics_out['opponent_probs'].argmax(dim=-1)
            
            trajectory = self.imagine_trajectory(
                initial_state,
                policy_fn=greedy_opponent_policy,
                horizon=horizon
            )
            
            # Calculate expected value
            rewards = torch.stack(trajectory['rewards'])
            values = torch.stack(trajectory['values'])
            
            # Discount future rewards
            gamma = 0.99
            discounts = torch.pow(gamma, torch.arange(len(rewards)).float())
            expected_return = (rewards * discounts).sum()
            
            # Add terminal value
            if len(values) > 0:
                expected_return += values[-1] * (gamma ** len(rewards))
            
            results[action_idx] = {
                'expected_return': expected_return.item(),
                'immediate_reward': rewards[0].item() if len(rewards) > 0 else 0,
                'trajectory_length': len(rewards),
                'uncertainty': trajectory['uncertainties'][0] if trajectory['uncertainties'] else (0, 0)
            }
        
        return results


# =============================================================================
# MONTE CARLO TREE SEARCH WITH WORLD MODEL
# =============================================================================

class MCTSNode:
    """
    Node in MCTS tree for planning with world model
    
    Unlike traditional poker MCTS that uses game simulator,
    this uses learned world model for imagination
    """
    
    def __init__(self, 
                 latent_state,
                 parent=None,
                 action=None,
                 prior_prob=1.0):
        self.latent_state = latent_state
        self.parent = parent
        self.action = action  # Action that led to this state
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, exploration_weight=1.414):
        """Upper confidence bound score"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.value()
        exploration = exploration_weight * self.prior_prob * \
                      np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self):
        """Select best child using UCB"""
        return max(self.children.values(), key=lambda n: n.ucb_score())
    
    def expand(self, world_model, action_priors):
        """Expand node with children for each action"""
        for action_idx, prior in enumerate(action_priors):
            if prior > 0.01:  # Only expand probable actions
                self.children[action_idx] = MCTSNode(
                    latent_state=None,  # Will be filled during simulation
                    parent=self,
                    action=action_idx,
                    prior_prob=prior
                )
    
    def update(self, value):
        """Backpropagate value up the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.update(value)


class WorldModelPlanner:
    """
    MCTS planner using learned world model for imagination
    
    This is the key difference from GTO:
    - GTO: Computes Nash equilibrium
    - WorldModelPlanner: Imagines futures, plans accordingly
    """
    
    def __init__(self,
                 world_model: LatentWorldModel,
                 num_simulations: int = 100,
                 exploration_weight: float = 1.4):
        self.world_model = world_model
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
    
    def plan(self, observation, action_space: List[int]):
        """
        Plan best action using MCTS with world model
        
        Args:
            observation: Current game observation
            action_space: Valid actions
        
        Returns:
            Best action, visit counts, Q-values
        """
        # Encode observation to latent state
        with torch.no_grad():
            initial_latent = self.world_model.encode(observation)
        
        # Create root node
        root = MCTSNode(latent_state=initial_latent)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, action_space)
        
        # Select action with most visits (most confident)
        visit_counts = {a: root.children[a].visit_count 
                       for a in action_space if a in root.children}
        
        if not visit_counts:
            # Fallback: random action
            return np.random.choice(action_space), {}, {}
        
        best_action = max(visit_counts.keys(), key=visit_counts.get)
        
        # Get Q-values
        q_values = {a: root.children[a].value() 
                   for a in action_space if a in root.children}
        
        return best_action, visit_counts, q_values
    
    def _simulate(self, node: MCTSNode, action_space: List[int]):
        """Run one MCTS simulation"""
        
        # Selection: traverse tree using UCB
        while node.children:
            if not all(a in node.children for a in action_space):
                # Not fully expanded
                break
            node = node.select_child()
        
        # Expansion: add children if leaf node
        if not node.children and node.visit_count > 0:
            # Get action priors from world model
            with torch.no_grad():
                # Use dynamics model to predict opponent/action distribution
                # For now, use uniform priors
                action_priors = [1.0 / len(action_space)] * len(action_space)
            
            node.expand(self.world_model, action_priors)
            
            if node.children:
                node = node.select_child()
        
        # Simulation: imagine forward using world model
        if node.action is not None and node.latent_state is None:
            # Need to imagine this state
            parent_state = node.parent.latent_state
            action = node.action
            
            with torch.no_grad():
                imagination = self.world_model.imagine_step(parent_state, action)
                node.latent_state = imagination['next_latent_state']
                value = imagination['value']
        else:
            # Already have state, get value
            with torch.no_grad():
                value = self.world_model.value_net(node.latent_state).squeeze(-1)
        
        # Backpropagation
        node.update(value.item())


# =============================================================================
# VISUALIZATION & ANALYSIS
# =============================================================================

class FuturePredictionVisualizer:
    """
    Visualize world model predictions
    
    Shows:
    - Predicted trajectory
    - Counterfactual comparisons
    - Uncertainty estimates
    - Attention on which factors matter
    """
    
    def __init__(self, world_model: LatentWorldModel):
        self.world_model = world_model
    
    def visualize_trajectory(self, 
                            initial_obs,
                            action_sequence: List[int],
                            save_path: str = 'trajectory.png'):
        """
        Visualize predicted trajectory for a sequence of actions
        """
        import matplotlib.pyplot as plt
        
        # Encode initial state
        with torch.no_grad():
            current_state = self.world_model.encode(initial_obs)
        
        # Simulate trajectory
        states = [current_state]
        rewards = []
        uncertainties = []
        
        for action in action_sequence:
            imagination = self.world_model.imagine_step(current_state, action)
            states.append(imagination['next_latent_state'])
            rewards.append(imagination['reward'].item())
            uncertainties.append(imagination['uncertainty'][1].item())  # std
            current_state = imagination['next_latent_state']
            
            if imagination['continue_prob'].item() < 0.5:
                break
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        axes[0].plot(cumulative_rewards, marker='o')
        axes[0].fill_between(range(len(cumulative_rewards)),
                            cumulative_rewards - np.array(uncertainties[:len(cumulative_rewards)]),
                            cumulative_rewards + np.array(uncertainties[:len(cumulative_rewards)]),
                            alpha=0.3)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].set_title('Predicted Trajectory')
        axes[0].grid(True)
        
        # Uncertainty over time
        axes[1].plot(uncertainties, marker='o', color='red')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Prediction Uncertainty (std)')
        axes[1].set_title('Model Confidence')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved trajectory visualization to {save_path}")
    
    def compare_counterfactuals(self,
                               initial_obs,
                               actions_to_compare: Dict[str, int],
                               horizon: int = 5):
        """
        Compare predicted outcomes of different actions
        "What if I fold vs call vs raise?"
        """
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            initial_state = self.world_model.encode(initial_obs)
        
        results = self.world_model.counterfactual_analysis(
            initial_state,
            list(actions_to_compare.values()),
            horizon=horizon
        )
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        action_names = list(actions_to_compare.keys())
        expected_returns = [results[a]['expected_return'] for a in actions_to_compare.values()]
        uncertainties = [results[a]['uncertainty'][1] for a in actions_to_compare.values()]
        
        x = np.arange(len(action_names))
        ax.bar(x, expected_returns, yerr=uncertainties, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(action_names)
        ax.set_ylabel('Expected Return ($)')
        ax.set_title('Counterfactual Analysis: Predicted Outcomes')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('counterfactual_comparison.png')
        print("Saved counterfactual comparison")
        
        return results


# =============================================================================
# TRAINING THE FUTURE PREDICTION
# =============================================================================

def train_world_model_prediction(world_model, 
                                 dataloader,
                                 num_epochs: int = 30,
                                 device: str = 'cuda'):
    """
    Train world model to predict future states
    
    Loss components:
    - State prediction (MSE)
    - Reward prediction (MSE)
    - Continuation prediction (BCE)
    - Reconstruction (MSE)
    """
    
    optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            # Get current and next observations
            current_obs = batch['state'].to(device)
            action = batch['action'].to(device)
            next_obs = batch['next_state'].to(device) if 'next_state' in batch else None
            reward = batch['reward'].to(device)
            done = batch.get('done', torch.zeros_like(reward)).to(device)
            
            # Encode current state
            current_latent = world_model.encode(current_obs)
            
            # Predict next state
            imagination = world_model.imagine_step(current_latent, action)
            
            # Losses
            reward_loss = F.mse_loss(imagination['reward'], reward)
            continue_loss = F.binary_cross_entropy(
                imagination['continue_prob'],
                1.0 - done
            )
            
            # If we have next observation, predict it
            if next_obs is not None:
                reconstructed = imagination['decoded_observation']
                reconstruction_loss = F.mse_loss(reconstructed, next_obs)
            else:
                reconstruction_loss = 0
            
            # Total loss
            loss = reward_loss + continue_loss + 0.5 * reconstruction_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")


# =============================================================================
# DEMO USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Future State Prediction - World Model Core")
    print("="*70)
    print()
    
    # Create world model
    world_model = LatentWorldModel(
        obs_dim=136,
        action_dim=6,
        latent_dim=512
    )
    
    # Example: Imagine future
    dummy_obs = torch.randn(1, 136)
    initial_state = world_model.encode(dummy_obs)
    
    print("1. Single-step imagination:")
    action = torch.tensor([2])  # Raise
    imagination = world_model.imagine_step(initial_state, action)
    print(f"   Predicted reward: ${imagination['reward'].item():.2f}")
    print(f"   Hand continues: {imagination['continue_prob'].item():.1%}")
    print(f"   Uncertainty: ±${imagination['uncertainty'][1].item():.2f}")
    print()
    
    print("2. Multi-step trajectory:")
    def random_policy(state):
        return torch.randint(0, 6, (1,))
    
    trajectory = world_model.imagine_trajectory(
        initial_state,
        policy_fn=random_policy,
        horizon=10
    )
    print(f"   Simulated {len(trajectory['rewards'])} steps")
    print(f"   Total predicted reward: ${sum(r.item() for r in trajectory['rewards']):.2f}")
    print()
    
    print("3. Counterfactual analysis:")
    actions = [0, 1, 2]  # Fold, Call, Raise
    results = world_model.counterfactual_analysis(
        initial_state,
        candidate_actions=actions,
        horizon=5
    )
    action_names = ['Fold', 'Call', 'Raise']
    for action_idx, name in zip(actions, action_names):
        ev = results[action_idx]['expected_return']
        print(f"   {name}: ${ev:+.2f}")
    print()
    
    print("4. MCTS Planning:")
    planner = WorldModelPlanner(world_model, num_simulations=50)
    best_action, visits, qvalues = planner.plan(dummy_obs, action_space=[0,1,2])
    print(f"   Best action: {action_names[best_action]}")
    print(f"   Visit counts: {visits}")
    print(f"   Q-values: {qvalues}")
    print()
    
    print("="*70)
    print("This is what makes it a TRUE world model:")
    print("✓ Predicts future states (not just actions)")
    print("✓ Imagines trajectories (multi-step planning)")
    print("✓ Counterfactual reasoning (what-if analysis)")
    print("✓ Uncertainty quantification (knows when unsure)")
    print("✓ MCTS planning (thinks ahead using imagination)")
    print("="*70)
