from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network with safeguards against NaN values.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        
        # Apply softmax with numerical stability
        # Using log_softmax and then exp is more numerically stable
        x = F.log_softmax(x, dim=-1)
        x = torch.exp(x)
        
        # Ensure no NaNs, replace with small positive values if needed
        x = torch.nan_to_num(x, nan=1e-6)
        
        # Ensure probabilities sum to 1
        x = x / torch.sum(x, dim=-1, keepdim=True).clamp(min=1e-6)
        
        return x
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.steps_done = 0
        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
    def scale_action_MC(self, action):
        """
        Scale the action appropriately for the environment.
        
        Args:
            action: Action from policy network.
        
        Returns:
            Scaled action suitable for the environment.
        """
        # The action manager expects a tensor with shape [batch_size, action_dim]
        if isinstance(action, torch.Tensor):
            # Convert to a 2D tensor with shape [1, 1]
            return torch.tensor([[action.item()]], dtype=torch.float32)
        else:
            # If not a tensor, convert to a 2D tensor with shape [1, 1]
            return torch.tensor([[action]], dtype=torch.float32)
        
    def calculate_stepwise_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Handle normalization more carefully
        if len(returns) > 1:  # Only normalize if we have enough samples
            mean = returns.mean()
            std = returns.std(unbiased=False)  # Use biased estimate to avoid warning
            returns = (returns - mean) / (std + 1e-8)
        
        return returns
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (episode_return, stepwise_returns, log_prob_actions, trajectory)
        """
        # ===== Initialize trajectory collection variables ===== #
        state, _ = env.reset()

        # Ensure state is on the correct device
        if isinstance(state["policy"], torch.Tensor):
            state = state["policy"].float().to(self.device).detach()
        else:
            state = torch.tensor(state["policy"], dtype=torch.float32, device=self.device)

        log_prob_actions = []
        rewards = []
        trajectory = []
        episode_return = 0.0
        done = False
        timestep = 0
        # ====================================== #
        
        # ===== Collect trajectory through agent-environment interaction ===== #
        while not done:
            
            # Make sure state is on the correct device
            state = state.to(self.device)  # Extra safeguard
            
            # Check for NaN in state
            if torch.isnan(state).any():
                print("WARNING: NaN values detected in state!")
                state = torch.nan_to_num(state, nan=0.0)
            
            probs = self.policy_net(state)
            
            # Check for NaN in probabilities
            if torch.isnan(probs).any():
                print("WARNING: NaN values detected in probabilities!")
                print(f"State that caused NaN: {state}")
                # Replace NaNs with uniform distribution
                probs = torch.ones_like(probs) / probs.size(-1)
                
        
 
            dist = distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            

            # Handle action for environment
            action_tensor = self.scale_action_MC(action)
            # Ensure action is in appropriate format for env interaction
            if hasattr(action_tensor, 'cpu'):
                action_for_env = action_tensor.cpu().numpy() if hasattr(action_tensor, 'numpy') else action_tensor.cpu()
            else:
                action_for_env = action_tensor

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated

            # Process next state
            if isinstance(next_obs["policy"], torch.Tensor):
                next_state = next_obs["policy"].float().to(self.device).detach()
            else:
                next_state = torch.tensor(next_obs["policy"], dtype=torch.float32, device=self.device)

            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)

            log_prob_actions.append(log_prob)
            rewards.append(reward)
            trajectory.append((state, action, reward_tensor, next_state, done))

            state = next_state
            episode_return += reward
            timestep += 1

        stepwise_returns = self.calculate_stepwise_returns(rewards)
        log_prob_actions = torch.stack(log_prob_actions)
        return episode_return, stepwise_returns, log_prob_actions, trajectory,timestep
    
    def calculate_loss(self, stepwise_returns, log_prob_actions):
        """
        Compute the loss for policy optimization.

        Args:
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        return -torch.sum(log_prob_actions * stepwise_returns)
        # ====================================== #

    def update_policy(self, stepwise_returns, log_prob_actions):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # ====================================== #
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory ,timestep = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, loss, trajectory,timestep
        # ====================================== #


    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'n_observations': self.policy_net.fc1.in_features,
                'hidden_size': self.policy_net.fc1.out_features,
                'n_actions': self.policy_net.output.out_features,
                'dropout': self.policy_net.dropout.p
            }
        }, path)

    def load_model(self,cls, path, device=None, **kwargs):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        model = cls(
            device=device,
            n_observations=config['n_observations'],
            hidden_dim=config['hidden_size'],
            num_of_action=config['n_actions'],
            dropout=config['dropout'],
            **kwargs  # any other optional args
        )

        model.policy_net.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model