import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm2
import wandb
import os

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, output_dim)  # Mean of the action distribution
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # Log standard deviation
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Normal: Distribution of actions.
        """
        x = self.net(state)
        mu = self.mu(x)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for state value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single value
        )
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for value estimation.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Estimated state value.
        """
        return self.net(state).squeeze(-1)

class Actor_Critic(BaseAlgorithm2):
    def __init__(self, 
                device = None, 
                num_of_action: int = 1,  # Changed from 2 to 1 to match environment
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 1.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        self.count = 0
        self.sum_count = 0
        self.reward_sum = 0
        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor
        self.clip_param = 0.2  # PPO clipping parameter

        # Initialize target networks with source network parameters
        self.update_target_networks(tau=1)

        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def select_action(self, state, noise=0.0):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - action: The selected action.
                - log_prob: Log probability of the selected action.
        """
        state = state.to(self.device)
        dist = self.actor(state)
        action = dist.sample()
        
        # Clamp action to allowed range
        action = torch.clamp(action, self.action_range[0], self.action_range[1])
        
        # Calculate log probability of the action
        log_prob = dist.log_prob(action).sum(-1)
        
        return action.detach(), log_prob.detach()
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing state, action, reward, next_state, done batches.
        """
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample()
        state, action, reward, next_state, done, log_prob = batch
        return state, action, reward, next_state, done, log_prob
    
    def calculate_loss(self, states, actions, rewards, next_states, dones, old_log_probs):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            returns = rewards + self.discount_factor * next_values * (1 - dones)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = self.actor(states)
        new_log_probs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(states)
        critic_loss = mse_loss(values, returns)

        return actor_loss, critic_loss

    def update_policy(self):
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return

        states, actions, rewards, next_states, dones, old_log_probs = sample
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        actor_loss, critic_loss = self.calculate_loss(states, actions, rewards, next_states, dones, old_log_probs)

        wandb.log({
            "PPO Actor loss function": actor_loss.item(),
            "PPO Critic loss function": critic_loss.item()
        })

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
        """
        Learning loop for the agent.
        
        Args:
            env: The environment to interact with
            max_steps: Maximum number of steps per episode
            num_agents: Number of agents in the environment
            noise_scale: Initial noise scale for exploration
            noise_decay: Decay rate for exploration noise
            
        Returns:
            float: Episode return (cumulative reward)
        """
        state, _ = env.reset()
        
        # Handle state properly - ensure it's on the right device
        if isinstance(state["policy"], torch.Tensor):
            state = state["policy"].to(self.device)
        else:
            state = torch.tensor(state["policy"], dtype=torch.float32, device=self.device)
            
        episode_return = 0
        done = False
        self.count = 0
        log_probs = []

        for step in range(max_steps):
            # Select action and get log prob
            action, log_prob = self.select_action(state, noise=noise_scale)
            
            # Fix the action shape for the environment
            # The environment is expecting a 2D tensor with shape [batch_size, action_dim]
            # So we reshape to [1, 1] for a single action
            action_env = action.detach().reshape(1, -1)[:, :1]
            
            # Step environment with the properly shaped tensor
            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            
            done = terminated or truncated
            
            # Handle next_state the same way as state
            if isinstance(next_obs["policy"], torch.Tensor):
                next_state = next_obs["policy"].to(self.device)
            else:
                next_state = torch.tensor(next_obs["policy"], dtype=torch.float32, device=self.device)

            # Convert reward and done to tensors
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            done_tensor = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

            # Store transition in memory
            self.memory.add(
            state.unsqueeze(0),
            action.unsqueeze(0),
            reward_tensor.unsqueeze(0),
            next_state.unsqueeze(0),
            done_tensor.unsqueeze(0),
            log_prob.unsqueeze(0)
            )
            log_probs.append(log_prob)

            state = next_state
            episode_return += reward

            # Decrease exploration noise over time
            noise_scale *= noise_decay
            self.count += 1
            
            if done:
                break

        # Store log_probs for PPO update
        self.old_log_probs = torch.stack(log_probs).detach()
        self.sum_count += self.count

        # Update policy and target networks
        self.update_policy()
        self.update_target_networks()
        self.reward_sum += episode_return

        return episode_return
    
    def save_model(self, path, filename):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor.optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic.optimizer.state_dict(),
            # Include any additional information you want to save
        }
        torch.save(checkpoint, os.path.join(path, filename))
        print(f"Model saved to {os.path.join(path, filename)}")

    def load_model(self, path, filename):
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {os.path.join(path, filename)}")