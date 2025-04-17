from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import os
import json
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import wandb

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input state tensor.
        
        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x
        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 1000,
            batch_size: int = 1,
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
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device = device
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.episode_durations = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        state = state.to(self.device).unsqueeze(0)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            # Random action
            return random.randrange(self.num_of_action)
        else:
            # Greedy action from policy_net
             with torch.no_grad():
                q_values = self.policy_net(state)  # shape: [1, num_actions]
                return q_values[0].argmax().item()  
            
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken (expected shape: [batch_size, 1]).
            reward_batch (Tensor): Batch of received rewards (expected shape: [batch_size, 1]).

        Returns:
            Tensor: Computed MSE loss.
        """
        # Move all tensors to device
        state_batch = state_batch.to(self.device).squeeze(1)
        non_final_next_states = non_final_next_states.to(self.device).squeeze(1)
        action_batch = action_batch.to(self.device).squeeze(-1).unsqueeze(1)
        reward_batch = reward_batch.to(self.device)

        # 🛠️ Ensure action_batch shape: [batch_size, 1] and dtype: int64
        # action_batch = action_batch.view(-1, 1).long()

        # # 🧯 Safety checks
        # assert action_batch.max() < self.num_of_action, f"[ERROR] action index too large: {action_batch.max().item()} >= {self.num_of_action}"
        # assert action_batch.min() >= 0, f"[ERROR] negative action index: {action_batch.min().item()}"

        if action_batch.ndim == 3:
            action_batch = action_batch.squeeze(-1)  # from [B, 1, 1] to [B, 1]
        if action_batch.ndim == 1:
            action_batch = action_batch.unsqueeze(1)  # from [B] to [B, 1]

        action_batch = action_batch.long()

        # 1️⃣ Get predicted Q(s, a) from policy_net
        q_values = self.policy_net(state_batch)  # [B, num_actions]

        # FIX ACTION SHAPE: squeeze to [B, 1]
        action_batch = action_batch.squeeze(-1)

        # Sanity checks
        assert q_values.dim() == 2, f"q_values shape: {q_values.shape}"
        assert action_batch.dim() == 2 or action_batch.dim() == 1, f"action_batch shape: {action_batch.shape}"

        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)  # [B] → [B, 1]

        state_action_values = q_values.gather(1, action_batch)  # Shape: [batch_size, 1]

        # 2️⃣ Get max Q(s', a') from target_net (for next states)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            max_next_q = self.target_net(non_final_next_states).max(dim=1)[0]
            next_state_values[non_final_mask] = max_next_q

        # 3️⃣ Compute expected Q value: r + γ * max Q(s', a')
        expected_state_action_values = reward_batch + self.discount_factor * next_state_values.unsqueeze(1)

        # 4️⃣ MSE Loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        return loss
        # ====================================== #

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor)
                - non_final_next_states (Tensor)
                - state_batch (Tensor)
                - action_batch (Tensor)
                - reward_batch (Tensor)
        """
        
        if len(self.memory) < batch_size:
            return None

        
        batch = self.memory.sample()

        # Unpack directly
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        # done == 1.0 → terminal
        # done == 0.0 → non-terminal
        non_final_mask = (done_batch == 0.0)

        

        # Only get next_states that are non-terminal
        non_final_next_states = next_state_batch[non_final_mask]

        action_batch = action_batch.unsqueeze(1)
        
        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # Generate a sample batch
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        
        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # Perform gradient descent step
        # ========= put your code here ========= #
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return None
        
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        
        loss = self.calculate_loss(non_final_mask, non_final_next_states,
                                   state_batch, action_batch, reward_batch)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Optionally clip gradients here if you want to prevent exploding grad
        self.optimizer.step()
        
        return loss.item()
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        # policy_state_dict = self.policy_net.state_dict()
        # target_state_dict = self.target_net.state_dict()
        # # ====================================== #
        
        # # Apply the soft update rule to each parameter in the target network
        # # ========= put your code here ========= #
        # for key in policy_state_dict:
        #     policy_param = policy_state_dict[key]
        #     target_param = target_state_dict[key]
        #     # Polyak: new_target = (1 - tau) * old_target + tau * policy
        #     updated_param = (1.0 - self.tau) * target_param + self.tau * policy_param
        #     target_state_dict[key] = updated_param
        # # ====================================== #
        
        # # Load the updated weights into the target network
        # # ========= put your code here ========= #
        # self.target_net.load_state_dict(target_state_dict)
        # ====================================== #
        
        # ------------------------------------------------------------------------

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * policy_param.data
            )
            
        # ------------------------------------------------------------------------

    def learn(self, env):
        """
        Train the agent on a single step.

        Args:
            env: The environment to train in.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        # ---------------------------------------------------------------------------------------------------------------
        # pass
        # # ====================================== #

        # while not done:
        #     # Predict action from the policy network
        #     # ========= put your code here ========= #
        #     pass
        #     # ====================================== #

        #     # Execute action in the environment and observe next state and reward
        #     # ========= put your code here ========= #
        #     pass
        #     # ====================================== #

        #     # Store the transition in memory
        #     # ========= put your code here ========= #
        #     pass
        #     # ====================================== #

        #     # Update state

        #     # Perform one step of the optimization (on the policy network)
        #     self.update_policy()

        #     # Soft update of the target network's weights
        #     self.update_target_networks()

        #     timestep += 1
        #     if done:
        #         self.plot_durations(timestep)
        #         break
        
        # ---------------------------------------------------------------------------------------------------------------
        state, _ = env.reset()
        state = torch.as_tensor(state["policy"], dtype=torch.float32, device=self.device)
        reward = 0
        episode_return = 0.0
        done = False
        timestep = 0
        loss_val = 0.0

        while not done:
            # 1. Select action
            action_idx = self.select_action(state)
            action_tensor = torch.tensor([[action_idx]], dtype=torch.float32, device=self.device)

            # 2. Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated
            next_state = torch.as_tensor(next_obs["policy"], dtype=torch.float32, device=self.device)

            # 3. Prepare tensors
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            action_tensor_long = torch.tensor([[action_idx]], dtype=torch.int64, device=self.device)
            done_tensor = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=self.device)

            # 4. Store in replay buffer
            self.memory.add(
                state.unsqueeze(0),
                action_tensor_long,
                reward_tensor.unsqueeze(0),
                next_state.unsqueeze(0),
                done_tensor
            )

            # 5. Learn and update target net

            loss = self.update_policy()
            if loss is not None:
                loss_val += float(loss)
            # self.update_policy()
            
            self.update_target_networks()

            # Move to next state
        
            state = next_state
            episode_return += reward
            timestep += 1

        
        wandb.log({"loss function": loss_val})
        
        self.sum_count += timestep
        self.reward_sum += episode_return
            # if done:
                # self.plot_durations(timestep)
                # break




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

    def save_w_DQN(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        
        torch.save(self.policy_net.state_dict(), path)
        # with open(os.path.join(path, filename), 'w') as f:
        #     json.dump(w_list, f)

    def load_w_DQN(self, path, filename):
        """
        Load the entire policy network model (state_dict).
        
        Args:
            path (str): Directory where the model is saved.
            filename (str): Name of the saved model file.
        """
        file_path = os.path.join(path, filename)
        self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.policy_net.to(self.device)
        self.policy_net.eval()  # Optional: switch to eval mode if you're done training
        print(f"Model loaded from {file_path}")
    