import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

from utils.constants import name_of_environment, env, lr, gamma 
from utils.llm_suggestion import suggest_llm_action

class SACNetworkCNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(SACNetworkCNN, self).__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate feature size
        feature_size = self._get_feature_size(obs_shape)
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU()
        )
        
        # Mean and log_std outputs for each action
        # For discrete actions, we'll use these to parameterize a categorical distribution
        self.mean = nn.Linear(512, action_dim)
        self.log_std = nn.Linear(512, action_dim)
        
        # Twin Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
    def _get_feature_size(self, obs_shape):
        # Forward pass with dummy input to get feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            return self.features(dummy_input).shape[1]
    
    def forward(self, x):
        features = self.features(x)
        # Actor outputs for discrete action probabilities
        actor_features = self.actor(features)
        mean = self.mean(actor_features)
        log_std = self.log_std(actor_features)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values
        
        # Q-values
        q1 = self.q1(features)
        q2 = self.q2(features)
        
        return mean, log_std, q1, q2
    
    def get_action_logprob(self, x):
        mean, log_std, _, _ = self(x)
        
        # For discrete actions, we use softmax to get probabilities
        action_probs = F.softmax(mean, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_q_values(self, x):
        features = self.features(x)
        q1 = self.q1(features)
        q2 = self.q2(features)
        return q1, q2

class SACAgent:
    def __init__(self, obs_shape, action_dim, alpha=0.2, tau=0.005,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.network = SACNetworkCNN(obs_shape, action_dim)
        self.target_network = SACNetworkCNN(obs_shape, action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.q_optimizer = optim.Adam(list(self.network.q1.parameters()) + 
                                      list(self.network.q2.parameters()) +
                                      list(self.network.features.parameters()), lr=lr)
        
        self.policy_optimizer = optim.Adam(list(self.network.actor.parameters()) +
                                          list(self.network.mean.parameters()) +
                                          list(self.network.log_std.parameters()), lr=lr)
        
        self.alpha = alpha  # Entropy regularization coefficient
        self.tau = tau      # Soft update parameter
        
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        
        # Replay buffer
        self.memory = []
        self.batch_size = 64
        self.memory_size = 10000
    
    def get_action(self, obs):
        # Epsilon decay
        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon = 1  # As specified in original code
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # return random.randint(0, self.action_dim - 1)
            action = suggest_llm_action(env, name_of_environment, env.action_space.n, obs) 
            return action 
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None], dtype=torch.float32) / 10.0
                mean, _, q1, _ = self.network(obs_tensor)
                # Use action with highest Q-value for deterministic policy
                return q1.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        # Store in memory, remove oldest if full
        transition = (state, action, reward, next_state, done)
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def update(self):
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32) / 10.0
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32) / 10.0
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # Calculate Q-values
        with torch.no_grad():
            next_mean, next_log_std, next_q1_target, next_q2_target = self.target_network(next_states)
            next_action_probs = F.softmax(next_mean, dim=-1)
            
            # Calculate entropy
            log_probs = torch.log(next_action_probs + 1e-10)  # Add small epsilon for numerical stability
            entropy = -torch.sum(next_action_probs * log_probs, dim=1, keepdim=True)
            
            # Calculate expected Q values for each action weighted by its probability
            next_q1 = torch.sum(next_q1_target * next_action_probs, dim=1, keepdim=True)
            next_q2 = torch.sum(next_q2_target * next_action_probs, dim=1, keepdim=True)
            next_q = torch.min(next_q1, next_q2) + self.alpha * entropy
            
            # Compute target Q value
            target_q = rewards + (1 - dones) * gamma * next_q
        
        # Get current Q values
        _, _, q1, q2 = self.network(states)
        current_q1 = q1.gather(1, actions.unsqueeze(1))
        current_q2 = q2.gather(1, actions.unsqueeze(1))
        
        # Q-network loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # Update Q-networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Policy loss
        mean, log_std, new_q1, new_q2 = self.network(states)
        action_probs = F.softmax(mean, dim=-1)
        log_probs = torch.log(action_probs + 1e-10)
        
        # Entropy
        entropy = -torch.sum(action_probs * log_probs, dim=1, keepdim=True)
        
        # Expected Q-value under current policy
        expected_q = torch.sum(torch.min(new_q1, new_q2) * action_probs, dim=1, keepdim=True)
        
        # Policy loss: maximize expected Q-value and entropy
        policy_loss = -(expected_q + self.alpha * entropy).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update of target network
        self._soft_update()
    
    def _soft_update(self):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)