import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

from utils.constants import name_of_environment, env, lr, gamma
from utils.llama import suggest_llm_distribution



class ActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        # Shared conv‐feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # figure out flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            feat_size = self.features(dummy).shape[1]

        # policy head
        self.actor = nn.Sequential(
            nn.Linear(feat_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        # value head
        self.critic = nn.Sequential(
            nn.Linear(feat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        feats = self.features(x)
        return self.actor(feats), self.critic(feats)

    def get_dist_and_value(self, x):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        return dist, value

class PPOAgent:
    def __init__(self,obs_shape, action_dim,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lambda_gae=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    ):
        self.net = ActorCriticCNN(obs_shape, action_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lambda_gae = lambda_gae

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # rollout buffers
        self.states, self.actions = [], []
        self.log_probs, self.values = [], []
        self.rewards, self.dones = [], []

        self.action_dim = action_dim
    def get_action(self, obs):
        # epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # choose between LLM-guided vs. policy
        if random.random() < self.epsilon:
            # LLM gives us a prob distribution
            probs = suggest_llm_distribution(env, name_of_environment, self.action_dim, obs)
            dist = Categorical(probs=torch.tensor(probs, dtype=torch.float32))
        else:
            # use our learned policy
            obs_t = torch.tensor(obs[None], dtype=torch.float32) / 10.0
            dist, _ = self.net.get_dist_and_value(obs_t)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        # get value from critic
        with torch.no_grad():
            # Get policy value 
            _, value = self.net.get_dist_and_value(torch.tensor(obs[None], dtype=torch.float32) / 10.0)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    def compute_advantages(self):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [0])  # bootstrap last value=0
        dones = np.array(self.dones)

        gae = 0
        advantages = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t+1] * mask - values[t]
            gae = delta + gamma * self.lambda_gae * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns
    def update(self, epochs=4, batch_size=64):
        advs, rets = self.compute_advantages()

        # tensorify
        S = torch.tensor(np.array(self.states), dtype=torch.float32) / 10.0
        A = torch.tensor(self.actions, dtype=torch.long)
        LP = torch.tensor(self.log_probs, dtype=torch.float32)
        ADV = torch.tensor(advs, dtype=torch.float32)
        RET = torch.tensor(rets, dtype=torch.float32)

        # normalize
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        for _ in range(epochs):
            idxs = np.random.permutation(len(S))
            for start in range(0, len(S), batch_size):
                b = idxs[start:start+batch_size]
                dist, V = self.net.get_dist_and_value(S[b])
                new_lp = dist.log_prob(A[b])
                entropy = dist.entropy().mean()

                # ratios
                ratio = (new_lp - LP[b]).exp()
                # surrogate
                obj = torch.min(
                    ratio * ADV[b],
                    torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * ADV[b]
                ).mean()
                policy_loss = -obj
                value_loss = F.mse_loss(V.squeeze(-1), RET[b])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

        # clear rollout
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
