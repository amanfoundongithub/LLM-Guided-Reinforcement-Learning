import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from minigrid.wrappers import ImgObsWrapper
from utils.constants import env, max_episodes, max_steps, name_of_environment
from utils.llm_suggestion import suggest_llm_distribution
from agents.PPOAgent import PPOAgent
from agents.QLAgent import QLearningAgent

# Initialize environment and agent
env = gym.make('MiniGrid-KeyCorridorS3R1-v0', render_mode='rgb_array')
env = ImgObsWrapper(env)
obs_shape = env.observation_space.shape
action_dim = env.action_space.n
# agent = PPOAgent(obs_shape, action_dim)
agent = QLearningAgent(obs_shape, action_dim) 

# Training loop
for episode in range(max_episodes):
    obs, _ = env.reset()
    obs = obs.astype(np.float32)
    total_reward = 0

    for step in range(max_steps):
        # Action selection 
        action = agent.get_action(obs) 
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs = next_obs.astype(np.float32)

        # Store transition and update
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.update()

        obs = next_obs
        total_reward += reward

        if done:
            break

    # Print episode statistics
    print(f"Episode {episode} | Reward: {total_reward:.2f}")

env.close()
