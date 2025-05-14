import gymnasium as gym


# Hyperparameters
lr = 3e-4
gamma = 0.99
max_episodes = 10000
max_steps = 500

name_of_environment = 'MiniGrid-DoorKey-6x6-v0'

env = gym.make(name_of_environment, render_mode='rgb_array', agent_view_size=3)