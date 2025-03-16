import torch
import numpy as np
import gymnasium as gym
import time
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv
from agilerl.algorithms.maddpg import MADDPG

# Load the environment
env = MultiAgentIntersectionEnv(config={})
num_agents = env.num_agents

# Load trained MADDPG model
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
hidden_dim = 128
lr = 0.001
gamma = 0.99
tau = 0.01
buffer_size = 100000
batch_size = 64

agents = MADDPG(
    num_agents=num_agents,
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=hidden_dim,
    lr=lr,
    gamma=gamma,
    tau=tau,
    buffer_size=buffer_size,
    batch_size=batch_size,
)

agents.load("maddpg_multiagent.pth")

# Run simulation
num_episodes = 10
for episode in range(num_episodes):
    states = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        env.render()  # Display the simulation
        
        actions = [agents.select_action(states[i], explore=False) for i in range(num_agents)]
        next_states, rewards, done, _ = env.step(actions)
        states = next_states
        episode_reward += sum(rewards)
        
        time.sleep(0.05)  # Slow down simulation for visualization
    
    print(f"Episode {episode}, Total Reward: {episode_reward}")

env.close()
print("Testing complete.")
