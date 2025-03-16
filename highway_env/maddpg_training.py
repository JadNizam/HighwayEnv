import torch
import numpy as np
import gymnasium as gym
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv
from agilerl.algorithms.maddpg import MADDPG
from agilerl.utils.memory import MultiAgentReplayBuffer

# Initialize the environment
env = MultiAgentIntersectionEnv(config={})
num_agents = env.num_agents

# Hyperparameters
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
hidden_dim = 128
lr = 0.001
gamma = 0.99
tau = 0.01
buffer_size = 100000
batch_size = 64

# Initialize MADDPG Agents
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

# Initialize Replay Buffer
memory = MultiAgentReplayBuffer(
    buffer_size=buffer_size,
    state_dim=state_dim,
    action_dim=action_dim,
    num_agents=num_agents,
)

# Training Loop
num_episodes = 10000
eps_decay = 0.995
epsilon = 1.0
min_epsilon = 0.05

for episode in range(num_episodes):
    states = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        actions = [agents.select_action(states[i], epsilon) for i in range(num_agents)]
        next_states, rewards, done, _ = env.step(actions)
        
        memory.add(states, actions, rewards, next_states, done)
        states = next_states
        episode_reward += sum(rewards)
        
        if len(memory) > batch_size:
            agents.learn(memory)
    
    epsilon = max(min_epsilon, epsilon * eps_decay)
    print(f"Episode {episode}, Total Reward: {episode_reward}")

    if episode % 100 == 0:
        agents.save("maddpg_multiagent.pth")

print("Training complete. Model saved as maddpg_multiagent.pth")
env.close()
