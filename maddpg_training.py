import os
import torch
import numpy as np
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.utils.utils import create_population
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv
from tqdm import trange

# ✅ Force CPU usage
device = torch.device("cpu")

# ✅ Initialize Multi-Agent Environment
env = MultiAgentIntersectionEnv(config={})
env.reset()

# ✅ Define Agent Hyperparameters
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
hidden_dim = 128
lr = 0.001
gamma = 0.99
tau = 0.01
buffer_size = 100000
batch_size = 64

# ✅ Agent IDs
num_agents = env.config["controlled_vehicles"]
agent_ids = [f"agent_{i}" for i in range(num_agents)]

# ✅ Initialize Replay Buffer
memory = MultiAgentReplayBuffer(
    memory_size=buffer_size,
    field_names=["state", "action", "reward", "next_state", "done"],
    agent_ids=agent_ids,
    device=device
)

# ✅ Create Initial Agent Population using AgileRL latest API
agents = create_population(
    algo=MADDPG,
    observation_space=env.observation_space,
    action_space=env.action_space,
    net_config={"arch": "mlp", "h_size": [hidden_dim, hidden_dim]},
    INIT_HP={
        "POPULATION_SIZE": num_agents,
        "LR": lr,
        "GAMMA": gamma,
        "TAU": tau,
        "BATCH_SIZE": batch_size,
        "DEVICE": device
    }
)

# ✅ Assign environment, memory, and agent ID
for i, agent in enumerate(agents):
    agent.set_env(env)
    agent.set_replay_buffer(memory)
    agent.set_id(agent_ids[i])

# ✅ Directory for saving models
os.makedirs("saved_models", exist_ok=True)

# ✅ Training Loop parameters
num_episodes = 10000  # Increased number of episodes
eps_decay = 0.999
epsilon = 1.0
min_epsilon = 0.05

best_reward = -float('inf')  # To track the best episode reward

for episode in trange(num_episodes, desc="Training Progress"):
    states = env.reset()
    done = False
    episode_reward = 0

    while not done:
        actions = [agent.select_action(states[i], epsilon) for i, agent in enumerate(agents)]
        next_states, rewards, done, info = env.step(tuple(actions))
        memory.save_to_memory(states, actions, rewards, next_states, done)
        states = next_states

        # Sum agent rewards from info (assuming info["agents_rewards"] is a tuple of rewards)
        episode_reward += sum(info["agents_rewards"])

        if len(memory) > batch_size:
            for agent in agents:
                agent.learn()

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * eps_decay)

    # Print reward for the episode
    print(f"Episode {episode}, Total Reward: {episode_reward}")

    # If a new best is achieved, save the models
    if episode_reward > best_reward:
        best_reward = episode_reward
        for i, agent in enumerate(agents):
            agent.save(f"saved_models/maddpg_agent_{i}_best.pth")

    # Also periodically save every 100 episodes
    if episode % 100 == 0:
        for i, agent in enumerate(agents):
            agent.save(f"saved_models/maddpg_agent_{i}_ep{episode}.pth")

print("✅ Training Complete. Models saved to /saved_models/")
env.close()
