import os
import time
import torch
import numpy as np
from agilerl.algorithms.maddpg import MADDPG
from agilerl.utils.utils import create_population
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv

# Instantiate the environment directly.
env = MultiAgentIntersectionEnv(config={"controlled_vehicles": 2})
env.render_mode = "human"  # Explicitly set render mode
env.reset()

# Get agent/environment dimensions
num_agents = env.config["controlled_vehicles"]
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

# Create agent population (same as training, but only need one per agent)
agents = create_population(
    algo=MADDPG,
    observation_space=env.observation_space,
    action_space=env.action_space,
    net_config={"arch": "mlp", "h_size": [128, 128]},
    INIT_HP={
        "POPULATION_SIZE": num_agents,
        "LR": 0.001,
        "GAMMA": 0.99,
        "TAU": 0.01,
        "BATCH_SIZE": 64,
        "DEVICE": torch.device("cpu")
    }
)

# Load trained weights for each agent, preferring "best" models if available.
for i, agent in enumerate(agents):
    # Get all model files for agent_i.
    model_files = [f for f in os.listdir("saved_models") if f"agent_{i}" in f]
    assert model_files, f"No model found for agent_{i} in saved_models/"
    # Filter for best models if available.
    best_models = [f for f in model_files if "best" in f]
    if best_models:
        chosen_model = sorted(best_models)[-1]
    else:
        chosen_model = sorted(model_files)[-1]
    agent.load(os.path.join("saved_models", chosen_model))
    agent.set_env(env)
    agent.set_id(f"agent_{i}")

# Run visual test episodes
num_episodes = 10

for episode in range(num_episodes):
    states = env.reset()
    done = False
    total_reward = 0
    print(f"\n▶️ Episode {episode + 1}")

    while not done:
        env.render()           # Display simulation window.
        time.sleep(0.3)        # Slow down simulation for better visualization.

        actions = [agent.select_action(states[i], epsilon=0.0) for i, agent in enumerate(agents)]
        next_states, rewards, done, info = env.step(tuple(actions))
        states = next_states
        total_reward += rewards  # Accumulate scalar reward

    print(f"✅ Episode {episode + 1} Total Reward: {total_reward:.2f}")

env.close()
