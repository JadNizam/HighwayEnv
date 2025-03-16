import numpy as np
from __future__ import annotations

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from gymnasium import spaces

class MultiAgentIntersectionEnv(AbstractEnv):
    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "MultiAgentAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "duration": 30,  # Multi-agent episode duration [s]
                "destination": "o1",
                "controlled_vehicles": 4,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -100,
                "high_speed_reward": 1,
                "arrived_reward": 5,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def __init__(self, config=None):
        super().__init__(config)
        self.num_agents = self.config.get("controlled_vehicles", 4)
        self.action_space = spaces.Tuple([spaces.Discrete(3)] * self.num_agents)
        self.observation_space = spaces.Tuple([spaces.Box(-np.inf, np.inf, shape=(15,))] * self.num_agents)
        self.agents = []
        self.episode_time = self.config.get("duration", 30)
        self.time_elapsed = 0
        self._init_multi_agents()
    
    def _init_multi_agents(self):
        """Initialize multiple agent-controlled vehicles."""
        self.agents = []
        for _ in range(self.num_agents):
            vehicle = self._spawn_vehicle()
            self.agents.append(vehicle)
    
    def _spawn_vehicle(self):
        """Spawn a new controlled vehicle in a valid location."""
        vehicle = self.action_type.vehicle_class.create_random(self.road, speed=10)
        self.road.vehicles.append(vehicle)
        return vehicle
    
    def step(self, actions):
        """Apply the given actions to each agent and step the environment."""
        for agent, action in zip(self.agents, actions):
            agent.act(action)
        
        self.road.act()
        self.road.step(self.config["simulation_frequency"])
        self.time_elapsed += 1 / self.config["simulation_frequency"]
        
        observations = self._get_observations()
        rewards, done = self._get_rewards_and_done()
        
        # End episode if time exceeds duration or all agents leave the intersection
        if self.time_elapsed >= self.episode_time or all(agent not in self.road.vehicles for agent in self.agents):
            done = True
        
        return observations, rewards, done, {}
    
    def _get_observations(self):
        """Get observations for all agents."""
        return tuple(self.observation_type.observe(agent) for agent in self.agents)
    
    def _get_rewards_and_done(self):
        """Compute rewards and done flags for all agents."""
        rewards = []
        done = False
        for agent in self.agents:
            reward = self._compute_reward(agent)
            rewards.append(reward)
            
            if agent.crashed:
                done = True
        
        return tuple(rewards), done
    
    def _compute_reward(self, agent):
        """Calculate reward for an agent based on collision avoidance and wait time minimization."""
        reward = 0
        if agent.crashed:
            reward -= 100  # Large penalty for collisions
        elif self.has_arrived(agent):
            reward += 5  # Reward for reaching destination
        else:
            reward += 1  # Small reward for staying active
            reward -= 0.1 * agent.speed  # Encourage maintaining higher speed
        return reward
    
    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )
