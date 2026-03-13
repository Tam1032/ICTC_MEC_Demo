import gymnasium as gym
import numpy as np
import copy
from .gym_environment import Environment as MECEnvBase


class MECDualTimeScaleEnv:
    """
    This class represents the core MEC environment for dual-timescale control.
    It is not a gym.Env itself but is managed by the wrapper environments.
    It adapts the original MEC_Environment to a dual-timescale framework.
    - Slow timescale: Caching decisions.
    - Fast timescale: Offloading and early-exit decisions for a batch of tasks.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the base MEC environment.
        Accepts the same arguments as the original MEC_Environment.
        """
        self.base_env = MECEnvBase(*args, **kwargs)
        self.K = self.base_env.num_devices

        # Actions from agents
        self.slow_actions = None  # Caching decisions
        self.fast_actions = None  # Offloading & exit decisions
        
        # Slow timescale counter
        self.slow_step_count = 0
        self.cumulative_fast_reward = 0.0

    def reset(self):
        """
        Resets the state of the environment for a new slow episode.
        This involves resetting the base environment.
        """
        obs, _ = self.base_env.reset()
        self.cumulative_fast_reward = 0.0
        return self._get_slow_obs()

    def _get_slow_obs(self):
        """
        Returns the observation for the slow agent (caching).
        The slow agent observes the model popularity and current cache status.
        """
        # Calculate model popularity from the base environment's history
        if self.base_env.popularity_window:
            model_popularity = np.sum(self.base_env.popularity_window, axis=0)
        else:
            model_popularity = np.zeros(self.base_env.num_models, dtype=np.float32)
        
        # Normalize model popularity to [0, 1] (probability distribution)
        total_popularity = np.sum(model_popularity)
        if total_popularity > 0:
            model_popularity_normalized = model_popularity / total_popularity
        else:
            model_popularity_normalized = model_popularity
        
        if self.base_env.caching_gain_window:
            model_caching_gain = np.average(self.base_env.caching_gain_window, axis=0)
        else:
            model_caching_gain = np.zeros(self.base_env.num_models, dtype=np.float32)
        
        edge_popularity = np.array([edge.model_request_counts for edge in self.base_env.edge_servers], dtype=np.float32)
        # Normalize edge popularity row-wise (per edge)
        row_sums = edge_popularity.sum(axis=1, keepdims=True)
        edge_popularity = np.divide(edge_popularity, row_sums, out=np.zeros_like(edge_popularity), where=row_sums!=0)

        # Normalize sizes and capacities by the maximum storage capacity to keep them in [0, 1]
        max_capacity = np.max([edge.cache_storage for edge in self.base_env.edge_servers])
        # Avoid division by zero if max_capacity is 0 (unlikely but safe)
        if max_capacity == 0:
            max_capacity = 1.0
        edge_cache_storage = self.base_env.edge_servers[0].cache_storage
        model_sizes_normalized = np.array([model.model_size / edge_cache_storage for model in self.base_env.task_types], dtype=np.float32)

        # Normalize slow timescale counter
        # Assuming max episode length of 10 timesteps, normalize to [0, 1]
        slow_timestep = np.array([(self.slow_step_count % 10) / 10.0], dtype=np.float32)

        # 2. Edge computing capacity (per edge) - Normalized by max computing power
        edge_computing_capacity = np.array([edge.core_computing_power for edge in self.base_env.edge_servers], dtype=np.float32)
        max_computing = max(edge_computing_capacity) if len(edge_computing_capacity) > 0 else 1.0
        edge_computing_capacity /= max_computing if max_computing > 0 else edge_computing_capacity
        
        # 2.5. Cloud computing capacity - Normalized by max computing power
        cloud_computing_capacity = np.array([self.base_env.cloud_server.computing_power / max_computing], dtype=np.float32)

        # 3. Connected edge servers (binary matrix: num_edges x num_devices)
        connected_edge_servers = np.zeros((self.base_env.num_edges, self.base_env.num_devices), dtype=np.float32)
        for dev_id, dev in enumerate(self.base_env.mobile_devices):
            connected_edge_servers[dev.edge_id, dev_id] = 1.0

        # 4. Model info (num_blocks, compute, accuracy) - Per Model
        num_blocks = np.array([model.num_blocks for model in self.base_env.task_types], dtype=np.float32)
        num_blocks /= 5.0  # Max blocks is 5 in generate_blocks

        block_compute_requirement = np.zeros((self.base_env.num_models, 5), dtype=np.float32)
        block_accuracy = np.zeros((self.base_env.num_models, 5), dtype=np.float32)
        for i, model in enumerate(self.base_env.task_types):
            comp = [b.compute_requirement for b in model.blocks]
            acc = [b.accuracy for b in model.blocks]
            block_compute_requirement[i, :len(comp)] = comp
            block_accuracy[i, :len(acc)] = acc
        
        block_compute_requirement /= max_computing  # Normalize by 1 TFLOPS

        # 5. Edge core completion times - Normalized by 1e6 (as in gym env)
        # edge_core_completion_times = np.array([
        #     np.array(edge.core_finish_times, dtype=np.float32)
        #     for edge in self.base_env.edge_servers
        # ], dtype=np.float32)
        # edge_core_completion_times /= 1e6

        # Build nearby edge servers matrix (num_edges x num_edges)
        # Each row contains information about distances to other edge servers
        nearby_edges = np.zeros((self.base_env.num_edges, self.base_env.num_edges), dtype=np.float32)
        for edge_i in range(self.base_env.num_edges):
            for edge_j in range(self.base_env.num_edges):
                if edge_i == edge_j:
                    nearby_edges[edge_i, edge_j] = 0.0  # Same edge
                else:
                    # Get hop distance from edge_connections
                    hops = self.base_env.edge_connections.get((edge_i, edge_j), np.inf)
                    # Normalize hops: closer edges (fewer hops) get higher values
                    # Using 1 / (1 + hops) to map hops to [0, 1)
                    if hops == np.inf:
                        nearby_edges[edge_i, edge_j] = 0.0
                    else:
                        nearby_edges[edge_i, edge_j] = 1.0 / (1.0 + hops)

        return {
            "model_caching_gain": model_caching_gain.astype(np.float32),
            "model_popularity": model_popularity_normalized.astype(np.float32),
            "edge_popularity": edge_popularity.astype(np.float32),
            "model_sizes": model_sizes_normalized.astype(np.float32),
            "slow_timestep": slow_timestep,
            # "nearby_edges": nearby_edges.astype(np.float32),
            # "edge_computing_capacity": edge_computing_capacity.astype(np.float32),
            # "cloud_computing_capacity": cloud_computing_capacity.astype(np.float32),
            # "connected_edge_servers": connected_edge_servers.astype(np.float32),
            # "num_blocks": num_blocks.astype(np.float32),
            # "block_compute_requirement": block_compute_requirement.astype(np.float32),
            # "block_accuracy": block_accuracy.astype(np.float32),
            #"edge_core_completion_times": edge_core_completion_times.astype(np.float32),
        }

    def _get_fast_obs(self):
        """
        Returns the observation for the fast agent (offloading/exit).
        This is the standard observation from the base environment.
        """
        return self.base_env._get_obs(self.base_env.pending_tasks)

    def step_slow(self, caching_actions):
        """
        The slow agent takes an action (caching decisions).
        This sets the cache state for the upcoming fast episode.

        Args:
            caching_actions (np.array): A binary array of shape (num_edges, num_models).
                                        1 means cache, 0 means do not.
        """
        self.slow_actions = caching_actions
        self.cumulative_fast_reward = 0.0
        # If cache_oblivious is True, use random caching instead of agent decisions
        # if self.base_env.cache_oblivious:
        #     cache_decisions = self.base_env.cache_random_models()
        #     # cache_decisions is a flat list for all edge servers
        #     for edge_id, edge_server in enumerate(self.base_env.edge_servers):
        #         start = edge_id * self.base_env.num_models
        #         end = (edge_id + 1) * self.base_env.num_models
        #         cached_models = []
        #         used_storage = 0
        #         for model_idx, should_cache in enumerate(cache_decisions[start:end]):
        #             model_size = self.base_env.task_types[model_idx].model_size
        #             if should_cache and (used_storage + model_size <= edge_server.cache_storage):
        #                 cached_models.append(model_idx)
        #                 used_storage += model_size
        #         edge_server.cached_models = cached_models
        #         edge_server.reset_request_counters(self.base_env.num_models)
        # else:
        for edge_id, edge_server in enumerate(self.base_env.edge_servers):
            edge_cache_decision = self.slow_actions[edge_id]
            cached_models = []
            used_storage = 0
            for model_idx, should_cache in enumerate(edge_cache_decision):
                model_size = self.base_env.task_types[model_idx].model_size
                if should_cache and (used_storage + model_size <= edge_server.cache_storage):
                    cached_models.append(model_idx)
                    used_storage += model_size
            edge_server.cached_models = cached_models
            edge_server.reset_request_counters(self.base_env.num_models)
        # --- Update latent popularity for each edge server ---
        self.base_env._update_latent_popularity()
        # --- Update Zipf rankings based on latent popularity ---
        self.base_env.update_zipf_rankings_from_popularity()
        # slow_obs = self._get_slow_obs()
        # return slow_obs

    def step_fast(self, offload_exit_actions):
        """
        The fast agent takes an action (offloading and exit decisions).
        This completes one fast time slot (processing one batch of tasks).
        """
        self.fast_actions = offload_exit_actions  
        # The base_env.step() handles everything for the fast timescale.
        # We pass the fast actions to it. Caching is now handled by step_slow.
        # We need to prevent the base_env.step from performing its own caching logic.
        original_freeze = getattr(self.base_env, "freeze_cache_updates", False)
        self.base_env.freeze_cache_updates = True
        obs, reward, terminated, truncated, info = self.base_env.step(self.fast_actions)
        self.cumulative_fast_reward += reward
        self.base_env.freeze_cache_updates = original_freeze  # Restore original state
        # The next observation for the fast agent would be for a new set of tasks,
        # but under the same slow-timescale caching decision.
        next_fast_obs = self._get_fast_obs()
        return next_fast_obs, reward, terminated, truncated, info
    
    def save_state(self):
        # Save the state of the dual timescale environment
        # Save the state of the dual timescale environment
        state = {
            'base_env': self.base_env.save_state() if hasattr(self.base_env, 'save_state') else copy.deepcopy(self.base_env.__dict__),
            'slow_actions': copy.deepcopy(self.slow_actions),
            'fast_actions': copy.deepcopy(self.fast_actions),
            'slow_step_count': self.slow_step_count,
            'cumulative_fast_reward': self.cumulative_fast_reward,
        }
        return state

    def load_state(self, state):
        import copy
        # Restore the state of the dual timescale environment
        if 'base_env' in state and hasattr(self.base_env, 'load_state'):
            self.base_env.load_state(state['base_env'])
        if 'slow_actions' in state:
            self.slow_actions = copy.deepcopy(state['slow_actions'])
        if 'fast_actions' in state:
            self.fast_actions = copy.deepcopy(state['fast_actions'])
        if 'slow_step_count' in state:
            self.slow_step_count = state['slow_step_count']
        if 'cumulative_fast_reward' in state:
            self.cumulative_fast_reward = state['cumulative_fast_reward']


class SlowEnvWrapper(gym.Env):
    """Wrapper for the slow agent (caching control)."""

    def __init__(self, dual_env: MECDualTimeScaleEnv):
        super().__init__()
        self.dual_env = dual_env
        self.base_env = self.dual_env.base_env

        # Action: Flattened caching decisions for each model on each edge server.
        self.action_space = gym.spaces.MultiBinary(
            self.base_env.num_edges * self.base_env.num_models
        )

        # Observation for the slow agent
        self.observation_space = gym.spaces.Dict({
            "model_caching_gain": gym.spaces.Box(
                low=0, high=np.inf, shape=(self.base_env.num_models,), dtype=np.float32
            ),
            "model_popularity": gym.spaces.Box(
                low=0, high=1.0, shape=(self.base_env.num_models,), dtype=np.float32
            ),
            "edge_popularity": gym.spaces.Box(
                low=0, high=1.0, shape=(self.base_env.num_edges,self.base_env.num_models), dtype=np.float32
            ),
            "model_sizes": gym.spaces.Box(
                low=0, high=1.0, shape=(self.base_env.num_models,), dtype=np.float32
            ),
            "slow_timestep": gym.spaces.Box(
                low=0, high=1.0, shape=(1,), dtype=np.float32
            ),
            # "nearby_edges": gym.spaces.Box(
            #     low=0, high=1.0, shape=(self.base_env.num_edges, self.base_env.num_edges), dtype=np.float32
            # ),
            # "edge_computing_capacity": gym.spaces.Box(
            #     low=0, high=1.0, shape=(self.base_env.num_edges,), dtype=np.float32
            # ),
            # "cloud_computing_capacity": gym.spaces.Box(
            #     low=0, high=1.0, shape=(1,), dtype=np.float32
            # ),
            # "connected_edge_servers": gym.spaces.MultiBinary(
            #     (self.base_env.num_edges, self.base_env.num_devices)
            # ),
            # "num_blocks": gym.spaces.Box(
            #     low=0, high=1.0, shape=(self.base_env.num_models,), dtype=np.float32
            # ),
            # "block_compute_requirement": gym.spaces.Box(
            #     low=0, high=1.0, shape=(self.base_env.num_models, 5), dtype=np.float32
            # ),
            # "block_accuracy": gym.spaces.Box(
            #     low=0, high=1.0, shape=(self.base_env.num_models, 5), dtype=np.float32
            # ),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.base_env.rng = np.random.default_rng(seed)
        slow_obs = self.dual_env.reset()
        return slow_obs, {}

    def step(self, action):
        # The slow agent's action is applied.
        # This returns the first observation for the fast agent.
        
        # Reshape the flattened action vector back to a 2D matrix
        reshaped_action = action.reshape(
            (self.base_env.num_edges, self.base_env.num_models)
        )
        self.dual_env.step_slow(reshaped_action)
        
        # Increment slow timestep counter
        self.dual_env.slow_step_count += 1
        
        # Return the next slow observation following gym convention
        # (obs, reward, terminated, truncated, info)
        next_obs = self.dual_env._get_slow_obs()
        reward = 0.0  # Reward is computed later via get_slow_reward_observations()
        terminated = False  # Slow episodes don't terminate early
        truncated = False
        info = {}
        
        return next_obs, reward, terminated, truncated, info

    def get_slow_reward_observations(self):
        # Calculate cache hit rate for the slow episode
        total_requests = sum(edge.total_requests for edge in self.base_env.edge_servers)
        cache_hits = sum(edge.cache_hits for edge in self.base_env.edge_servers)
        slow_reward = cache_hits / total_requests if total_requests > 0 else 0.0
        obs = self.dual_env._get_slow_obs()
        avg_fast_reward = self.dual_env.cumulative_fast_reward / self.base_env.large_timescale_size
        return slow_reward, obs, avg_fast_reward


class FastEnvWrapper(gym.Env):
    """Wrapper for the fast agent (offloading and early-exit control)."""

    def __init__(self, dual_env: MECDualTimeScaleEnv):
        super().__init__()
        self.dual_env = dual_env
        self.base_env = self.dual_env.base_env

        # Action space is inherited from the base environment
        self.action_space = self.base_env.action_space
        # Observation space is also inherited
        self.observation_space = self.base_env.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Resetting the fast env gets the current observation from the dual env state.
        # It does not reset the entire base environment.
        obs = self.dual_env._get_fast_obs()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.dual_env.step_fast(action)
        return obs, reward, terminated, truncated, info
    
    def save_state(self):
        # Save the state of the dual_env and base_env
        state = {
            'dual_env': self.dual_env.save_state() if hasattr(self.dual_env, 'save_state') else copy.deepcopy(self.dual_env.__dict__),
            'base_env': self.base_env.save_state() if hasattr(self.base_env, 'save_state') else copy.deepcopy(self.base_env.__dict__),
        }
        return state

    def load_state(self, state):
        # Restore the state of the dual_env and base_env
        if 'dual_env' in state and hasattr(self.dual_env, 'load_state'):
            self.dual_env.load_state(state['dual_env'])
        if 'base_env' in state and hasattr(self.base_env, 'load_state'):
            self.base_env.load_state(state['base_env'])