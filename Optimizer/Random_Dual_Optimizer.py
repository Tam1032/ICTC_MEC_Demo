import numpy as np


class RandomDualOptimizer:
    """
    Random optimizer for dual-timescale MEC environment.
    Provides random decisions for both caching (slow) and offloading (fast).
    """

    def __init__(self, dual_env, seed=42):
        self.dual_env = dual_env
        self.base_env = dual_env.base_env
        self.rng = np.random.default_rng(seed)

    def predict_slow(self, obs):
        """
        Randomly selects models to cache for each edge server while 
        respecting storage constraints, following a shuffle-and-fill logic.
        """
        num_edges = self.base_env.num_edges
        num_models = self.base_env.num_models
        # Initialize an empty decision matrix (all zeros)
        caching_decisions = np.zeros((num_edges, num_models), dtype=int)
        for edge_id, edge_server in enumerate(self.base_env.edge_servers):
            available_storage = edge_server.cache_storage
            # Create and shuffle indices to ensure random priority for each server
            model_indices = np.arange(num_models)
            self.rng.shuffle(model_indices)
            for model_idx in model_indices:
                model_size = self.base_env.task_types[model_idx].model_size
                # If the model fits in the remaining space, cache it
                if model_size <= available_storage:
                    caching_decisions[edge_id, model_idx] = 1
                    available_storage -= model_size
        return caching_decisions

    def predict_fast(self, obs):
        """
        Generates random offloading and early-exit decisions.
        Returns: action array matching the fast environment's action space
        """
        # num_active_tasks = np.count_nonzero(obs["edge_transmission_delays"])
        num_devices = self.base_env.num_devices
        num_edges = self.base_env.num_edges
        # Random edge selection for each device
        offload_decisions = np.zeros(num_devices, dtype=int)
        exit_decisions = np.zeros(num_devices, dtype=int)
        # Generate decisions only for active tasks
        # Check if any edge has a valid transmission delay for this device
        active_task_indices = np.where(np.any(obs["edge_transmission_delays"] > 0, axis=1))[0]
        # Random early-exit decisions for each task
        for i in active_task_indices:
            # Random edge selection (0 to num_edges, where num_edges is the cloud)
            offload_decisions[i] = self.rng.integers(0, num_edges + 1)
            # Random early-exit decision
            max_exits = obs["num_blocks"][i]
            if max_exits > 0:
                exit_decisions[i] = self.rng.integers(0, max_exits)
        # The environment expects a flat array: offload decisions first, then exit decisions
        action = np.concatenate([offload_decisions, exit_decisions]).astype(int)
        return action
