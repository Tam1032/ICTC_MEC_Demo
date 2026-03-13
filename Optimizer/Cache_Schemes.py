import numpy as np

class PopularityOptimizer:
    """
    An optimizer that makes caching decisions based on model popularity.
    This class is designed to replace the "slow" RL agent in the dual-timescale setup.
    It caches the most popular models that fit within each edge's storage capacity.
    """
    def __init__(self, env):
        """
        Initializes the optimizer.

        Args:
            env: The MECDualTimeScaleEnv environment instance to get parameters from.
        """
        self.num_edges = env.base_env.num_edges
        self.num_models = env.base_env.num_models

    def predict_slow(self, obs):
        """
        Determines the caching action based on model popularity from the observation.
        This method mimics the slow_agent.predict() interface.

        Args:
            obs (dict): The observation from the SlowEnvWrapper. 
                        Expected keys: 'model_popularity', 'model_sizes', 'edge_storage_capacity'.

        Returns:
            np.ndarray: A 2D numpy array of shape (num_edges, num_models) representing
                        the caching decision (1 for cache, 0 for not).
        """
        model_popularity = obs['edge_popularity']
        model_sizes = obs['model_sizes']

        # Get model indices sorted by popularity (descending)
        #sorted_model_indices = np.argsort(model_popularity)[::-1]

        # Initialize the caching decision matrix with zeros
        cache_decision = np.zeros((self.num_edges, self.num_models), dtype=int)

        # For each edge, fill its cache with the most popular models that fit
        for edge_idx in range(self.num_edges):
            # Get model indices sorted by popularity for this specific edge (descending)
            remaining_storage = 1.0  # Since sizes are normalized
            edge_popularity = model_popularity[edge_idx]
            sorted_model_indices = np.argsort(edge_popularity)[::-1]
            for model_idx in sorted_model_indices:
                model_size = model_sizes[model_idx]
                if model_size <= remaining_storage:
                    cache_decision[edge_idx, model_idx] = 1
                    remaining_storage -= model_size
        
        return cache_decision
    
class RandomCacheOptimizer:
    """
    An optimizer that makes random caching decisions.
    This class is designed to replace the "slow" RL agent in the dual-timescale setup.
    It randomly selects models to cache for each edge while respecting storage constraints.
    """
    def __init__(self, env, seed=42):
        """
        Initializes the optimizer.

        Args:
            env: The MECDualTimeScaleEnv environment instance to get parameters from.
            seed: Random seed for reproducibility.
        """
        self.num_edges = env.base_env.num_edges
        self.num_models = env.base_env.num_models
        self.rng = np.random.default_rng(seed)

    def predict_slow(self, obs):
        """
        Randomly selects models to cache for each edge server while
        respecting storage constraints, following a shuffle-and-fill logic.

        Args:
            obs (dict): The observation from the SlowEnvWrapper (not used, but for interface consistency).

        Returns:
            np.ndarray: A 2D numpy array of shape (num_edges, num_models) representing
                        the caching decision (1 for cache, 0 for not).
        """
        model_sizes = obs['model_sizes']
        # Initialize an empty decision matrix (all zeros)
        caching_decisions = np.zeros((self.num_edges, self.num_models), dtype=int)
        for edge_idx in range(self.num_edges):
            remaining_storage = 1.0  # Since sizes are normalized
            # Create and shuffle indices to ensure random priority for each server
            model_indices = np.arange(self.num_models)
            self.rng.shuffle(model_indices)
            for model_idx in model_indices:
                model_size = model_sizes[model_idx]
                # If the model fits in the remaining space, cache it
                if model_size <= remaining_storage:
                    caching_decisions[edge_idx, model_idx] = 1
                    remaining_storage -= model_size

        return caching_decisions

