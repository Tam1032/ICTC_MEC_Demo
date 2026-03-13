import numpy as np


class OffloadOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        obs, info = self.env.reset()
        unwrapped_env = self.env.unwrapped
        action = []
        for task in unwrapped_env.tasks:
            # Make offload decision always local (max exit index)
            offload_decision = 0
            # Always choose local processing
            action.append(offload_decision)
        # Random caching decision (0 or 1 for each model)
        cache_decision = unwrapped_env.cache_random_models()
        # Combine all actions as required by your environment
        # Example: [offload_decisions, exit_selections, cache_decision]
        # If your env expects a flat array:
        action = np.concatenate([action, cache_decision])
        return action