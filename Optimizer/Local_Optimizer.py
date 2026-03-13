import numpy as np


class LocalOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        obs, info = self.env.reset()
        #unwrapped_env = self.env.unwrapped
        action = []
        for task in self.env.tasks:
            # Make offload decision always local (max exit index)
            offload_decision = task.task_type.num_exits
            # Always choose local processing
            action.append(offload_decision)
        # Random caching decision (0 or 1 for each model)
        cache_decision = self.env.cache_random_models()
        # Combine all actions as required by your environment
        # Example: [offload_decisions, exit_selections, cache_decision]
        # If your env expects a flat array:
        action = np.concatenate([action, cache_decision])
        return action