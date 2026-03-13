import numpy as np


class RandomOptimizer:
    def __init__(self, env, seed=42):
        self.env = env
        np.random.seed(seed)

    def predict(self, obs):
        obs, info = self.env.reset()
        #unwrapped_env = self.env.unwrapped
        action = []
        num_edges = self.env.num_edges
        edge_selection = np.random.randint(0, num_edges, size=self.env.num_devices)
        action.extend(edge_selection)
        for task in self.env.tasks:
            # Random offload selection (include split computing)
            max_exits = task.task_type.num_blocks
            offload_decision = np.random.randint(0, max_exits+1)
            action.append(offload_decision)
        return action