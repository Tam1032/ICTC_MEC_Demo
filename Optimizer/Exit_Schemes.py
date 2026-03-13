import numpy as np

class NoExitOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        exit_actions = [num_block - 1 if num_block > 0 else num_block for num_block in obs["num_blocks"]]
        return exit_actions
    
class RandomExitOptimizer:
    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs):
        exit_actions = np.zeros(len(obs["num_blocks"]), dtype=int)
        for i, num_block in enumerate(obs["num_blocks"]):
            if num_block > 0:
                offload_decision = self.rng.integers(0, num_block)
            else:
                offload_decision = num_block
            exit_actions[i] = offload_decision
        return exit_actions
    
class ExitAtFirstOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        exit_actions = [0 for _ in obs["num_blocks"]]
        return exit_actions
    
class ExitAtSecondLastOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        exit_actions = [num_block - 2 if num_block > 0 else num_block for num_block in obs["num_blocks"]]
        return exit_actions