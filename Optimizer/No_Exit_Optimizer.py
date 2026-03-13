import numpy as np


class NoExitOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        exit_actions = [num_block - 1 if num_block > 0 else num_block for num_block in obs["num_blocks"]]
        return exit_actions