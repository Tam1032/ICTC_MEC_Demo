import numpy as np


class LocalOffloadOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        offload_actions = np.zeros(len(self.env.base_env.mobile_devices), dtype=int)
        for task in self.env.base_env.pending_tasks:
            device_id = task.device_id
            device = self.env.base_env.mobile_devices[device_id]
            # Always offload to the local edge server
            offload_decision = device.edge_id
            offload_actions[device_id] = offload_decision
        return offload_actions
    
class RandomOffloadOptimizer:
    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs):
        """
        Generates random offloading decisions.
        Returns: array of offload decisions for each device.
        """
        num_devices = self.env.base_env.num_devices
        num_edges = self.env.base_env.num_edges
        # Random edge selection for each device
        offload_decisions = np.zeros(num_devices, dtype=int)
        # Generate decisions only for active tasks
        active_task_indices = np.where(np.any(obs["edge_transmission_delays"] > 0, axis=1))[0]
        for i in active_task_indices:
            # Random edge selection (0 to num_edges, where num_edges is the cloud)
            offload_decisions[i] = self.rng.integers(0, num_edges + 1)
        return offload_decisions

class CloudOnlyOptimizer:
    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs):
        """
        Always offloads to cloud.
        Returns: array of offload decisions for each device (all to cloud).
        """
        num_devices = self.env.base_env.num_devices
        num_edges = self.env.base_env.num_edges
        # Always offload to cloud
        offload_decisions = np.full(num_devices, num_edges, dtype=int)
        return offload_decisions