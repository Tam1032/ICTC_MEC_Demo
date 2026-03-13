import numpy as np


class LocalOffloadOptimizer:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        offload_actions = np.zeros(len(self.env.mobile_devices), dtype=int)
        for task in self.env.pending_tasks:
            device_id = task.device_id
            device = self.env.mobile_devices[device_id]
            # Always offload to the local edge server
            offload_decision = device.edge_id
            offload_actions[device_id] = offload_decision
        return offload_actions