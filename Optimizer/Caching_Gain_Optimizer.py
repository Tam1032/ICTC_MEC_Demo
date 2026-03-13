import numpy as np

class CachingGainOptimizer:
    """
    A non-learning optimizer for the slow timescale (caching) that prioritizes
    models based on their "caching gain".
    
    Caching Gain is defined as the processing time saved by executing on the edge
    instead of the cloud.
    Gain = (Cloud Processing Time) - (Edge Processing Time)
    """
    def __init__(self, dual_env):
        """
        Initializes the optimizer.
        
        Args:
            dual_env (MECDualTimeScaleEnv): The dual-timescale environment instance.
        """
        self.base_env = dual_env.base_env
        self.num_edges = self.base_env.num_edges
        self.num_models = self.base_env.num_models
        self.task_types = self.base_env.task_types
        self.edge_servers = self.base_env.edge_servers
        self.cloud_server = self.base_env.cloud_server

    def _calculate_caching_gain(self, edge_server):
        """
        Calculates the caching gain for each model type for a specific edge server.
        """
        gains_per_request = []
        for model in self.task_types:
            # --- Calculate Latency for processing on the Cloud ---
            # 1. Transmission to Cloud (approximated)
            # We assume a representative task size for this calculation.
            task_input_size = self.base_env.task_size_input
            cloud_transmit_delay = self.cloud_server.calculate_transmission_delay(task_input_size)

            # 2. Compute on Cloud (for the full model)
            required_cycles = model.get_total_compute_requirement()
            cloud_compute_delay = self.cloud_server.calculate_computing_delay(required_cycles)
            
            total_cloud_latency = cloud_transmit_delay + cloud_compute_delay

            # --- Calculate Latency for processing on the Edge (if cached) ---
            # On a cache hit, the only delay is local edge computation.
            edge_compute_delay = edge_server.calculate_computing_delay(required_cycles)

            # --- Gain is the latency saved ---
            gain = total_cloud_latency - edge_compute_delay
            gains_per_request.append(gain)
        
        # Total expected gain = (gain per request) * (number of requests)
        total_expected_gain = np.array(gains_per_request) * model_popularity
        return total_expected_gain

    def predict_slow(self, obs):
        """
        Makes a caching decision for all edge servers based on caching gain.
        
        Args:
            obs (dict): The observation for the slow agent. Expected keys:
                        'model_caching_gain', 'model_popularity', 'model_sizes',
                        'edge_storage_capacity'.
            
        Returns:
            np.array: A 2D binary array of shape (num_edges, num_models) indicating
                      which models to cache on each edge server.
        """
        # Unpack observations from the environment
        avg_gain_per_request = obs['model_caching_gain']
        #model_popularity = obs['model_popularity']
        model_sizes = obs['model_sizes']

        # Calculate the total expected gain for each model
        # This weights the average gain by how popular the model is.
        #total_expected_gain = avg_gain_per_request * model_popularity

        # Get model indices sorted by total expected gain (descending)
        sorted_model_indices = np.argsort(avg_gain_per_request)[::-1]

        # Initialize the caching decision matrix with zeros
        caching_decisions = np.zeros((self.num_edges, self.num_models), dtype=int)

        # For each edge, greedily fill its cache with the highest-gain models
        for edge_id in range(self.num_edges):
            remaining_storage = 1.0  # Since sizes are normalized
            
            for model_idx in sorted_model_indices:
                model_size = model_sizes[model_idx]
                if model_size <= remaining_storage:
                    caching_decisions[edge_id, model_idx] = 1
                    remaining_storage -= model_size
        
        return caching_decisions