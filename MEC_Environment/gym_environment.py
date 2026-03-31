import itertools
import numpy as np
import copy
import json
import os
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from .tasks import Block, DNN_Model, Task
from .devices import CloudServer, EdgeServer, MobileDevice


def load_block_config():
    """Load configuration from block_config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'block_config.json')
    with open(config_path, 'r') as f:
        profiles = json.load(f)
    return profiles

# Load config at module level
_CONFIG = load_block_config()


class Environment(gym.Env):
    def __init__(
        self,
        num_edges,
        num_devices,
        num_models,
        local_computing_range,
        cloud_computing,
        edge_computing,
        edge_storage,
        bandwidth,
        i2i_tranmit_rate,
        cloud_download_rate,
        cloud_propagation_delay,
        models_size_range,
        compute_size_range,
        large_timescale_size,
        task_size_input,
        task_arrival_rate=0.7,
        time_step_duration=1.0,
        edge_cores=8,
        zipf_a=1.2,
        acc_weight=0.5,
        latency_weight=1.0,
        offload_oblivious=False,
        exit_oblivious=False,
        cache_oblivious=False,
        profile="all",
        seed=42,
    ):
        # Set the random seed for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Initialize the multi-user, multi-edge environment
        self.num_edges = num_edges
        self.num_devices = num_devices
        self.i2i_transmit_rate = i2i_tranmit_rate  # in Mbps
        self.cloud_server = CloudServer(cloud_computing, 8, cloud_download_rate, propagation_delay=cloud_propagation_delay)
        # Initialize multiple edge servers
        self.edge_servers = []
        for edge_id in range(num_edges):
            self.edge_servers.append(EdgeServer(edge_computing, bandwidth, edge_storage, edge_cores, edge_id))
            self.edge_servers[edge_id].reset_request_counters(num_models)
        # Define edge server connections and hop distances
        self.edge_connections = self._initialize_edge_connections(self.num_edges)
        #self.edge_connections = self.initialize_grid_connections(self.num_edges)
        self.edge_phase_offsets = np.array([
            2 * np.pi * e / self.num_edges for e in range(self.num_edges)
        ])
        # Initialize multiple mobile devices
        self.mobile_devices = []
        self._initialize_mobile_devices(local_computing_range)
        self.num_models = num_models
        # Get exit
        self.profiles = _CONFIG.get('profiles', {})
        if profile != "all":
            self.profiles = {profile_name: profile_data for profile_name, profile_data in self.profiles.items() if profile_name == profile}
        #self.task_type = [DNN_Model]
        self.tasks = []
        self.task_types = self.generate_task_type(models_size_range, compute_size_range)
        self.task_size_input = task_size_input
        # Denote Zipf parameter for generating tasks based on Zipf distribution
        self.zipf_a = zipf_a
        self.acc_weight = acc_weight
        self.latency_weight = latency_weight
        # Define global min-max values for normalization
        self._update_scaling_bounds(task_size_input, compute_size_range, self.profiles)
        self.zipf_rankings = self.generate_zipf_rankings(self.num_models, self.num_edges)
        self.time = 0
        self.num_steps = 0
        self.time_step_duration = time_step_duration  # Each step processes this much time
        #self.task_arrival_rate = task_arrival_rate / self.num_devices # Average number of tasks arriving per timestep
        self.task_arrival_rate = task_arrival_rate  # Average number of tasks arriving per timestep
        self.pending_tasks = []
        self.action_space = gym.spaces.MultiDiscrete(
            [self.num_edges+1] * self.num_devices + [5] * self.num_devices
        )
        # Define special case when cache or early exit is not considered
        self.offload_oblivious = offload_oblivious
        self.exit_oblivious = exit_oblivious
        self.cache_oblivious = cache_oblivious
        self.freeze_cache_updates = False
        # Redefine action space if cache or exit is not considered
        if self.offload_oblivious:
            self.action_space = gym.spaces.MultiDiscrete(
                [5] * self.num_devices  # No offloading decision
            )
        if self.exit_oblivious:
            self.action_space = gym.spaces.MultiDiscrete(
                [self.num_edges+1] * self.num_devices # No exit selection
            )
        # =============== Observation Space ===============
        # We'll create a Dict space with:
        # - per-device: input_size, bandwidth, channel_gain
        # - global: model_size, computation_demand, local/edge compute capacity
        # Note: time is normalized relative to current timestep
        # Note: full_compute_requirement and block_compute_requirement are normalized by allocated_edge_computing
        self.observation_space = gym.spaces.Dict(
            {
                # Per-device dynamic (includes cloud as the last column)
                "edge_transmission_delays": gym.spaces.Box(
                    low=0, high=10.0, shape=(num_devices, num_edges + 1), dtype=np.float32
                ),  # seconds
                # Global constants (broadcasted or repeated)
                "feasible_edge_caches": gym.spaces.MultiBinary((num_devices, num_edges)),
                "num_blocks": gym.spaces.Box(
                    low=0, high=8, shape=(num_devices,), dtype=np.int32
                ),
                # Allocated computing resources per device (normalized values)
                "allocated_edge_computing": gym.spaces.Box(
                    low=0, high=1e6, shape=(num_edges,), dtype=np.float32
                ),
                "allocated_cloud_computing": gym.spaces.Box(
                    low=0, high=1e6, shape=(1,), dtype=np.float32
                ),
                # Compute requirements normalized by allocated_edge_computing
                "block_compute_requirement": gym.spaces.Box(
                    low=0, high=1.0, shape=(num_devices, 5), dtype=np.float32
                ),
                "block_accuracy": gym.spaces.Box(
                    low=0, high=1.0, shape=(num_devices, 5), dtype=np.float32
                ),
                # Core completion times for each edge server, normalized relative to current time (shape: num_edges x edge_cores)
                "edge_core_completion_times": gym.spaces.Box(
                    low=0, high=1e6, shape=(self.num_edges, edge_cores), dtype=np.float32
                ),
                # Add task arrival times for each device (shape: num_devices,)
                "task_arrival_times": gym.spaces.Box(
                    low=0, high=1e6, shape=(num_devices,), dtype=np.float32
                ),
            }
        )
        # Define code for tracking model popularity
        self.popularity_history = np.zeros(self.num_models, dtype=int)
        self.popularity_period = 200
        self.popularity_amplitude = 0.8
        self.base_popularity = 1.0
        # --- Latent true popularity (exogenous, not influenced by caching) ---
        # self.current_log_popularity = np.zeros(self.num_models)  # log of unnormalized popularity
        self.latent_log_popularity_per_edge = [
            np.zeros(self.num_models) for _ in range(self.num_edges)
        ]  # log-space, starts uniform
        self.popularity_drift_scale = 0.05  # tune: 0.005–0.02 for gradual change
        # Keep a history window for popularity estimation and logging purposes
        self.popularity_window = []  # To keep track of last N timesteps
        self.caching_gain_window = [] # To keep track of the caching gains history
        self.large_timescale_size = large_timescale_size
        self.popularity_window_size = large_timescale_size  # Or any window size you want
        self.caching_gain_window_size = large_timescale_size
        # Initialize cache hit counters
        for edge in self.edge_servers:
            edge.cache_hits = 0
            edge.total_requests = 0

    def _initialize_edge_connections(self, num_edges):
        """
        Initializes the connections and hop distances between edge servers.
        This implementation creates a ring topology.
        """
        connections = {}
        for i in range(num_edges):
            for j in range(num_edges):
                if i == j:
                    continue
                # Calculate shortest distance in a ring topology
                hop_distance = min(abs(i - j), num_edges - abs(i - j))
                connections[(i, j)] = hop_distance
        return connections
    
    def _initialize_mobile_devices(self, local_computing_range):
        for i in range(self.num_devices):
            local_computing = int(self.rng.uniform(*local_computing_range))
            distance_km = self.rng.uniform(0.05, 0.2)
            channel_gain = self._generate_channel_gain(distance_km)
            #edge_id = self.rng.choice(self.num_edges)
            if self.num_edges == 3:
                edge_id = self.rng.choice(self.num_edges, p=[0.4, 0.3, 0.3])
            else:
                edge_id = self.rng.integers(0, self.num_edges)
            #edge_id = 0
            self.edge_servers[edge_id].connect_device()
            self.mobile_devices.append(
                MobileDevice(
                    i, distance_km, channel_gain, local_computing, edge_id
                )
            )
        for i in range(self.num_devices):
            edge_id = self.mobile_devices[i].edge_id
            allocated_bandwidth = self.edge_servers[edge_id].calculate_bandwidth_allocated()
            self.mobile_devices[i].assign_bandwidth(allocated_bandwidth)  # Convert Hz to MHz

    def _update_scaling_bounds(self, input_size_range, compute_size_range, profiles):
        # Compute min and max from block_config.json
        all_accuracies = []
        all_compute_percentages = []
        for profile_name, profile_data in profiles.items():
            base_accuracies = profile_data.get('base_accuracies', {})
            for num_blocks, acc_list in base_accuracies.items():
                all_accuracies.extend(acc_list)
            base_compute_percentages = profile_data.get('base_compute_percentages', {})
            for num_blocks, pct_list in base_compute_percentages.items():
                all_compute_percentages.extend(pct_list)
        
        if all_accuracies:
            self.min_accuracy = min(all_accuracies)
            self.max_accuracy = max(all_accuracies)
        else:
            self.min_accuracy = 0.0
            self.max_accuracy = 1.0
        
        if all_compute_percentages:
            min_compute_percentage = min(all_compute_percentages)
        else:
            min_compute_percentage = 0.0
        
        # Estimate min and max latency based on input size and compute size ranges
        min_input_size = input_size_range[0]
        max_input_size = input_size_range[1]
        min_compute_size = compute_size_range[0]
        max_compute_size = compute_size_range[1]

        # Estimate min latency (best case: local compute with min input and compute sizes)
        min_transmit_latency = min_input_size / 4
        min_compute_latency = min_compute_percentage * min_compute_size / self.edge_servers[0].core_computing_power
        self.min_latency = min_transmit_latency + min_compute_latency
        # Estimate max latency (worst case: cloud compute with max input and compute sizes)
        max_transmit_latency = max_input_size / 4 + max_input_size / self.cloud_server.download_rate + self.cloud_server.propagation_delay
        max_compute_latency = max_compute_size * 1e9 / self.cloud_server.computing_power
        self.max_latency = max_transmit_latency + max_compute_latency
    
    def generate_blocks(self, num_blocks=None, selected_profile=None, total_compute=None):
        if num_blocks is None:
            num_blocks = self.rng.integers(3, 6)  # 3, 4, or 5
        
        # Randomly select a profile from the environment's filtered profiles
        if selected_profile is None:
            selected_profile = self.rng.choice(list(self.profiles.keys()))
        profile_data = self.profiles[selected_profile]
        
        # Load base values from randomly selected profile
        base_accuracies = {int(k): v for k, v in profile_data['base_accuracies'].items()}
        base_compute_percentages = {int(k): v for k, v in profile_data['base_compute_percentages'].items()}

        # --- Load base values for this num_blocks ---
        base_accs = np.array(base_accuracies[num_blocks])
        base_pcts = np.array(base_compute_percentages[num_blocks])
        accuracies = base_accs
        computes = base_pcts * total_compute
        # --- Add variance to accuracies ---
        # acc_noise_level = 0.015  # ±1.5% absolute (adjust as needed)
        # acc_noise = self.rng.uniform(-acc_noise_level, acc_noise_level, size=num_blocks)
        # accuracies = np.clip(base_accs + acc_noise, 0.0, 1.0)
        # # --- Add variance to compute percentages (with constraints) ---
        # pct_noise_level = 0.03  # ±3% relative noise
        # pct_noise = base_pcts * self.rng.uniform(-pct_noise_level, pct_noise_level, size=num_blocks)
        # # Apply noise, but enforce:
        # noisy_pcts = base_pcts + pct_noise
        # # 1. Ensure non-decreasing order
        # for i in range(1, num_blocks):
        #     noisy_pcts[i] = max(noisy_pcts[i], noisy_pcts[i-1])
        # # 2. Force last block to be exactly 1.0 (100%)
        # noisy_pcts[-1] = 1.0
        # # 3. Clip to [0, 1] for safety
        # noisy_pcts = np.clip(noisy_pcts, 0.0, 1.0)
        # # --- Scale to absolute compute ---
        # # You can also sample total_compute from a distribution if desired
        # #total_compute = 2e8 * (1.5 ** (num_blocks - 1))  # or self.rng.uniform(1.8e8, 2.5e8) * (1.5**(num_blocks-1))
        # computes = noisy_pcts * total_compute
        return [Block(c, a) for c, a in zip(computes, accuracies)]

    def generate_task_type(self, models_size_range, compute_size_range, profile_type=None):
        task_types = []
        min_compute, max_compute = compute_size_range
        # Generate block settings
        block_options = [3, 4, 5]
        profile_options = list(self.profiles.keys())
        combinations = list(itertools.product(block_options, profile_options))
        num_types = len(combinations)
        for model_idx in range(self.num_models):
            i = model_idx % num_types
            num_blocks, selected_profile = combinations[i]
            # Scale compute size based on num_blocks within the range
            scale = (num_blocks - 3) / 2.0  # 0 for 3 blocks, 0.5 for 4, 1 for 5
            compute_size = (min_compute + scale * (max_compute - min_compute)) * 1e9  # in FLOPS
            all_blocks = self.generate_blocks(num_blocks=num_blocks, selected_profile=selected_profile, total_compute=compute_size)
            task_type = DNN_Model(
                model_size=round(self.rng.uniform(*models_size_range), 1),  # in GB
                blocks=all_blocks,
            )
            task_types.append(task_type)
        return task_types
    
    def generate_zipf_rankings(self, num_items, num_edges, a=1.2):
        """
        Generate a unique Zipf ranking (permutation) for each edge server.
        Returns: List of lists, each is a permutation of model indices.
        """
        rankings = []
        for _ in range(num_edges):
            perm = self.rng.permutation(num_items)
            rankings.append(perm)
        return rankings

    def update_zipf_rankings_from_popularity(self):
        """
        Update Zipf rankings per edge based on that edge's latent popularity.
        Each edge can have a different popularity trend.
        """
        new_rankings = []
        for edge_id in range(self.num_edges):
            true_pop = np.exp(self.latent_log_popularity_per_edge[edge_id])
            ranking = np.argsort(-true_pop)  # most popular first
            new_rankings.append(ranking.copy())
        self.zipf_rankings = new_rankings

    def _update_latent_popularity(self):
        """
        Update popularity for each edge independently using sinusoidal dynamics.
        Each edge has its own phase (e.g., simulating time zones or user behavior).
        Each model has its own peak within the edge's cycle.
        """
        t = self.num_steps
        global_phase = 2 * np.pi * t / self.popularity_period  # shared time base
        self.latent_log_popularity_per_edge = []
        
        for edge_id in range(self.num_edges):
            # Edge-specific phase shift (e.g., time zone difference)
            edge_phase = global_phase + self.edge_phase_offsets[edge_id]
            
            # Compute popularity for each model on this edge
            model_popularity = np.zeros(self.num_models)
            for model_idx in range(self.num_models):
                # Model-specific offset within the edge's cycle
                model_phase_offset = 2 * np.pi * model_idx / self.num_models
                total_phase = edge_phase + model_phase_offset
                
                model_popularity[model_idx] = (
                    self.base_popularity + 
                    self.popularity_amplitude * np.sin(total_phase)
                )
            
            # Ensure positivity
            model_popularity = np.clip(model_popularity, 0.1, None)
            log_pop = np.log(model_popularity)
            self.latent_log_popularity_per_edge.append(log_pop)

    def _generate_task_for_device(self, device_id):
        """Generates a single task for a specific device."""
        dev = self.mobile_devices[device_id]
        edge_id = dev.edge_id
        ranking = self.zipf_rankings[edge_id]
        a = self.zipf_a
        ranks = np.arange(1, self.num_models + 1)
        probs = 1 / np.power(ranks, a)
        probs /= probs.sum()
        sampled_rank = self.rng.choice(ranks, p=probs) - 1
        model_idx = ranking[sampled_rank]
        # Sample directly from latent popularity (normalized)
        # true_pop = np.exp(self.latent_log_popularity)
        # true_pop /= true_pop.sum()  # shape: (num_models,)
        # model_idx = self.rng.choice(self.num_models, p=true_pop)
        task_type = self.task_types[model_idx] 
        task = Task(
            device_id=dev.device_id,
            input_size=self.rng.uniform(*self.task_size_input),
            task_type=task_type,
            arrival_time=0,  # Placeholder, will be set in __generate_timestep_tasks 
        )
        return task
    
    def __generate_timestep_tasks(self):
        """
        Process all events that occur within the current timestep window.
        Collects tasks based on Poisson arrival process.
        """
        self.pending_tasks = []
        # Generate number of tasks using Poisson distribution, at least 1
        num_tasks = self.rng.poisson(self.task_arrival_rate*self.num_devices)
        # if num_tasks == 0:
        #     print("No tasks generated this timestep.")
        # Randomly select devices without replacement
        device_ids = self.rng.choice(self.num_devices, size=min(num_tasks, self.num_devices), replace=False)
        for device_id in device_ids:
            # Generate a random arrival time within this timestep
            arrival_time = self.time + self.rng.uniform(0, self.time_step_duration)
            new_task = self._generate_task_for_device(device_id)
            new_task.arrival_time = arrival_time
            self.pending_tasks.append(new_task)
            # Track this task on the device for visualization
            if hasattr(self.mobile_devices[device_id], 'assigned_tasks'):
                self.mobile_devices[device_id].assigned_tasks.append(new_task)
                
    def _get_obs(self, tasks):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        
        # Per-device data (aligned by device_id)
        # Added +1 to num_edges to include CloudServer as the last column
        edge_transmission_delays = np.zeros((self.num_devices, self.num_edges + 1), dtype=np.float32)
        num_blocks_obs = np.zeros(self.num_devices, dtype=np.int32)
        allocated_edge_computing = np.zeros(self.num_edges, dtype=np.float32)
        allocated_cloud_computing = np.zeros(1, dtype=np.float32)
        block_compute_obs = np.zeros((self.num_devices, 5), dtype=np.float32)
        block_accuracy_obs = np.zeros( (self.num_devices, 5), dtype=np.float32)
        task_arrival_times = np.zeros(self.num_devices, dtype=np.float32)
        
        # Feasible edge caches matrix (num_devices x num_edges)
        feasible_edge_caches = np.zeros((self.num_devices, self.num_edges), dtype=np.int32)
        
        # Thermal noise spectral density for data rate calculation
        N0_w_per_hz = 10 ** ((-174 - 30) / 10)
        
        # Normalize compute requirements by allocated_edge_computing
        compute_normalization = max(edge.core_computing_power for edge in self.edge_servers)
        # Get allocated computing resources
        allocated_edge_computing = [edge_server.core_computing_power/compute_normalization for edge_server in self.edge_servers]
        allocated_cloud_computing = [self.cloud_server.computing_power/compute_normalization]
        for task in tasks:
            d_id = task.device_id
            dev = self.mobile_devices[d_id]
            model_idx = self.task_types.index(task.task_type)
            
            # 1. Edge-specific Transmission Delays (Upload + Relay)
            sigma_squared = N0_w_per_hz * dev.bandwidth
            sinr = (0.2 * dev.channel_gain) / sigma_squared
            rate = dev.bandwidth * np.log2(1 + sinr) # bps
            upload_delay = (task.input_size * 8 * 1e6) / rate
            
            for e_id in range(self.num_edges):
                relay_delay = self.calculate_relay(task.input_size, dev.edge_id, e_id)
                edge_transmission_delays[d_id, e_id] = upload_delay + relay_delay
            
            # Cloud Delay (Upload to connected edge + Cloud connection delay)
            edge_transmission_delays[d_id, self.num_edges] = upload_delay + self.cloud_server.calculate_transmission_delay(task.input_size)
            
            # 2. Model info
            num_blocks_obs[d_id] = task.task_type.num_blocks
            # Normalize task arrival time relative to current time
            task_arrival_times[d_id] = task.arrival_time - self.time
            # Cumulative compute requirements
            for block_id in range(task.task_type.num_blocks):
                block_compute_obs[d_id, block_id] = task.task_type.blocks[block_id].compute_requirement / compute_normalization
            # Accuracies
            accs = [b.accuracy for b in task.task_type.blocks]
            block_accuracy_obs[d_id, :len(accs)] = accs

            # 3. Feasible edge caches
            for e_id, edge in enumerate(self.edge_servers):
                if model_idx in edge.cached_models:
                    feasible_edge_caches[d_id, e_id] = 1
        # Gather edge core completion times, normalized relative to current time (shape: num_edges x edge_cores)
        # Only subtract time from non-zero values
        edge_core_completion_times = np.zeros((self.num_edges, len(self.edge_servers[0].core_finish_times)), dtype=np.float32)
        for e_id, edge in enumerate(self.edge_servers):
            for core_id, finish_time in enumerate(edge.core_finish_times):
                edge_core_completion_times[e_id, core_id] = max(0, finish_time - self.time)
        
        obs = {
            "edge_transmission_delays": edge_transmission_delays,
            "feasible_edge_caches": feasible_edge_caches,
            "num_blocks": num_blocks_obs,
            "allocated_edge_computing": allocated_edge_computing,
            "allocated_cloud_computing": allocated_cloud_computing,
            "block_compute_requirement": block_compute_obs,
            "block_accuracy": block_accuracy_obs,
            "edge_core_completion_times": edge_core_completion_times,
            "task_arrival_times": task_arrival_times
        }
        return obs

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.num_steps = 0
        self.pending_tasks = []
        # Reset edge servers
        for edge_server in self.edge_servers:
            edge_server.reset_server()
        # Process events until we have tasks for the first timestep
        self.__generate_timestep_tasks()  # Change from _run_to_next_decision()
        # Construct initial observation from pending_tasks
        obs = self._get_obs(self.pending_tasks)
        info = self._get_info()
        return obs, info
        # return self.get_state()

    def clear_info(self):
        # Reset edge servers
        for edge_server in self.edge_servers:
            edge_server.core_finish_times = [t if t > self.time else 0 for t in edge_server.core_finish_times]

    def cache_random_models(self):
        """
        Randomly selects models to cache while respecting storage constraints.
        Returns: np.array of binary decisions (1 = cache, 0 = don't cache)
        """
        cache_decisions = []
        for edge_server in self.edge_servers:
            cache_decision = np.zeros(self.num_models, dtype=int)
            available_storage = edge_server.cache_storage
            model_indices = np.arange(self.num_models)
            self.rng.shuffle(model_indices)
            for idx in model_indices:
                model_size = self.task_types[idx].model_size
                if model_size <= available_storage:
                    cache_decision[idx] = 1
                    available_storage -= model_size
            cache_decisions.extend(cache_decision)
        return cache_decisions
    
    def count_model_requests(self, tasks):
        """
        Count how many times each model is requested in the current timestep.
        Returns: np.array of counts, shape (num_models,)
        """
        counts = np.zeros(self.num_models, dtype=int)
        for task in tasks:
            model_idx = self.task_types.index(task.task_type)
            counts[model_idx] += 1
        return counts
    
    def calculate_caching_gain(self, tasks):
        """
        Calculates the potential caching gain for each model requested in the current step.
        Gain = Latency(Cloud) - Latency(Edge, if cached)
        """
        gains = np.zeros(self.num_models, dtype=np.float32)
        counts = np.zeros(self.num_models, dtype=int)
        for task in tasks:
            model_idx = self.task_types.index(task.task_type)
            device = self.mobile_devices[task.device_id]
            connected_edge = self.edge_servers[device.edge_id]
            # Use the final exit point to represent the full model computation
            exit_decision = task.task_type.num_blocks - 1
            # 1. Simulate latency if processed at the connected edge server
            # We pass a copy of the server to avoid modifying its actual state during this hypothetical calculation
            _, _, _, edge_latency = self.simulate_latency(task, exit_decision, copy.deepcopy(connected_edge), True, "edge", queueing=False)
            # 2. Simulate latency if processed at the cloud
            _, _, _, cloud_latency = self.simulate_latency(task, exit_decision, connected_edge, True, "cloud", queueing=False)
            # 3. Calculate gain and update
            gain = cloud_latency - edge_latency
            gains[model_idx] += gain
            counts[model_idx] += 1

        # Return the average gain for each model
        # Add a small epsilon to avoid division by zero
        return gains

    def update_model_popularity(self, popularity_history, tasks):
        """
        Update the running sum of model requests.
        Args:
            popularity_history: np.array of shape (num_models,)
            tasks: list of Task objects for this timestep
        Returns:
            Updated popularity_history (np.array)
        """
        counts = self.count_model_requests(tasks)
        return popularity_history + counts

    def select_popular_models_to_cache(self, popularity_history, storage_limit):
        """
        Select models to cache based on popularity and storage constraint.
        Args:
            popularity_history: np.array of shape (num_models,)
            storage_limit: int, total cache storage available (same unit as model_size)
        Returns:
            cache_decision: np.array of 0/1, shape (num_models,)
        """
        model_indices = np.argsort(-popularity_history)  # Descending order
        cache_decision = np.zeros(self.num_models, dtype=int)
        used_storage = 0
        for idx in model_indices:
            model_size = self.task_types[idx].model_size
            if used_storage + model_size <= storage_limit:
                cache_decision[idx] = 1
                used_storage += model_size
        return cache_decision

    def _generate_channel_gain(self, distance_km):
        """
        Realistic log-normal path loss model (urban macrocell)
        Path loss in dB = 128.1 + 37.6 * log10(d), where d in km
        Assume distance d ~ Uniform(0.05, 0.5) km
        """
        path_loss_db = 128.1 + 37.6 * np.log10(distance_km)
        fading_db = self.rng.normal(loc=0, scale=6)  # Shadowing (optional)
        total_loss_db = path_loss_db  # + fading_db
        channel_gain_linear = 10 ** (-total_loss_db / 10)
        return channel_gain_linear

    def _calculate_bandwidth_allocation(self):
        """
        Equally divide the total edge server bandwidth among all devices.
        Returns a list/array of bandwidth allocations (Hz) for each device.
        """
        #equal_bandwidth = self.edge_server.bandwidth / self.num_devices
        equal_bandwidth = 2*1e6
        return np.full(self.num_devices, equal_bandwidth, dtype=np.float32)

    def get_state(self):
        return {
            "time": self.time,
            "device_tasks": [d["input_size"] for d in self.mobile_devices],
            "channel_gains": [d["channel_gain"] for d in self.mobile_devices],
        }
    
    def calculate_relay(self, size_mb, edge_from, edge_to):
        """
        Calculate relay delay between two edge servers based on hop distance.
        """
        if edge_from == edge_to:
            return 0
        hops = self.edge_connections.get((edge_from, edge_to), 0)
        relay_per_hop = size_mb / self.i2i_transmit_rate  # Assume 10 Mbps relay rate per hop
        relay_time = hops * relay_per_hop
        return relay_time
    
    def update_edge_cache(self, edge_server, new_model_idx, popularity_history):
        """
        Update the cache of an edge server:
        - Add new_model_idx to the cache.
        - Remove least popular models if needed to fit within storage.
        - Always keep the most popular models cached.
        Args:
            edge_server: EdgeServer object
            new_model_idx: int, index of the model to add
            popularity_history: np.array of shape (num_models,)
        """
        # Get current cached models and their sizes
        cached_models = set(getattr(edge_server, "cached_models", []))
        cached_models.add(new_model_idx)
        # Sort models by popularity (descending)
        sorted_models = sorted(list(cached_models), key=lambda idx: -popularity_history[idx])
        used_storage = 0
        new_cache = []
        for idx in sorted_models:
            model_size = self.task_types[idx].model_size
            if used_storage + model_size <= edge_server.cache_storage:
                new_cache.append(idx)
                used_storage += model_size
            else:
                continue  # Skip if not enough space
        edge_server.cached_models = new_cache
        return new_cache

    def calculate_download_delay(self, model):
        """
        Calculate download delay from cloud server to mobile device.
        """
        size = model.model_size * 1000 
        download_time = size / self.cloud_server.download_rate
        return download_time

    def calculate_transmission_delay(self, size_mb, bandwidth_hz, channel_gain, target_edge, connected_edge):
        """
        Calculate transmission delay in seconds based on task size, bandwidth, and channel gain.
        """
        # Thermal noise spectral density (constant)
        N0_dbm_per_hz = -174  # dBm/Hz
        # Convert -174 dBm/Hz → Watts/Hz
        # Formula: W = 10^((dBm - 30) / 10)
        N0_w_per_hz = 10 ** ((N0_dbm_per_hz - 30) / 10)
        # Final noise power sigma^2
        sigma_squared = N0_w_per_hz * bandwidth_hz
        p_k = 0.2  # Transmission power (W)
        # Convert size from MB to bits
        size_bits = size_mb * 8 * 1e6
        sinr = (p_k * channel_gain) / sigma_squared
        rate = bandwidth_hz * np.log2(1 + sinr)  # bits per second
        #print("Data rate:", rate/(8 * 1e6))
        upload_delay = size_bits / rate  # in seconds
        # Calculate relay delay if offloading to a different edge server
        relay_delay = self.calculate_relay(size_mb, connected_edge, target_edge)
        transmission_delay = upload_delay + relay_delay
        return transmission_delay

    def simulate_latency(self, task, exit_decision, edge_server, offloadable, server_type, queueing):
        # Calculate computing delay based on the task type
        computation = task.get_compute_requirement(exit_decision)
        transmit_size = task.get_transmitted_size(exit_decision)
        device = self.mobile_devices[task.device_id]
        queueing_delay = 0 # No queueing for the cloud
        # Calculate transmission delay from user device to edge server
        target_edge = device.edge_id
        connected_edge = edge_server.edge_id
        transmission_delay = self.calculate_transmission_delay(
            transmit_size, device.bandwidth, device.channel_gain, target_edge, connected_edge
        )
        # Calculate delay        
        if server_type == "cloud":
            # Add cloud server transmission delay
            transmission_delay += self.cloud_server.calculate_transmission_delay(transmit_size)
            computing_delay = self.cloud_server.calculate_computing_delay(computation)
        else:
            processing_time = edge_server.calculate_computing_delay(computation)
            if queueing:
                # Use task's actual arrival time, not timestep boundary
                task_arrival_at_server = task.arrival_time + transmission_delay
                # Get the queueing delay (waiting time) from the server
                queueing_delay = edge_server.get_queuing_delay(processing_time, task_arrival_at_server)
                # If waiting time exceeds 1 second, offload to cloud instead
                if queueing_delay > 1.0:
                    transmission_delay += self.cloud_server.calculate_transmission_delay(transmit_size)
                    computing_delay = self.cloud_server.calculate_computing_delay(computation)
                    queueing_delay = 0  # Reset queueing delay since we're using cloud
                else:
                    computing_delay = queueing_delay + processing_time
                    # Update core finish time
                    edge_server.process_task_on_core(computation, task_arrival_at_server)
            else:
                computing_delay = processing_time
            if not offloadable:
                print("Something is wrong here")
                # If offloading to edge server is not possible, use cloud server
                # self.update_edge_cache(edge_server, self.task_types.index(task.task_type), self.popularity_history)
                # download_time = self.calculate_download_delay(task.task_type)
                # transmission_delay = max(transmission_delay, download_time)
        total_delay = computing_delay + transmission_delay
        return computing_delay, transmission_delay, queueing_delay, total_delay
    
    def simulate_accuracy(self, task, exit_decision):
        accuracy = task.get_accuracy(exit_decision)
        return accuracy
    
    def _update_server_request_counters(self, tasks):
        """
        Update request counters for each edge server based on the tasks processed.
        The counter is incremented based on the device's connected edge server,
        not the offloading decision.
        
        Args:
            tasks: list of Task objects processed in this timestep
        """
        for task in tasks:
            # Get the device that generated this task
            device_id = task.device_id
            device = self.mobile_devices[device_id]
            # The connected edge server is where the request originated
            edge_id = device.edge_id
            edge_server = self.edge_servers[edge_id]
            # Get the model index from the task by comparing task_type objects using identity
            model_idx = self.task_types.index(task.task_type)
            # Update the edge server's request counter for this model
            edge_server.update_request_counter(model_idx)

    def step(self, actions):
        # Clear task queues from previous timestep
        for edge_server in self.edge_servers:
            edge_server.clear_task_queue()
        self.cloud_server.clear_task_queue()
        
        # Process pending tasks and track waiting times
        tasks_to_process = self.pending_tasks
        num_actions = len(tasks_to_process)

        rewards = 0
        avg_accuracy = 0
        avg_latency = 0
        avg_compute_latency = 0
        avg_transmit_latency = 0
        avg_waiting_time = 0
        # Track per-step cache stats
        step_cache_hits = 0
        step_cache_requests = 0
        
        if num_actions == 0:
            # Advance time and process next timestep
            self.time += self.time_step_duration
            self.__generate_timestep_tasks()
            obs = self._get_obs(self.pending_tasks)
            infos = {
                "accuracy": 0.0, "latency": 0.0, "compute_latency": 0.0,
                "transmit_latency": 0.0, "waiting_time": 0.0, "cache_hit_rate": 0.0,
                "cumulative_cache_hit_rate": 0.0, "num_tasks": 0,
                "current_time": self.time,
            }
            return obs, 0, False, self.time >= 2000, infos
        
        task_offload_decisions = []
        task_exit_decisions = []
        for task in tasks_to_process:
            device_id = task.device_id
            # Extract the offload decision for this device
            offload_decision = actions[device_id]
            task_offload_decisions.append(offload_decision)
            # Extract the exit decision for this device
            exit_decision = actions[self.num_devices + device_id]
            # Mask exit decision to valid range
            if exit_decision >= task.task_type.num_blocks:
                exit_decision = task.task_type.num_blocks - 1
            task_exit_decisions.append(exit_decision)
        
        # Phase 2: Determine offloadability FIRST, THEN assign to queues
        masked_server_types = []
        server_assignments = []
        
        for task, edge_id in zip(tasks_to_process, task_offload_decisions):
            server_type = "cloud"  # Default to cloud
            server = None
            
            if edge_id < self.num_edges:
                # Task wants to offload to edge - check if possible
                edge_server = self.edge_servers[edge_id]
                model_idx = self.task_types.index(task.task_type)
                step_cache_requests += 1
                offloadable = edge_server.check_offloadable(task, self.task_types)
                is_hit = int(offloadable)
                step_cache_hits += is_hit
                edge_server.cache_hits = getattr(edge_server, "cache_hits", 0) + is_hit
                
                if offloadable:
                    # Model is cached on edge - can offload
                    server_type = "edge"
                    server = edge_server
                else:
                    # Model not cached - must go to cloud
                    server_type = "cloud"
                    server = self.cloud_server
            else:
                # Edge ID >= num_edges means explicit cloud request
                server_type = "cloud"
                server = self.cloud_server
            
            masked_server_types.append(server_type)
            server_assignments.append(server)
        
        # NOW add tasks to queues based on final server assignments
        for task, server, server_type in zip(tasks_to_process, server_assignments, masked_server_types):
            server.add_task_to_queue(task)
        
        # Iterate through each device's action and task
        for exit_decision, edge_id, task, server_type, server in zip(
            task_exit_decisions, task_offload_decisions, tasks_to_process, masked_server_types, server_assignments
        ):
            if exit_decision >= task.task_type.num_blocks:
                exit_decision = task.task_type.num_blocks-1
            # Calculate latency
            compute_latency, transmit_latency, waiting_time, latency = self.simulate_latency(task, exit_decision, server, True, server_type, queueing=True)
            accuracy = self.simulate_accuracy(task, exit_decision)
            # Normalize accuracy and latency using min-max normalization
            normalized_accuracy = (accuracy - self.min_accuracy) / (self.max_accuracy - self.min_accuracy) if self.max_accuracy > self.min_accuracy else 0.0
            normalized_latency = (latency - self.min_latency) / (self.max_latency - self.min_latency) if self.max_latency > self.min_latency else 0.0
            if self.exit_oblivious:
                rewards += -self.latency_weight * normalized_latency
            else:
                rewards += -self.latency_weight * normalized_latency + self.acc_weight * normalized_accuracy
            avg_accuracy += accuracy
            avg_latency += latency
            avg_compute_latency += compute_latency
            avg_transmit_latency += transmit_latency
            avg_waiting_time += waiting_time
        #Normalize metrics
        if num_actions > 0:
            rewards /= num_actions
            avg_accuracy /= num_actions
            avg_latency /= num_actions
            avg_compute_latency /= num_actions
            avg_transmit_latency /= num_actions
            avg_waiting_time /= num_actions
        # Advance time by one timestep BEFORE processing next timestep
        self.time += self.time_step_duration
        self.num_steps += 1
        # Example: terminate after 1000 steps
        terminated = False
        truncated = self.time >= 2000
        
        # Start caching after processing the tasks
        # --- Track model requests for popularity-based caching ---
        timestep_counts = self.count_model_requests(tasks_to_process)
        self.popularity_window.append(timestep_counts)
        if len(self.popularity_window) > self.popularity_window_size:
            self.popularity_window.pop(0)
        #popularity_sum = np.sum(self.popularity_window, axis=0)
        # ---- Calculate caching gain for gain-based caching ---
        caching_gains = self.calculate_caching_gain(tasks_to_process)
        self.caching_gain_window.append(caching_gains)
        if len(self.caching_gain_window) > self.caching_gain_window_size:
            self.caching_gain_window.pop(0)
        
        # --- Update caches if not frozen (slow timescale) ---
        if not getattr(self, "freeze_cache_updates", False):
            # If cache is oblivious, use random caching
            if self.cache_oblivious:
                # --- Use random caching if desired ---
                cache_decisions = self.cache_random_models()
                # cache_decisions is a flat list for all edge servers
                for edge_id, edge_server in enumerate(self.edge_servers):
                    # Get the slice for this edge server
                    start = edge_id * self.num_models
                    end = (edge_id + 1) * self.num_models
                    edge_cache = []
                    for model_idx, flag in enumerate(cache_decisions[start:end]):
                        if flag:
                            edge_cache.append(model_idx)
                edge_server.cached_models = edge_cache

        self.__generate_timestep_tasks()
        obs = self._get_obs(self.pending_tasks)
        self.clear_info()
        #obs = self.reset()
        step_hit_rate = (step_cache_hits / step_cache_requests) if step_cache_requests > 0 else 0.0
        cum_requests = sum(getattr(e, "total_requests", 0) for e in self.edge_servers)
        cum_hits = sum(getattr(e, "cache_hits", 0) for e in self.edge_servers)
        cum_hit_rate = (cum_hits / cum_requests) if cum_requests > 0 else 0.0
        # Update server request counters based on the connected edge of each device
        self._update_server_request_counters(tasks_to_process)
        infos = {
            "num_tasks": num_actions,
            "accuracy": avg_accuracy,
            "latency": avg_latency,
            "compute_latency": avg_compute_latency,
            "transmit_latency": avg_transmit_latency,
            "waiting_time": avg_waiting_time,
            "cache_hit_rate": step_hit_rate,
            "cumulative_cache_hit_rate": cum_hit_rate,
            "rewards": rewards
        }
        return obs, rewards, terminated, truncated, infos

    def save_state(self):
        import copy
        # Save RNG state separately
        state = copy.deepcopy(self.__dict__)
        if hasattr(self.rng, 'bit_generator'):
            state['_rng_state'] = self.rng.bit_generator.state
        return state

    def load_state(self, state):
        import copy
        # Restore all attributes
        rng_state = state.pop('_rng_state', None)
        self.__dict__ = copy.deepcopy(state)
        if rng_state is not None and hasattr(self.rng, 'bit_generator'):
            self.rng.bit_generator.state = rng_state

if __name__ == "__main__":
    gym.register(
        id="MECOffload-v0",
        entry_point="gym_environment:Environment",
    )

    # Then later:
    env = gym.make(
        "MECOffload-v0",
        num_devices=30,
        num_edges=3,
        num_models=10,
        local_computing_range=[1, 2],
        cloud_computing=5,
        edge_computing=30,
        edge_storage=20,
        bandwidth=20,
        i2i_tranmit_rate=10,
        cloud_download_rate=100,
        models_size_range=[2, 4],
        task_size_input=[0.5, 1],
        seed=42,
        cache_oblivious=False,
        exit_oblivious=False,
        offload_oblivious=False,
    )