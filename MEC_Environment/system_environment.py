import numpy as np


from .tasks import Block, DNN_Model, Task
from .devices import CloudServer, EdgeServer, MobileDevice


class Eval_Environment():
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
        models_size_range,
        task_size_input,
        edge_cores=8,
        zipf_a=1.2,
        acc_weight=0.5,
        offload_oblivious=False,
        exit_oblivious=False,
        cache_oblivious=False,
        seed=42,
    ):
        # Set the random seed for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Initialize the multi-user, single-edge environment
        self.num_edges = num_edges
        self.num_devices = num_devices
        self.i2i_transmit_rate = i2i_tranmit_rate # in Mbps
        self.cloud_server = CloudServer(cloud_computing, 5, cloud_download_rate)
        self.edge_servers = []
        for edge_id in range(self.num_edges):
            self.edge_servers.append(EdgeServer(edge_computing, bandwidth, edge_storage, edge_cores, edge_id))
        # Initialize cache counters
        for edge in self.edge_servers:
            edge.cache_hits = 0
            edge.total_requests = 0
        # Define edge server connections and hop distances
        self.edge_connections = {
            (0, 1): 1,  # Edge server 0 and 1 are 1 hop away
            (1, 0): 1,  # Edge server 1 and 0 are 1 hop away
            (0, 2): 2,  # Edge server 0 and 2 are 2 hops away
            (2, 0): 2,  # Edge server 2 and 0 are 2 hops away
            (1, 2): 1,  # Edge server 1 and 2 are 1 hop away
            (2, 1): 1   # Edge server 2 and 1 are 1 hop away
        }
        # Initialize multiple mobile devices
        self.mobile_devices = []
        for i in range(num_devices):
            local_computing = int(self.rng.uniform(*local_computing_range))
            distance_km = self.rng.uniform(0.05, 0.2)
            channel_gain = self._generate_channel_gain(distance_km)
            edge_id = self.rng.choice(self.num_edges, p=[0.5, 0.3, 0.2])
            self.edge_servers[edge_id].connect_device()
            self.mobile_devices.append(
                MobileDevice(
                    i, distance_km, channel_gain, local_computing, edge_id
                )
            )
        # Allocate bandwidth for connected devices
        for i in range(num_devices):
            edge_id = self.mobile_devices[i].edge_id
            allocated_bandwidth = self.edge_servers[edge_id].calculate_bandwidth_allocated()
            self.mobile_devices[i].assign_bandwidth(allocated_bandwidth)  # Convert Hz to MHz
        self.num_models = num_models
        # Generate Zipf rankings for model popularity at each edge server
        self.zipf_rankings = self.generate_zipf_rankings(self.num_models, self.num_edges)
        #self.task_type = [DNN_Model]
        self.acc_weight = acc_weight
        self.tasks = []
        self.task_types = self.generate_task_type(models_size_range)
        self.task_size_input = task_size_input
        self.zipf_a = zipf_a
        self.time = 0
        # Define code for tracking model popularity
        self.popularity_history = np.zeros(self.num_models, dtype=int)
        self.popularity_window = []  # To keep track of last N timesteps
        self.popularity_window_size = 20  # Or any window size you want
        # Define special case when cache or early exit is not considered
        self.offload_oblivious = offload_oblivious
        self.exit_oblivious = exit_oblivious
        self.cache_oblivious = cache_oblivious
        self.freeze_cache_updates = False

    # def generate_blocks(self, num_blocks=None):
    #     if num_blocks is None:
    #         num_blocks = self.rng.integers(3, 6)  # between 3–5

    #     # Accuracy curve: saturating growth
    #     start_acc = self.rng.uniform(0.65, 0.7)
    #     final_acc = self.rng.uniform(0.90, 0.99)
    #     steps = np.arange(num_blocks)
    #     accuracies = start_acc + (final_acc - start_acc) * (1 - np.exp(-0.8 * steps))

    #     # Compute curve: grows with depth
    #     base_compute = 2e8
    #     computes = [base_compute * (1.5 ** i) for i in range(num_blocks)]

    #     return [Block(c, a) for c, a in zip(computes, accuracies)]

    def generate_blocks(self, num_blocks=None, total_compute=None):
        if num_blocks is None:
            num_blocks = self.rng.integers(3, 6)  # 3, 4, or 5

        # --- Exact base values from your table ---
        base_accuracies = {
            3: [0.3162, 0.5584, 0.9173],
            4: [0.3426, 0.6219, 0.8523, 0.9143],
            5: [0.3169, 0.6008, 0.8146, 0.8892, 0.9265]
        }

        base_compute_percentages = [
            0.1498,  # 14.98%
            0.3420,  # 34.20%
            0.6613,  # 66.13%
            0.9384,  # 93.84%
            1.0000   # 100%
        ]

        # --- Load base values for this num_blocks ---
        base_accs = np.array(base_accuracies[num_blocks])
        base_pcts = np.array(base_compute_percentages[:num_blocks])

        # --- Add variance to accuracies ---
        acc_noise_level = 0.015  # ±1.5% absolute (adjust as needed)
        acc_noise = self.rng.uniform(-acc_noise_level, acc_noise_level, size=num_blocks)
        accuracies = np.clip(base_accs + acc_noise, 0.0, 1.0)

        # --- Add variance to compute percentages (with constraints) ---
        pct_noise_level = 0.03  # ±3% relative noise
        pct_noise = base_pcts * self.rng.uniform(-pct_noise_level, pct_noise_level, size=num_blocks)

        # Apply noise, but enforce:
        noisy_pcts = base_pcts + pct_noise

        # 1. Ensure non-decreasing order
        for i in range(1, num_blocks):
            noisy_pcts[i] = max(noisy_pcts[i], noisy_pcts[i-1])

        # 2. Force last block to be exactly 1.0 (100%)
        noisy_pcts[-1] = 1.0

        # 3. Clip to [0, 1] for safety
        noisy_pcts = np.clip(noisy_pcts, 0.0, 1.0)

        # --- Scale to absolute compute ---
        # You can also sample total_compute from a distribution if desired
        #total_compute = 2e8 * (1.5 ** (num_blocks - 1))  # or self.rng.uniform(1.8e8, 2.5e8) * (1.5**(num_blocks-1))
        computes = noisy_pcts * total_compute

        return [Block(c, a) for c, a in zip(computes, accuracies)]

    def select_blocks(self, blocks):
        combinations = [
            [0, 1, 3],
            [0, 1, 2, 4],
            [0, 1, 2, 3, 4],
        ]
        idx = self.rng.integers(0, len(combinations))
        chosen_indices = combinations[idx]
        return [blocks[i] for i in chosen_indices if i < len(blocks)]

    def generate_task_type(self, models_size_range):
        task_types = []
        for _ in range(self.num_models):
            all_blocks = self.generate_blocks()
            task_type = DNN_Model(
                model_size=round(self.rng.uniform(*models_size_range), 1),  # in GB
                blocks=self.select_blocks(all_blocks),
            )
            task_types.append(task_type)
        return task_types

    def generate_tasks(self):
        """
        Generate a list of Task objects for each device at the current timestep,
        sampling task types according to Zipf distribution with per-edge ranking.
        """
        self.tasks = []
        for dev in self.mobile_devices:
            edge_id = dev.edge_id
            ranking = self.zipf_rankings[edge_id]
            # Generate Zipf probabilities
            ranks = np.arange(1, self.num_models + 1)
            probs = 1 / np.power(ranks, self.zipf_a)
            probs /= probs.sum()
            # Sample a rank according to Zipf
            sampled_rank = self.rng.choice(ranks, p=probs) - 1  # 0-based
            model_idx = ranking[sampled_rank]
            task_type = self.task_types[model_idx]
            task = Task(
                device_id=dev.device_id,
                input_size=self.rng.uniform(*self.task_size_input),
                task_type=task_type,
            )
            self.tasks.append(task)
        # --- Sort tasks by model popularity (descending) ---
        # Get model indices for each task
        task_model_indices = [self.task_types.index(task.task_type) for task in self.tasks]
        # Get popularity for each model
        task_popularities = [self.popularity_history[idx] for idx in task_model_indices]
        # Sort tasks by popularity (descending)
        sorted_tasks = [task for _, task in sorted(zip(task_popularities, self.tasks), key=lambda x: -x[0])]
        self.tasks = sorted_tasks

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

    def _get_obs(self, tasks):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        obs = {
            "time": np.array([self.time], dtype=np.float32),  # Scalar: current timestep
            # Per-device: input sizes (MB)
            "input_sizes": np.array(
                [task.input_size for task in tasks], dtype=np.float32
            ),
            # Per-device: bandwidth allocation (Hz)
            "bandwidth_allocation": np.array(
                [dev.bandwidth for dev in self.mobile_devices], dtype=np.float32
            ),
            # Per-device: wireless channel gain (linear)
            "channel_gains": np.array(
                [dev.channel_gain for dev in self.mobile_devices], dtype=np.float32
            ),
            # Global: DNN model size (GB)
            "edge_cached_models": np.array([
                [1 if model_idx in edge.cached_models else 0 for model_idx in range(self.num_models)]
                for edge in self.edge_servers
                ], dtype=np.int32),
            "edge_computing_capacity": np.array(
                [edge.computing_power for edge in self.edge_servers],
                dtype=np.float32,
            ),  # Global: local computing capacity (FLOPS)
            "connected_edge_servers": np.array([
                [1 if dev.edge_id == edge_id else 0 for dev in self.mobile_devices]
                for edge_id in range(self.num_edges)
            ], dtype=np.int32),
            # "edge_computing_capacity": np.array(
            #     [self.edge_server.computing_power], dtype=np.float32
            # ),  # Global: edge computing capacity (FLOPS)
            "num_blocks": np.array(
                [task.task_type.num_blocks for task in tasks], dtype=np.int32
            ),
            "block_compute_requirement": np.array(
                [
                    np.pad(
                        [block.compute_requirement for block in task.task_type.blocks],
                        (0, 5 - task.task_type.num_blocks),
                        mode="constant",
                        constant_values=999,
                    )
                    for task in tasks
                ],
                dtype=np.float32,
            ),
            "block_accuracy": np.array(
                [
                    np.pad(
                        [block.accuracy for block in task.task_type.blocks],
                        (0, 5 - task.task_type.num_blocks),
                        mode="constant",
                        constant_values=999,
                    )
                    for task in tasks
                ],
                dtype=np.float32,
            ),
        }
        return obs

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {}

    def reset(self, seed=None, options=None):
        self.time = 0
        for edge_server in self.edge_servers:
            edge_server.reset_server()
        # Construct initial observation
        self.generate_tasks()
        obs = self._get_obs(self.tasks)
        info = self._get_info()
        return obs, info
        # return self.get_state()

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
            np.random.shuffle(model_indices)
            for idx in model_indices:
                model_size = self.task_types[idx].model_size
                if model_size <= available_storage:
                    cache_decision[idx] = 1
                    available_storage -= model_size
            cache_decisions.extend(cache_decision)
        return cache_decisions
    
    # ...existing code...

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
    # ...existing code...
    
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
    
    def calculate_download_delay(self, model):
        """
        Calculate download delay from cloud server to mobile device.
        """
        size = model.model_size * 1000 
        download_time = size / self.cloud_server.download_rate
        return download_time
    
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

    def calculate_transmission_delay(self, size_mb, bandwidth_hz, channel_gain, target_edge, connected_edge, offloadable):
        """
        Calculate transmission delay in seconds based on task size, bandwidth, and channel gain.
        """
        # Assume a simple model for transmission delay
        sigma_squared = 1e-13  # Noise power (W)
        p_k = 0.1  # Transmission power (W)
        # Convert size from MB to bits
        size_bits = size_mb * 8 * 1e6
        sinr = (p_k * channel_gain) / sigma_squared
        rate = bandwidth_hz * np.log2(1 + sinr)  # bits per second
        # print("Data rate:", rate/(8 * 1e6))
        upload_delay = size_bits / rate  # in seconds
        # Calculate relay delay if offloading to a different edge server
        relay_delay = self.calculate_relay(size_mb, connected_edge, target_edge)
        transmission_delay = upload_delay + relay_delay
        return transmission_delay

    def simulate_latency(self, task, exit_decision, edge_server, offloadable, server_type):
        # Calculate computing delay based on the task type
        computation = task.get_compute_requirement(exit_decision)
        computing_delay = edge_server.calculate_computing_delay(computation)
        # Calculate transmission delay
        transmit_size = task.get_transmitted_size(exit_decision)
        device = self.mobile_devices[task.device_id]
        if server_type == "cloud":
           transmission_delay = self.calculate_transmission_delay(
                transmit_size, device.bandwidth, device.channel_gain, device.edge_id, device.edge_id, True
            )
           transmission_delay += self.cloud_server.calculate_transmission_delay(transmit_size)
        else:
            target_edge = device.edge_id
            connected_edge = edge_server.edge_id
            transmission_delay = self.calculate_transmission_delay(
                transmit_size, device.bandwidth, device.channel_gain, target_edge, connected_edge, offloadable
            )
            if not offloadable:
                # If offloading to edge server is not possible, use cloud server
                self.update_edge_cache(edge_server, self.task_types.index(task.task_type), self.popularity_history)
                download_time = self.calculate_download_delay(task.task_type)
                transmission_delay = max(transmission_delay, download_time)
        total_delay = computing_delay + transmission_delay
        return computing_delay, transmission_delay, total_delay
    
    def simulate_accuracy(self, task, exit_decision):
        accuracy = task.get_accuracy(exit_decision)
        return accuracy

    def step(self, actions):
        rewards = 0
        avg_accuracy = 0
        avg_latency = 0
        avg_compute_latency = 0
        avg_transmit_latency = 0
        # Track per-step cache stats
        step_cache_hits = 0
        step_cache_requests = 0
        # Convert actions to numpy array
        # If exit oblivious, set offload actions to 0 (offload) or 1 (local execution)
        if self.offload_oblivious:
            offload_actions = [self.mobile_devices[task.device_id].edge_id for task in self.tasks]
            exit_decisions = actions
        elif self.exit_oblivious:
            offload_actions = actions
            exit_decisions = [task.task_type.num_blocks-1 for task in self.tasks] # Index start at 0
        else:
            offload_actions = actions[:self.num_devices]
            exit_decisions = actions[self.num_devices:self.num_devices*2]


        offloadable_mask = [self.edge_servers[self.mobile_devices[task.device_id].edge_id].check_offloadable(task, self.task_types) for task in self.tasks]
        offloadable_rate = np.mean(offloadable_mask)

        # Print model requests and cache status at each edge server for debugging
        # print("\n--- Offloading Requests and Edge Server Cache Status ---")
        # edge_requests = {edge.edge_id: [] for edge in self.edge_servers}
        # for task, edge_id in zip(self.tasks, offload_actions):
        #     model_idx = self.task_types.index(task.task_type)
        #     edge_requests[edge_id].append(model_idx)

        # for edge_server in self.edge_servers:
        #     edge_id = edge_server.edge_id
        #     requested_models = edge_requests[edge_id]
        #     print(f"Edge Server {edge_id}:")
        #     print(f"  Requested model indices this step: {requested_models}")
        #     print(f"  Cached model indices: {getattr(edge_server, 'cached_models', [])}")

        #print("--- End Debug Info ---\n")
        # Phase 2: Commit all offloadable tasks (for resource allocation)
        # for task, offloadable in zip(self.tasks, offloadable_mask):
        #     if offloadable:
        #         edge_server[edge_id].add_task_to_queue(task)
        server_assignments = []
        server_types = []
        for task, edge_id in zip(self.tasks, offload_actions):
            if edge_id < self.num_edges:
                edge_server = self.edge_servers[edge_id]
                edge_server.add_task_to_queue(task)
                server_assignments.append(edge_server)
                server_types.append("edge")
            else:
                server_assignments.append(self.cloud_server)
                server_types.append("cloud")
        # Iterate through each device's action and task
        for exit_decision, edge_id, task, server_type, server in zip(
            exit_decisions, offload_actions, self.tasks, server_types, server_assignments
        ):
            if server_type == "edge":
                model_idx = self.task_types.index(task.task_type)
                step_cache_requests += 1
                is_hit = 1 if model_idx in getattr(server, "cached_models", []) else 0
                step_cache_hits += is_hit
                server.total_requests = getattr(server, "total_requests", 0) + 1
                server.cache_hits = getattr(server, "cache_hits", 0) + is_hit
                offloadable = server.check_offloadable(task, self.task_types)
                if not offloadable:
                    server_type = "edge"
                    offloadable = True
            else:
                offloadable = True
            # device = self.mobile_devices[task.device_id]
            if exit_decision >= task.task_type.num_blocks:
                exit_decision = task.task_type.num_blocks - 1
            # Calculate latency
            compute_latency, transmit_latency, latency = self.simulate_latency(task, exit_decision, server, offloadable, server_type)
            accuracy = self.simulate_accuracy(task, exit_decision)
            rewards += -latency + self.acc_weight*accuracy
            avg_accuracy += accuracy
            avg_latency += latency 
            avg_compute_latency += compute_latency
            avg_transmit_latency += transmit_latency
        rewards /= self.num_devices
        avg_accuracy /= self.num_devices
        avg_latency /= self.num_devices
        avg_compute_latency /= self.num_devices
        avg_transmit_latency /= self.num_devices
        # Example: terminate after 1000 steps
        terminated = False
        truncated = self.time >= 1000

        # Start caching after processing the tasks
        # --- Track model requests for popularity-based caching ---
        timestep_counts = self.count_model_requests(self.tasks)
        self.popularity_window.append(timestep_counts)
        if len(self.popularity_window) > self.popularity_window_size:
            self.popularity_window.pop(0)
        popularity_sum = np.sum(self.popularity_window, axis=0)
        # --- Update cache decisions for each edge server if not frozen (large timescale) ---
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
            # --- Use popularity-based caching if not cache_oblivious ---
            # For each edge server, select popular models to cache
            else:
                cache_decisions = []
                for edge_server in self.edge_servers:
                    cache_decision_edge = self.select_popular_models_to_cache(
                        popularity_sum, edge_server.cache_storage
                    )
                    # Update the edge server's cached_models list based on cache_decision_edge
                    cached_models = []
                    for model_idx, flag in enumerate(cache_decision_edge):
                        if flag:
                            cached_models.append(model_idx)
                    edge_server.cached_models = cached_models
                    cache_decisions.extend(cache_decision_edge)

        # Generate new tasks and get observation
        for edge_server in self.edge_servers:
            edge_server.reset_server()
        self.generate_tasks()
        obs = self._get_obs(self.tasks)
        # Compute cache hit rates
        step_hit_rate = (step_cache_hits / step_cache_requests) if step_cache_requests > 0 else 0.0
        cum_requests = sum(getattr(e, "total_requests", 0) for e in self.edge_servers)
        cum_hits = sum(getattr(e, "cache_hits", 0) for e in self.edge_servers)
        cum_hit_rate = (cum_hits / cum_requests) if cum_requests > 0 else 0.0
        infos = {
            "accuracy": avg_accuracy,
            "latency": avg_latency,
            "compute_latency": avg_compute_latency,
            "transmit_latency": avg_transmit_latency,
            "offloadable_rate": offloadable_rate,
            "cache_decision": cache_decisions,
            "cache_hit_rate": step_hit_rate,
            "cumulative_cache_hit_rate": cum_hit_rate,
        }
        # Increment time
        self.time += 1
        return obs, rewards, terminated, truncated, infos


if __name__ == "__main__":
    print("Hihi haha")
