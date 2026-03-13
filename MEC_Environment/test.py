import numpy as np
from collections import Counter  # Import Counter

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
        cache_oblivious=False,
        exit_oblivious=False,
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
            edge_id = self.rng.integers(0, self.num_edges)
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
        #self.task_type = [DNN_Model]
        self.tasks = []
        self.task_types = self.generate_task_type(models_size_range)
        self.task_size_input = task_size_input
        self.time = 0
        # Define special case when cache or early exit is not considered
        self.cache_oblivious = cache_oblivious
        self.exit_oblivious = exit_oblivious
        self.model_access_counts = Counter()  # Initialize model access counts with Counter
        self.access_history_length = 20  # Length of access history to consider
        self.model_access_history = [] # Keep track of model access history

    def generate_blocks(self):
        # Create 5 block with random properties
        first_block = Block(compute_requirement=5e8, accuracy=0.6)
        blocks = [first_block]
        for i in range(5):
            block = Block(
                compute_requirement=2e8, # Randomize compute requirement,
                accuracy = 0.6 + (i+1)*0.08, # Randomize accuracy
            )
            blocks.append(block)
        return blocks
    
    def select_blocks(self, blocks):
        # Define the combinations (indices are 0-based)
        combinations = [
            [0, 1, 2, 4],        # [1, 2, 4]
            [0, 1, 2, 3, 5],     # [1, 2, 3, 5]
            [0, 1, 2, 3, 4, 5],  # [1, 2, 3, 4, 5]
        ]

        # Choose one combination randomly (or specify which one you want)
        idx = self.rng.integers(0, len(combinations))
        chosen_indices = combinations[idx]
        selected_blocks = [blocks[i] for i in chosen_indices]
        return selected_blocks

    def generate_task_type(self, models_size_range):
        """
        Generate a random task type from the available task types.
        """
        all_blocks = self.generate_blocks()
        task_types = []
        for _ in range(self.num_models):
            # Randomize model size, compute per layer, and number of layers
            task_type = DNN_Model(
                model_size=int(self.rng.uniform(*models_size_range)), # in GB
                blocks = self.select_blocks(all_blocks),
            )
            task_types.append(task_type)
        return task_types

    def generate_tasks(self):
        """
        Generate a list of Task objects for each device at the current timestep.
        """
        self.tasks = []  # Reset task types for each timestep
        for dev in self.mobile_devices:
            # Example: input_size = 1.5 MB, you can randomize or parameterize as needed
            task = Task(
                device_id=dev.device_id,
                #input_size=1.5,  # in MB
                input_size=self.rng.uniform(*self.task_size_input),  # Randomize input size between 0.5 and 1.5 MB
                task_type=self.rng.choice(self.task_types),  # Assuming a single task type for simplicity
            )
            self.tasks.append(task)

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
            "model_size_gb": np.array(
                [task.task_type.model_size for task in tasks], dtype=np.float32
            ),
            "local_computing_capacity": np.array(
                [dev.local_computing_power for dev in self.mobile_devices],
                dtype=np.float32,
            ),  # Global: local computing capacity (FLOPS)
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
                        (0, 7 - task.task_type.num_blocks),
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
    
    def calculate_download_delay(self, size_mb):
        """
        Calculate download delay from cloud server to mobile device.
        """
        return size_mb / self.cloud_server.download_rate

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

    def simulate_latency(self, task, exit_decision, edge_server, offloadable):
        # Calculate computing delay based on the task type
        computation = task.get_compute_requirement(exit_decision)
        computing_delay = edge_server.calculate_computing_delay(computation)
        # Calculate transmission delay
        transmit_size = task.get_transmitted_size(exit_decision)
        device = self.mobile_devices[task.device_id]
        target_edge = device.edge_id
        connected_edge = edge_server.edge_id
        transmission_delay = self.calculate_transmission_delay(
            transmit_size, device.bandwidth, device.channel_gain, target_edge, connected_edge, offloadable
        )
        if not offloadable:
            # If offloading to edge server is not possible, use cloud server
            model_size = task.type.model_size
            download_time = self.calculate_download_delay(model_size)
            transmission_delay = max(transmission_delay, download_time)
        total_delay = computing_delay + transmission_delay
        return computing_delay, transmission_delay, total_delay
    
    def simulate_accuracy(self, task, exit_decision):
        accuracy = task.get_accuracy(exit_decision)
        return accuracy

    def update_model_access_counts(self, tasks):
        """
        Updates the access counts for each model based on the tasks in the current timestep.
        """
        accessed_models = [task.task_type for task in tasks]
        self.model_access_history.append(accessed_models)
        if len(self.model_access_history) > self.access_history_length:
            self.model_access_history.pop(0)  # Remove oldest entry

        # Recalculate access counts from the history
        self.model_access_counts = Counter()
        for timestep_accesses in self.model_access_history:
            self.model_access_counts.update(timestep_accesses)

    def cache_most_frequent_models(self):
        """
        Selects the most frequently accessed models to cache while respecting storage constraints.
        Returns: np.array of binary decisions (1 = cache, 0 = don't cache)
        """
        cache_decisions = []
        for edge_id, edge_server in enumerate(self.edge_servers):
            cache_decision = np.zeros(self.num_models, dtype=int)
            available_storage = edge_server.cache_storage

            # Sort models by access frequency (most frequent first)
            sorted_models = sorted(
                range(self.num_models),
                key=lambda idx: self.model_access_counts[self.task_types[idx]],
                reverse=True,
            )

            for idx in sorted_models:
                model_size = self.task_types[idx].model_size
                if model_size <= available_storage:
                    cache_decision[idx] = 1
                    available_storage -= model_size
            cache_decisions.extend(cache_decision)
        return cache_decisions

    def step(self, actions):
        rewards = 0
        acc_weight = 0.1 # Weight for accuracy in the reward function
        # Convert actions to numpy array
        offload_actions = actions[:self.num_devices]
        exit_decisions = actions[self.num_devices:self.num_devices*2]
        # If exit oblivious, set offload actions to 0 (offload) or 1 (local execution)
        if self.exit_oblivious:
            offload_actions = [1 if action > 0 else 0 for action in offload_actions]
        # If cache oblivious, cache decision is random
        if self.cache_oblivious:
            cache_decision = self.cache_random_models()
        # If cache is considered, extract cache decisions
        else:
            cache_decision = self.cache_most_frequent_models()
        # Cache model selection
        for edge_id, edge_server in enumerate(self.edge_servers):
            temp_cache = cache_decision[edge_id*self.num_models:(edge_id+1)*self.num_models]
            cached_models = [self.task_types[i] for i in range(self.num_models) if temp_cache[i] == 1]
            edge_server.cache_models(cached_models)

        # Update model access counts based on offloading decisions
        offloaded_tasks = [self.tasks[i] for i in range(self.num_devices) if offload_actions[i] != 0]
        self.update_model_access_counts(offloaded_tasks)

        # Phase 1: Check offloadability (atomic w.r.t. initial state)

        #offloadable_mask = [edge_server.check_offloadable(task) for task in self.tasks]

        # Phase 2: Commit all offloadable tasks (for resource allocation)
        # for task, offloadable in zip(self.tasks, offloadable_mask):
        #     if offloadable:
        #         edge_server[edge_id].add_task_to_queue(task)
        for task, edge_id in zip(self.tasks, offload_actions):
           self.edge_servers[edge_id].add_task_to_queue(task)
        # Iterate through each device's action and task
        for exit_decision, edge_id, task in zip(exit_decisions, offload_actions, self.tasks):
            offloadable = True
            # device = self.mobile_devices[task.device_id]
            if exit_decision > task.task_type.num_blocks:
                exit_decision = task.task_type.num_blocks
            # Calculate latency
            compute_latency, transmit_latency, latency = self.simulate_latency(task, exit_decision, self.edge_servers[edge_id], offloadable)
            accuracy = self.simulate_accuracy(task, exit_decision)
            rewards += -latency + acc_weight*accuracy
        rewards /= self.num_devices
        # Example: terminate after 1000 steps
        terminated = False
        truncated = self.time >= 1000
        # Generate new tasks and get observation
        for edge_server in self.edge_servers:
            edge_server.reset_server()
        self.generate_tasks()
        obs = self._get_obs(self.tasks)
        infos = self._get_info()
        # Increment time
        self.time += 1
        return obs, rewards, terminated, truncated, infos


if __name__ == "__main__":
    print("Hihi haha")