import numpy as np

class CloudServer:
    def __init__(self, computing_power=5, transmit_data_rate=1, download_rate=250, propagation_delay=0.2):
        self.edge_id = -1  # Cloud server identifier (not an edge, use -1)
        self.computing_power = computing_power * 1e9
        self.transmit_data_rate = transmit_data_rate # in MB/s
        self.propagation_delay = propagation_delay # in seconds
        self.download_rate = download_rate # in MB/s
        self.task_queue = []  # Queue of tasks being processed at cloud
        self.num_tasks_submitted = 0
    
    def add_task_to_queue(self, task):
        """Add a task to the cloud queue."""
        self.task_queue.append(task)
        self.num_tasks_submitted += 1
    
    def remove_task_from_queue(self, task):
        """Remove a task from the cloud queue."""
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.num_tasks_submitted -= 1
    
    def clear_task_queue(self):
        """Clear all tasks from the queue (called at start of next timestep)."""
        self.task_queue = []
    
    def calculate_computing_delay(self, required_cycles):
        """
        Calculate the computing delay based on the required cycles and computing power.
        """
        return required_cycles / self.computing_power
    
    def calculate_transmission_delay(self, size_mb):
        """
        Calculate the transmission delay based on the task size.
        """
        transmission_delay = size_mb / self.transmit_data_rate
        return self.propagation_delay + transmission_delay  # Assuming a fixed transmission rate

class EdgeServer:
    def __init__(self, computing_power, bandwidth, storage, edge_cores=8, edge_id=0):
        self.edge_id = edge_id  # Unique identifier for the edge server
        self.edge_cores = edge_cores
        self.computing_power = computing_power * 1e9
        self.core_computing_power = self.computing_power / self.edge_cores
        self.core_finish_times = [0] * self.edge_cores # Track finish times for each core
        self.task_queue = [] # Queue of tasks being processed
        self.num_tasks_submitted = 0
        self.num_devices_connected = 0
        self.cache_storage = storage  # Cache for storing tasks (in GB)
        self.bandwidth = bandwidth * 1e6  # Bandwidth in Hz
        self.cached_models = []  # List of cached models
        self.cache_hits = 0
        self.total_requests = 0
        self.model_request_counts = []  # Dictionary to track model request counts
        # self.edge_queue = []  # List of tuples (device_id, task_id, time, action)]

    def reset_request_counters(self, num_models):
        """
        Initialize the request counters for cache hit rate calculation.
        """
        self.model_request_counts = [0] * num_models
        self.cache_hits = 0
        self.total_requests = 0
    
    def check_cache_available(self, task, task_types):
        """
        Check if the edge server has the required model cached.
        Returns True if the model is cached, False otherwise.
        """
        # Find the model index for the task's model
        try:
            model_idx = task_types.index(task.task_type)
        except ValueError:
            return False
        return model_idx in self.cached_models

    def check_offloadable(self, task, task_types):
        """
        Check if the task can be offloaded to the edge server.
        Returns True if offloadable, False otherwise.
        """    
        #return self.check_cache_available(task)    
        return self.check_cache_available(task, task_types)
    
    def connect_device(self):
        """
        Connect a device to the edge server.
        """
        self.num_devices_connected += 1

    def add_task_to_queue(self, task):
        """
        Add a task to the edge queue if it can be offloaded.
        """
        self.task_queue.append(task)
        self.num_tasks_submitted += 1

    def update_request_counter(self, model_idx):
        """
        Update the request counter for a specific model.
        """
        self.total_requests += 1
        self.model_request_counts[model_idx] += 1

    def remove_task_from_queue(self, task):
        """
        Remove a task from the edge queue when it is completed.
        """
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.num_tasks_submitted -= 1
    
    def clear_task_queue(self):
        """
        Clear all tasks from the queue (called at start of next timestep).
        """
        self.task_queue = []

    def check_caching(self, model_size):
        """
        Check if the model can be cached on the edge server.
        """
        return model_size <= self.cache_storage

    def cache_model(self, model):
        """
        Cache a model on the edge server.
        """
        self.cached_models.append(model)
        self.cache_storage -= model.model_size

    def cache_models(self, list_of_models):
        """
        Cached the models based on the cache decision
        Returns True if caching is successful, stop the caching and return False otherwise.
        """
        for model in list_of_models:
            if self.check_caching(model.model_size):
                self.cache_model(model)
            else:
                return False
        return True
    
    def calculate_bandwidth_allocated(self):
        """
        Calculate the bandwidth allocated to each connected device.
        """
        if self.num_devices_connected == 0:
            return 0
        return self.bandwidth / self.num_devices_connected

    def calculate_computing_delay(self, required_cycles):
        """
        Calculate the computing delay based on the required cycles and core computing power
        """
        allocated_computing = self.core_computing_power
        return required_cycles / allocated_computing
    
    def get_queuing_delay(self, processing_time, current_time):
        """
        Calculate the queuing delay based on the processing time of available cores.
        """
        # Find the earliest available cores
        earliest_finish_time = min(self.core_finish_times)
        # Calculate wait time
        wait_time = max(0, earliest_finish_time - current_time)
        return wait_time
    
    def process_task_on_core(self, required_cycles, current_time):
        # Remove the task from queue and assign it to the earliest available core
        earliest_finish_time = min(self.core_finish_times)
        earliest_core_idx = self.core_finish_times.index(min(self.core_finish_times))
        # Task starts when BOTH the core is free AND the task has arrived
        start_time = max(earliest_finish_time, current_time)
        self.core_finish_times[earliest_core_idx] = start_time + self.calculate_computing_delay(required_cycles)


    def reset_server(self):
        """
        Reset the edge server state.
        """
        self.num_tasks_submitted = 0
        # Only keep future finish times relative to current; reset past/current ones to 0
        # This prevents unbounded accumulation of core finish times across timesteps
        self.core_finish_times = [0] * self.edge_cores
        self.task_queue = []


class MobileDevice:
    def __init__(
        self, device_id, distance_km, channel_gain, local_computing_power, edge_id, bandwidth=5
    ):
        self.device_id = device_id
        self.distance_km = distance_km  # Store distance ONCE
        self.channel_gain = channel_gain  # Channel gain for transmission
        self.local_computing_power = (
            local_computing_power * 1e9
        )  # Local computing power in cycles/sec
        self.bandwidth = bandwidth * 1e6  # Bandwidth in Hz
        self.edge_id = edge_id  # Associated edge server ID
        self.assigned_tasks = []  # List of tasks assigned to this device in current step

    def assign_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth  # Bandwidth in Hz

    def calculate_computing_delay(self, required_cycles):
        """
        Calculate the local latency based on the input size and local computing power.
        """
        return required_cycles / self.local_computing_power  # Convert to seconds
    
    def reset_tasks(self):
        """
        Reset the assigned tasks list (called each step or on environment reset).
        """
        self.assigned_tasks = []
