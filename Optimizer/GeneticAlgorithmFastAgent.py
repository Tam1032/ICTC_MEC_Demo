import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from MEC_Environment.dual_timescale_wrapper import FastEnvWrapper, MECDualTimeScaleEnv


class GeneticAlgorithmFastAgent:
    """
    Genetic Algorithm-based agent for offloading and early exit decisions (fast timescale).
    This is a placeholder for a real genetic algorithm implementation.
    Replace the predict() method with your actual genetic algorithm logic.
    """
    def __init__(self, dual_env, population_size=10, generations=5, mutation_rate=0.1, num_workers=4):
        self.dual_env = dual_env
        self.base_env = dual_env.base_env
        self.num_devices = self.base_env.num_devices
        self.num_edges = self.base_env.num_edges
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_workers = num_workers
        self.record_accuracy = []
        self.record_delay = []

    def predict(self, obs, deterministic=True):
        # Genetic Algorithm: evolve a population of candidate actions
        # Each individual: [offload_0, ..., offload_N, exit_0, ..., exit_N]
        population = self._initialize_population(obs)
        for _ in range(self.generations):
            fitness = self._evaluate_fitness_parallel(population, obs)
            parents = self._select_parents(population, fitness)
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring, obs)
            population = np.vstack((parents, offspring))
        # Select the best individual from the final population
        fitness = self._evaluate_fitness_parallel(population, obs)
        best_idx = np.argmax(fitness)
        best_action = population[best_idx]
        predicted_accuracy, predicted_delay, expected_reward = self._estimate_delay_accuracy(best_action, obs)
        self.record_accuracy.append(predicted_accuracy)
        self.record_delay.append(predicted_delay)
        return best_action, None

    def _initialize_population(self, obs):
        # Prepare constraints for each device
        edge_cached_models = obs["edge_cached_models"] if "edge_cached_models" in obs else None
        num_blocks = obs["num_blocks"] if "num_blocks" in obs else None
        task_models = []
        if hasattr(self.base_env, "pending_tasks"):
            for task in self.base_env.pending_tasks:
                # Find model index by comparing task_type objects
                model_idx = 0
                for i, task_type in enumerate(self.base_env.task_types):
                    if task.task_type is task_type:
                        model_idx = i
                        break
                task_models.append(model_idx)
        else:
            task_models = [0] * self.num_devices

        population = []
        for _ in range(self.population_size):
            offload_actions = np.zeros(self.num_devices, dtype=int)
            exit_actions = np.zeros(self.num_devices, dtype=int)
            for d in range(self.num_devices):
                # Allowed edges: only those where the model is cached
                allowed_edges = []
                if edge_cached_models is not None and len(task_models) > d:
                    model_idx = task_models[d]
                    allowed_edges = [e for e in range(self.num_edges) if edge_cached_models[e][model_idx]]
                if not allowed_edges:
                    allowed_edges = [0]  # fallback to 0 if none available
                allowed_edges.append(self.num_edges)  # always allow cloud
                offload_actions[d] = np.random.choice(allowed_edges)
                # Exit constraint: must be < num_blocks for this device
                max_exit = 1
                if num_blocks is not None and len(num_blocks) > d:
                    max_exit = num_blocks[d]
                if max_exit < 1:
                    max_exit = 1
                exit_actions[d] = np.random.randint(0, max_exit)
            ind = np.concatenate([offload_actions, exit_actions])
            population.append(ind)
        return np.array(population)

    def _evaluate_fitness_parallel(self, population, obs):
        """Evaluate fitness for multiple individuals in parallel."""
        fitness = np.zeros(len(population))
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._evaluate_fitness, ind, obs): i 
                       for i, ind in enumerate(population)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    fitness[idx] = future.result()
                except Exception as e:
                    fitness[idx] = 0.0
        return fitness

    def _estimate_delay_accuracy(self, action, obs):
        # More precise delay estimation using M/G/K queue, processing, transmission, and accuracy from obs
        offload_actions = action[:self.num_devices]
        exit_actions = action[self.num_devices:]
        cloud_idx = self.num_edges
        edge_cores = self.base_env.edge_servers[0].edge_cores if hasattr(self.base_env.edge_servers[0], 'edge_cores') else 8
        # Transmission delay
        edge_transmission_delays = obs.get('edge_transmission_delays', np.zeros((self.num_devices, self.num_edges+1)))
        # Processing delay
        allocated_edge_computing = obs.get('allocated_edge_computing', np.ones(self.num_edges))
        allocated_cloud_computing = obs.get('allocated_cloud_computing', np.ones(1))
        block_compute_requirement = obs.get('block_compute_requirement', np.zeros((self.num_devices, 5)))
        num_blocks = obs.get('num_blocks', np.ones(self.num_devices, dtype=int))
        # Mask the action based on the availability of the tasks
        valid_masks = [True if compute_req[0]> 0 else False for compute_req in block_compute_requirement ]
        # Accuracy
        block_accuracy = obs.get('block_accuracy', np.zeros((self.num_devices, 5)))
        # Estimate per-device delay and accuracy
        min_accuracy = getattr(self.base_env, 'min_accuracy', 0.0)
        max_accuracy = getattr(self.base_env, 'max_accuracy', 1.0)
        min_latency = getattr(self.base_env, 'min_latency', 0.0)
        max_latency = getattr(self.base_env, 'max_latency', 1.0)
        delays = []
        accuracies = []
        for i in range(self.num_devices):
            if valid_masks[i]:
                offload = offload_actions[i]
                exit_block = exit_actions[i] if i < len(exit_actions) else 0
                # Task size
                block_idx = num_blocks[i] - 1 if i < len(num_blocks) else 0
                compute_req = block_compute_requirement[i, block_idx] if i < block_compute_requirement.shape[0] and block_idx < block_compute_requirement.shape[1] else 1e5
                # Transmission delay
                tx_delay = edge_transmission_delays[i, offload] if offload < edge_transmission_delays.shape[1] else 0
                # Processing delay: M/G/K queue approximation for edge, no queue for cloud
                if offload == cloud_idx:
                    # Cloud processing: no queue, just service time
                    compute_power = allocated_cloud_computing[0] if len(allocated_cloud_computing) > 0 else 1e6
                    service_time = compute_req / compute_power
                    proc_delay = service_time
                else:
                    compute_power = allocated_edge_computing[offload] if offload < len(allocated_edge_computing) else 1e6
                    K = edge_cores
                    service_time = compute_req / compute_power
                    arrival_rate = 0.6  # Approximate from env_args
                    rho = arrival_rate * service_time / K
                    if rho < 1:
                        queue_delay = (rho/(1-rho)) * (service_time/K)
                    else:
                        queue_delay = 10 * service_time  # Penalize overload
                    proc_delay = service_time + queue_delay
                # Total delay
                total_delay = tx_delay + proc_delay
                delays.append(total_delay)
                # Accuracy
                acc = block_accuracy[i, exit_block] if i < block_accuracy.shape[0] and exit_block < block_accuracy.shape[1] else 0.5
                accuracies.append(acc)
        if len(delays) > 0:
            avg_delay = np.mean(delays)
        else:
            avg_delay = 0
        if len(accuracies) > 0:
            avg_accuracy = np.mean(accuracies)
        else:
            avg_accuracy = 0
        # Normalization using environment min/max
        normalized_accuracy = (avg_accuracy - min_accuracy) / (max_accuracy - min_accuracy) if max_accuracy > min_accuracy else 0.0
        normalized_latency = (avg_delay - min_latency) / (max_latency - min_latency) if max_latency > min_latency else 0.0
        # Fitness: reward function from environment weights
        acc_weight = getattr(self.base_env, 'acc_weight', 0.5)
        latency_weight = getattr(self.base_env, 'latency_weight', 1.0)
        expected_reward = acc_weight * normalized_accuracy - latency_weight * normalized_latency
        # Reward = acc_weight * normalized_accuracy - latency_weight * normalized_latency
        return avg_accuracy, avg_delay, expected_reward

    def _evaluate_fitness(self, action, obs):
        avg_accuracy, avg_delay, expected_reward = self._estimate_delay_accuracy(action, obs)
        return expected_reward

    def _select_parents(self, population, fitness):
        # Tournament selection
        idx = np.argsort(fitness)[-self.population_size//2:]
        return population[idx]

    def _crossover(self, parents):
        # Single-point crossover
        offspring = []
        num_offspring = self.population_size - len(parents)
        for _ in range(num_offspring):
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
            point = np.random.randint(1, len(p1))
            child = np.concatenate([p1[:point], p2[point:]])
            offspring.append(child)
        return np.array(offspring)

    def _mutate(self, offspring, obs):
        # Prepare constraints for each device
        edge_cached_models = obs["edge_cached_models"] if "edge_cached_models" in obs else None
        num_blocks = obs["num_blocks"] if "num_blocks" in obs else None
        task_models = []
        if hasattr(self.base_env, "pending_tasks"):
            for task in self.base_env.pending_tasks:
                # Find model index by comparing task_type objects
                model_idx = 0
                for i, task_type in enumerate(self.base_env.task_types):
                    if task.task_type is task_type:
                        model_idx = i
                        break
                task_models.append(model_idx)
        else:
            task_models = [0] * self.num_devices

        for ind in offspring:
            for i in range(self.num_devices):
                # Mutate offload action
                if np.random.rand() < self.mutation_rate:
                    allowed_edges = []
                    if edge_cached_models is not None and len(task_models) > i:
                        model_idx = task_models[i]
                        allowed_edges = [e for e in range(self.num_edges) if edge_cached_models[e][model_idx]]
                    if not allowed_edges:
                        allowed_edges = [0]
                    ind[i] = np.random.choice(allowed_edges)
                # Mutate exit action
                if np.random.rand() < self.mutation_rate:
                    max_exit = 1
                    if num_blocks is not None and len(num_blocks) > i:
                        max_exit = num_blocks[i]
                    if max_exit < 1:
                        max_exit = 1
                    ind[self.num_devices + i] = np.random.randint(0, max_exit)
        return offspring

    def predict_fast(self, obs):
        return self.predict(obs)[0]
