import numpy as np

class SwarmOptimizationFastAgent:
    """
    Swarm Optimization-based agent for offloading and early exit decisions (fast timescale).
    This is a placeholder for a real swarm optimization algorithm (e.g., PSO, ACO, etc.).
    Replace the predict() method with your actual swarm optimization logic.
    """
    def __init__(self, dual_env, num_particles=10, max_iters=5):
        self.dual_env = dual_env
        self.base_env = dual_env.base_env
        self.num_devices = self.base_env.num_devices
        self.num_edges = self.base_env.num_edges
        self.num_particles = num_particles
        self.max_iters = max_iters

    def predict(self, obs, deterministic=True):
        """
        Particle Swarm Optimization (PSO) for fast-timescale decision-making.
        Each particle represents a candidate action vector: [offload_0, ..., offload_N, exit_0, ..., exit_N].
        Only offload actions are optimized; exit actions remain zero.
        """
        particles = self._initialize_particles(obs)
        velocities = np.zeros_like(particles, dtype=float)
        personal_best = particles.copy()
        personal_best_scores = np.array([self._evaluate_fitness(p, obs) for p in particles])
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()

        for _ in range(self.max_iters):
            for i, particle in enumerate(particles):
                # PSO update
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    0.5 * velocities[i]
                    + 1.5 * r1 * (personal_best[i] - particle)
                    + 1.5 * r2 * (global_best - particle)
                )
                # Discrete update: round and clip offload actions
                particle_new = np.round(particle + velocities[i]).astype(int)
                for d in range(self.num_devices):
                    if isinstance(obs, dict) and "offload_candidates" in obs:
                        candidates = obs["offload_candidates"][d]
                        if len(candidates) > 0:
                            if particle_new[d] not in candidates:
                                particle_new[d] = np.random.choice(candidates)
                        else:
                            particle_new[d] = 0
                    else:
                        particle_new[d] = np.clip(particle_new[d], 0, self.num_edges - 1)
                # Exit actions remain zero
                particle_new[self.num_devices:] = 0
                particles[i] = particle_new
                score = self._evaluate_fitness(particle_new, obs)
                if score > personal_best_scores[i]:
                    personal_best[i] = particle_new.copy()
                    personal_best_scores[i] = score
            global_best_idx = np.argmax(personal_best_scores)
            global_best = personal_best[global_best_idx].copy()
        return global_best, None

    def _initialize_particles(self, obs):
        particles = []
        for _ in range(self.num_particles):
            if isinstance(obs, dict) and "offload_candidates" in obs:
                candidates = obs["offload_candidates"]
                offload_actions = np.array([
                    np.random.choice(candidates[d]) if len(candidates[d]) > 0 else 0
                    for d in range(self.num_devices)
                ])
            else:
                offload_actions = np.random.randint(0, self.num_edges, size=self.num_devices)
            exit_actions = np.zeros(self.num_devices, dtype=int)
            ind = np.concatenate([offload_actions, exit_actions])
            particles.append(ind)
        return np.array(particles)

    def _evaluate_fitness(self, action, obs):
        fast_env = self.dual_env.fast_env if hasattr(self.dual_env, 'fast_env') else None
        if fast_env is None:
            return np.random.rand()
        state = fast_env.save_state() if hasattr(fast_env, 'save_state') else None
        try:
            _, reward, _, _, info = fast_env.step(action)
        except Exception:
            reward = 0.0
        if state is not None and hasattr(fast_env, 'load_state'):
            fast_env.load_state(state)
        return reward

    def predict_fast(self, obs):
        return self.predict(obs)[0]
