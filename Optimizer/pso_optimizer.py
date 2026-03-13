import numpy as np

"""
class PSOOptimizer:
    def __init__(self, environment, num_particles=10, max_iter=30, inertia=0.5, c1=1.0, c2=1.5):
        self.env = environment
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

        self.scheme_choices = [0, 1, 2]  # Discrete options

    def optimize(self):
        # Initialize particles randomly
        particles = np.random.choice(self.scheme_choices, size=self.num_particles)
        velocities = np.zeros(self.num_particles)

        p_best = particles.copy()
        p_best_scores = np.array([self.env.evaluate(s) for s in particles])

        g_best = p_best[np.argmin(p_best_scores)]
        g_best_score = np.min(p_best_scores)

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                # Update velocity
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.c1 * r1 * (p_best[i] - particles[i])
                    + self.c2 * r2 * (g_best - particles[i])
                )

                # Update position (discretize by rounding to closest scheme)
                new_position = int(np.clip(round(particles[i] + velocities[i]), 0, 2))
                particles[i] = new_position

                # Evaluate new position
                score = self.env.evaluate(new_position)

                if score < p_best_scores[i]:
                    p_best[i] = new_position
                    p_best_scores[i] = score

                    if score < g_best_score:
                        g_best = new_position
                        g_best_score = score

        return g_best, g_best_score
"""


class PSOOptimizer:
    def __init__(self, env, num_particles=10, max_iter=20, inertia=0.5, c1=1.5, c2=1.5):
        self.env = env
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.num_devices = env.num_devices

    def _evaluate(self, actions):
        # Copy the environment temporarily
        temp_env = self._clone_env()
        rewards, _ = temp_env.step(actions)
        return sum(rewards)  # Total negative latency

    def _clone_env(self):
        import copy

        return copy.deepcopy(self.env)

    def optimize(self):
        # Initialize particles
        particles = np.random.randint(0, 3, size=(self.num_particles, self.num_devices))
        velocities = np.zeros_like(particles, dtype=float)

        p_best = particles.copy()
        p_best_scores = np.array([self._evaluate(p) for p in particles])

        g_best = p_best[np.argmax(p_best_scores)]
        g_best_score = np.max(p_best_scores)

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                for j in range(self.num_devices):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i][j] = (
                        self.inertia * velocities[i][j]
                        + self.c1 * r1 * (p_best[i][j] - particles[i][j])
                        + self.c2 * r2 * (g_best[j] - particles[i][j])
                    )
                    # Discrete update: round and clip to [0, 2]
                    particles[i][j] = int(
                        np.clip(round(particles[i][j] + velocities[i][j]), 0, 2)
                    )

                score = self._evaluate(particles[i])
                if score > p_best_scores[i]:
                    p_best[i] = particles[i].copy()
                    p_best_scores[i] = score

                    if score > g_best_score:
                        g_best = particles[i].copy()
                        g_best_score = score

        return g_best.tolist(), -g_best_score  # return actions and total latency
