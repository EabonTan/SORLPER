import random
import numpy as np
from PER.SumTree import SumTree


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # small epsilon to avoid zero priority
    a = 0.6   # prioritization exponent, used for calculating the priority
    beta = 0.4  # importance-sampling weight, you can keep it fixed at 0.4 or 1.0
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)  # Initialize SumTree for experience replay
        self.capacity = capacity

    def _get_priority(self, error):
        # Use TD error for prioritization (simple version)
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        # Add experience with its associated priority
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / n  # Divide the total priority by batch size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # Sample from each segment of the SumTree
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Normalize priorities for IS weight calculation
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities + 1e-6, -self.beta)  # Importance-sampling weight
        is_weight /= is_weight.max()  # Normalize the IS weights

        return batch, idxs, is_weight

    def update(self, idx, error):
        # Update priority after learning
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def memory_length(self):
        # Return the number of entries in the memory
        return self.tree.n_entries
