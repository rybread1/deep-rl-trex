import numpy as np
from sum_tree import SumTree
from collections import deque
import random


class BasicMemory:
    def __init__(self, memory_size):
        self.experience_replay = deque(maxlen=memory_size)

    def get_batch(self, batch_size):
        return random.sample(self.experience_replay, batch_size)

    def add(self, row):
        self.experience_replay.append(row)


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = SumTree(memory_size)
        self.epsilon = 0.0001  # small amount to avoid zero priority
        self.alpha = 0.6  # adj_pri = pri^alpha
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_max = 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped td error

    def add(self, row):
        max_p = np.max(self.memory.tree[-self.memory.capacity:])    # max adj_pri of leaves
        if max_p == 0:
            max_p = self.abs_err_upper
        self.memory.add(max_p, row)  # set the max adj_pri for new adj_pri

    def get_batch(self, batch_size):
        leaf_idx, batch_memory, ISWeights = np.empty(batch_size, dtype=np.int32), np.empty(batch_size,dtype=object), np.empty(batch_size)
        pri_seg = self.memory.total_p / batch_size  # adj_pri segment
        self.beta = np.min([self.beta_max, self.beta + self.beta_increment_per_sampling])  # max = 1

        # Pi = Prob(i) = softmax(priority(i)) = adj_pri(i) / ∑_i(adj_pri(i))
        # ISWeight = (N*Pj)^(-beta) / max_i[(N*Pi)^(-beta)] = (Pj / min_i[Pi])^(-beta)
        min_prob = np.min(self.memory.tree[self.memory.capacity-1:self.memory.capacity-1+self.memory.counter]) / self.memory.total_p
        for i in range(batch_size):
            # sample from each interval
            a, b = pri_seg * i, pri_seg * (i + 1)   # interval
            v = np.random.uniform(a, b)
            idx, p, data = self.memory.get_leaf(v)
            prob = p / self.memory.total_p
            ISWeights[i] = np.power(prob / min_prob, -self.beta)
            leaf_idx[i], batch_memory[i] = idx, data
        return leaf_idx, batch_memory, ISWeights

    def update_sum_tree(self, tree_idx, td_errors):
        for ti, td_error in zip(tree_idx, td_errors):
            p = self._calculate_priority(td_error)
            self.memory.update(ti, p)

    def _calculate_priority(self, td_error):
        priority = abs(td_error) + self.epsilon
        clipped_pri = np.minimum(priority, self.abs_err_upper)
        return np.power(clipped_pri, self.alpha)

    @property
    def length(self):
        return self.memory.counter

    def load_memory(self, memory):
        self.memory = memory

    def get_memory(self):
        return self.memory
