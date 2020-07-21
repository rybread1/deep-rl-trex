import numpy as np


class SumTree(object):
    """restore adjust priority in leaves and sum in nodes"""
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.counter = 0
        self.tree = np.zeros(2 * capacity - 1)  # for all nodes(n - 1) and all leaves(n)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1    # first leaf index
        self.data[self.data_pointer] = data  # update transition
        self.update(tree_idx, p)  # update tree

        if self.counter < self.capacity:
            self.counter += 1
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, adj_pri):
        change = adj_pri - self.tree[tree_idx]    # change between previous adj_pri and current adj_pri
        self.tree[tree_idx] = adj_pri
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # index of relative node of this leaf
            self.tree[tree_idx] += change   # add change to the sum node

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing adj_priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing adj_priority for transitions
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # total adj_pri
