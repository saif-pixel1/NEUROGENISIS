import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []   # (X, y, importance)
        self.n = 0

    def add(self, X, y, importance):
        for i in range(len(X)):
            if self.n < self.capacity:
                self.buffer.append((X[i], y[i], importance[i]))
            else:
                j = random.randint(0, self.n)
                if j < self.capacity:
                    self.buffer[j] = (X[i], y[i], importance[i])
            self.n += 1

    def sample(self, k):
        if not self.buffer:
            return None, None
        probs = np.array([imp for _, _, imp in self.buffer])
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), size=min(k, len(self.buffer)), replace=False, p=probs)
        X_s = np.array([self.buffer[i][0] for i in indices])
        y_s = np.array([self.buffer[i][1] for i in indices])
        return X_s, y_s