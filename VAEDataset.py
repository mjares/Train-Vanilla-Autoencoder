import numpy as np
from numpy.random import default_rng


class VAEDataset:
    def __init__(self):
        self.features = np.zeros(1)
        self.cost_values = 0
        self.tags = []
        self.no_samples = 1

        self.ranges = np.zeros((2,1))
        self.standard = 'Raw'

    def add_features(self, features, costs, tags):
        self.features = features
        self.cost_values = costs
        self.tags = tags
        self.no_samples = len(tags)

    def standardize_minmax(self, sample):
        standardized = (sample - self.ranges[0, :])/(self.ranges[1, :] - self.ranges[0, :])
        return standardized

    def destandardize_minmax(self, sample):
        destandardized = sample * (self.ranges[1, :] - self.ranges[0, :]) + self.ranges[0, :]
        return destandardized

    def shuffle(self):
        rng = default_rng()
        idx = rng.permutation(self.no_samples)
        self.features = self.features[idx, :]
        self.cost_values = self.cost_values[idx]
        self.tags = np.array(self.tags)[idx]

    def add_sample(self, feature, cost, tag):
        self.features = np.vstack((self.features, feature))
        self.cost_values = np.hstack((self.cost_values, cost))
        self.tags.append(tag)
        self.no_samples = self.no_samples + 1
