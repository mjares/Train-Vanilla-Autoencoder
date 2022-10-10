import numpy as np
from numpy.random import default_rng
from TrajectoryDataset import TrajectoryDataset


class SupervisedDataset:
    """ A dataset for supervised mode estimation
    - State ii + 1 (next_state ii) is the result of applying control command ii to state ii.
    """
    def __init__(self):
        self.no_samples = 0
        self.state = np.empty(1)
        self.next_state = np.empty(1)
        self.weight = np.empty(1)  # 0 is the loe controller
        self.control = np.empty(1)  # motor commands
        self.mode = np.empty(1)
        self.classes = 'Extreme'

    def from_trajectory_dataset(self, trajectory_dataset):
        """
        Parameters
        ----------
        trajectory_dataset : HildensiaDataset
        """
        self.no_samples = trajectory_dataset.weight.shape[0]
        obs = trajectory_dataset.observations
        self.state = obs['state']
        self.control = obs['motor_command']
        self.weight = trajectory_dataset.weight
        self.mode = [trajectory_dataset.conditions]*(self.no_samples - 1)

        # For trajectory_dataset: state ii is the result of applying control command ii to state ii - 1.
        self.control = np.delete(self.control, 0,  axis=0)
        self.next_state = self.state
        self.next_state = np.delete(self.next_state, 0,  axis=0)
        self.state = np.delete(self.state, -1,  axis=0)

        self.no_samples = self.no_samples - 1
        self.weight = np.delete(self.weight, 0,  axis=0)

    def to_numpy_array(self):
        """
        Parameters
        ----------
        self.classes: str
            Extreme: Extreme LOE and att noise conditions (2 classes)
            NLA: Nominal conditions, att. noise and LOE (3 classes)
            NL: Nominal conditions and LOE (2 classes)
            Single: Single class (1 class)
            UnsupSweep: Nominal and LOE in one class, and att. noise in another (2 classes) (sweep threshold for border)
        """
        features = np.hstack([self.state, self.next_state, self.control])
        fault_mag = np.zeros(self.no_samples)
        if self.classes == 'Extreme':
            labels = np.zeros(self.no_samples)
            for ii in range(self.no_samples):
                if self.mode[ii]['Fault'] == 'Rotor':
                    labels[ii] = 0
                elif self.mode[ii]['Fault'] == 'AttNoise':
                    labels[ii] = 1
                fault_mag[ii] = self.mode[ii]['Mag']

        if self.classes == 'NLA':
            labels = np.zeros([self.no_samples, 3])
            for ii in range(self.no_samples):
                fault_mag[ii] = self.mode[ii]['Mag']
                if self.mode[ii]['Fault'] == 'Rotor' and fault_mag[ii] >= 0.1:
                    labels[ii, 0] = 1
                elif self.mode[ii]['Fault'] == 'AttNoise' and fault_mag[ii] >= 0.84:
                    labels[ii, 1] = 1
                else:
                    labels[ii, 2] = 1

        if self.classes == 'NL':
            labels = np.zeros([self.no_samples, 2])
            for ii in range(self.no_samples):
                fault_mag[ii] = self.mode[ii]['Mag']
                if self.mode[ii]['Fault'] == 'Rotor' and fault_mag[ii] >= 0.1:
                    labels[ii, 0] = 1
                else:
                    labels[ii, 1] = 1

        if self.classes == 'Single':
            labels = np.zeros([self.no_samples, 1])
            for ii in range(self.no_samples):
                fault_mag[ii] = self.mode[ii]['Mag']
                labels[ii] = 0

        if self.classes == 'UnsupSweep':
            labels = np.zeros(self.no_samples)
            for ii in range(self.no_samples):
                fault_mag[ii] = self.mode[ii]['Mag']
                if self.mode[ii]['Fault'] == 'AttNoise':
                    if fault_mag[ii] >= 0.84:
                        labels[ii] = 1
                    else:
                        labels[ii] = 2  # This third class represents non-extreme att noise conditions.
                else:
                    labels[ii] = 0
                fault_mag[ii] = self.mode[ii]['Mag']

        return features, labels, fault_mag


class HyperSupervisedDataset:
    def __init__(self):
        self.features = np.array([])
        self.labels = np.array([])
        self.fault_mag = np.array([])
        self.no_samples = 0
        self.shuffled = False
        # Episode timestamps
        self.ep_timestamps = None
        self.ep_timestamps_updated = False
        self.no_episodes = 0

    def append_dataset(self, sup_data):
        features, labels, mags = sup_data.to_numpy_array()
        if self.no_samples == 0:
            self.features = features
            self.labels = labels
            self.fault_mag = mags
            self.no_samples = sup_data.no_samples
        else:
            self.features = np.vstack([self.features, features])
            self.labels = np.concatenate([self.labels, labels])  # Dimensions dont match
            self.fault_mag = np.hstack([self.fault_mag, mags])
            self.no_samples = self.no_samples + sup_data.no_samples

    def shuffle(self):
        rng = default_rng()
        idx = rng.permutation(self.no_samples)
        self.features = self.features[idx, :]
        self.labels = self.labels[idx]
        self.fault_mag = self.fault_mag[idx]
        self.shuffled = True
        self.ep_timestamps_updated = False

    def normalize(self, return_scale=False):
        """ Standardize each feature column to zero mean and unit std
        """
        mean = self.features.mean(axis=0)
        std = self.features.std(axis=0)
        normed = (self.features - mean) / std
        new_dataset = HyperSupervisedDataset()
        new_dataset.no_samples = self.no_samples
        new_dataset.features = normed
        new_dataset.labels = self.labels
        new_dataset.fault_mag = self.fault_mag

        if return_scale:
            return new_dataset, [mean, std]
        else:
            return new_dataset

    def subset(self, sub_range):
        new_dataset = HyperSupervisedDataset()
        new_dataset.no_samples = len(sub_range)
        new_dataset.features = self.features[sub_range, :]
        new_dataset.labels = self.labels[sub_range]
        new_dataset.fault_mag = self.fault_mag[sub_range]

        return new_dataset

    def get_episode_timestamps(self):

        if self.shuffled:
            print('WARNING!: Dataset is shuffled. Episode timestamps do not represent actual episodes.')

        if self.ep_timestamps_updated:
            return self.ep_timestamps, self.no_episodes

        self.ep_timestamps = [0]
        for ii in range(1, self.no_samples):
            if self.fault_mag[ii] != self.fault_mag[ii - 1]:
                self.ep_timestamps.append(ii)
        self.no_episodes = len(self.ep_timestamps)
        self.ep_timestamps.append(self.no_samples)  # Last sample
        self.ep_timestamps = np.array(self.ep_timestamps)
        self.ep_timestamps_updated = True

        return self.ep_timestamps, self.no_episodes
