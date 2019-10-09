import numpy as np
from federated_aggregator.models.diff import Diffs, PartialDiffs


class MetricsHandler(object):

    def __init__(self, size):
        self.noise = np.random.randint(0, 10, (1, size))[0]

    def get_diffs(self, encr_diffs, encr_partial_diffs):
        diffs = Diffs(encr_diffs)
        diffs.add_noise(self.noise)
        partial_diffs = PartialDiffs(encr_partial_diffs)
        partial_diffs.add_noise(self.noise)
        return diffs.get_arrays(), partial_diffs.get_dict()

    def get_mses(self, diffs, partial_diffs):
        diffs = Diffs(diffs)
        diffs.substract_noise(self.noise)
        partial_diffs = PartialDiffs(partial_diffs)
        partial_diffs.substract_noise(self.noise)
        mse, mses_per_valid = diffs.get_mses()
        return mse, mses_per_valid, partial_diffs.get_partial_mses()

    def get_noise(self):
        return self.noise.tolist()
