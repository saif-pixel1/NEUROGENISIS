from skmultiflow.drift_detection import ADWIN
from scipy.stats import ks_2samp
import numpy as np

class HybridDriftDetector:
    def __init__(self, delta=0.01, window_size=100):
        self.adwin = ADWIN(delta=delta)
        self.feature_window = []  
        self.window_size = window_size
        self.drift_flag = False

    def update(self, error, X_batch):

        self.adwin.add_element(error)
        self.drift_flag = self.adwin.detected_change()
        # KS test on first 10 PCs
        if X_batch is not None and len(self.feature_window) > 0:
            current_mean = np.mean(X_batch[:, :10], axis=0)
            ref_mean = np.mean(self.feature_window, axis=0)
            _, p_val = ks_2samp(current_mean, ref_mean)
            if p_val < 0.01:
                self.drift_flag = True

        if X_batch is not None:
            self.feature_window.append(np.mean(X_batch[:, :10], axis=0))
            if len(self.feature_window) > self.window_size:
                self.feature_window.pop(0)
        return self.drift_flag

    def reset(self):
        self.adwin = ADWIN(delta=self.adwin.delta)
        self.drift_flag = False