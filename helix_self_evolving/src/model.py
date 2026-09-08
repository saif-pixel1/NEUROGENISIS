from sklearn.ensemble import RandomForestClassifier
import numpy as np

class StaticModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class NaiveStreamingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_all = []
        self.y_all = []
    def partial_fit(self, X, y):
        self.X_all.append(X)
        self.y_all.append(y)
        X_comb = np.vstack(self.X_all)
        y_comb = np.hstack(self.y_all)
        self.model.fit(X_comb, y_comb)
    def predict(self, X):
        return self.model.predict(X)

class HELIXModel:
    def __init__(self, buffer_capacity=5000):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.detector = HybridDriftDetector()
        self.adapt_count = 0
        self.validation_window = []   # store (X, y) for recent batches

    def process_batch(self, X_batch, y_batch):
        # Predict
        y_pred = self.model.predict(X_batch)
        error = np.mean(y_pred != y_batch)
        # Detect drift
        drift = self.detector.update(error, X_batch)
        if drift:
            self._adapt(X_batch, y_batch)
            self.adapt_count += 1
            self.detector.reset()
        # Update buffer with importance (1 - max prob)
        proba = self.model.predict_proba(X_batch)
        importance = 1 - np.max(proba, axis=1)
        self.buffer.add(X_batch, y_batch, importance)
        # Store for validation
        self.validation_window.append((X_batch, y_batch))
        if len(self.validation_window) > 5:
            self.validation_window.pop(0)
        return y_pred

    def _adapt(self, X_new, y_new):
        # Simple strategy selection: if recent accuracy < 0.7, retrain with replay
        if len(self.validation_window) >= 3:
            recent_acc = np.mean([np.mean(self.model.predict(Xv) == yv) for Xv, yv in self.validation_window[-3:]])
            if recent_acc < 0.7:
                strategy = 'retrain'
            else:
                strategy = 'partial'
        else:
            strategy = 'retrain'
        # Apply
        X_replay, y_replay = self.buffer.sample(1000)
        if X_replay is not None:
            X_comb = np.vstack([X_replay, X_new])
            y_comb = np.hstack([y_replay, y_new])
        else:
            X_comb = X_new
            y_comb = y_new
        self.model.fit(X_comb, y_comb)