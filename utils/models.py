from sklearn.multioutput import RegressorChain
from sklearn.base import clone
import numpy as np

class LastOnlyRegressorChain(RegressorChain):
    def fit(self, X, Y):
        self.chain_ = range(Y.shape[1]) if self.order is None else self.order
        self.estimators_ = []
        X_extended = X

        for chain_idx, target_idx in enumerate(self.chain_):
            y = Y[:, target_idx]
            estimator = clone(self.base_estimator)
            estimator.fit(X_extended, y)
            self.estimators_.append(estimator)

            y_pred = estimator.predict(X_extended).reshape(-1, 1)
            # Only use the last prediction, not all
            X_extended = np.hstack([X, y_pred])

        return self

    def predict(self, X):
        Y_pred_chain = np.zeros((X.shape[0], len(self.chain_)))
        X_extended = X

        for chain_idx, estimator in enumerate(self.estimators_):
            y_pred = estimator.predict(X_extended).reshape(-1, 1)
            target_idx = self.chain_[chain_idx]
            Y_pred_chain[:, target_idx] = y_pred.ravel()

            # Only use the last prediction
            X_extended = np.hstack([X, y_pred])

        return Y_pred_chain