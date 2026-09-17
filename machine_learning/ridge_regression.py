import numpy as np
from ucimlrepo import fetch_ucirepo

class RidgeGD:
    def __init__(self, lr=0.001, lam=1.0, max_iter=5000, tol=1e-6):
        self.lr = lr          # learning rate
        self.lam = lam        # regularization strength
        self.max_iter = max_iter  # max iteration epochs
        self.tol = tol        # gradient convergence threshold
        self.w = None         # model weights

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Xavier Gaussian initialization
        self.w = np.random.randn(n_features) / np.sqrt(n_features)

        for i in range(self.max_iter):
            y_pred = X @ self.w
            error = y_pred - y
            # compute gradient for ridge regression
            grad = (2.0 / n_samples) * (X.T @ error) + 2.0 * self.lam * self.w

            if np.linalg.norm(grad) < self.tol:
                print(f"Converged at iteration {i}")
                break

            self.w -= self.lr * grad

        return self

    def predict(self, X):
        return X @ self.w      # predict output

if __name__ == "__main__":
    np.random.seed(0)  # fix random seed
    real_estate_valuation = fetch_ucirepo(id=477)
    X_raw = real_estate_valuation.data.features.values
    y_raw = real_estate_valuation.data.targets.values.ravel()
    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of original features:", X_raw.shape[1])
    X_raw = X_raw[:, [1, 2]]
    print("Number of selected features:", X_raw.shape[1])
    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)  # standardization
    n_samples = X.shape[0]
    X = np.hstack([np.ones((n_samples, 1)), X])  # add bias column

    model = RidgeGD(lr=0.03, lam=0.1, max_iter=5000)
    model.fit(X, y_raw)

    print("Optimal weight w (bias + features):", model.w)
