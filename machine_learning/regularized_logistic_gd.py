import numpy as np
from ucimlrepo import fetch_ucirepo

class RegularizedLogisticGD:
    def __init__(self, lr=0.01, lam=0.1, max_iter=5000, tol=1e-6):
        self.lr = lr          # learning rate
        self.lam = lam        # L2 regularization strength
        self.max_iter = max_iter  # max iteration epochs
        self.tol = tol        # gradient convergence threshold
        self.w = None         # model weights

    def sigmoid(self, z):
        # sigmoid activation function for logistic regression
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Xavier Gaussian initialization same as ridge regression homework
        self.w = np.random.randn(n_features) / np.sqrt(n_features)

        for i in range(self.max_iter):
            z = X @ self.w
            y_pred = self.sigmoid(z)
            error = y_pred - y
            # gradient of L2 regularized logistic regression
            grad = (1.0 / n_samples) * (X.T @ error) + 2.0 * self.lam * self.w

            if np.linalg.norm(grad) < self.tol:
                print(f"Converged at iteration {i}")
                break

            self.w -= self.lr * grad
        return self

    def predict_prob(self, X):
        # return probability of positive class
        z = X @ self.w
        return self.sigmoid(z)

if __name__ == "__main__":
    np.random.seed(0)  # fix random seed
    bank_marketing = fetch_ucirepo(id=222)
    X_raw = bank_marketing.data.features.values
    y_raw = bank_marketing.data.targets.values.ravel()

    print("Dataset name:", bank_marketing.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of original features:", X_raw.shape[1])

    # Select only numeric features: age, balance, duration
    X_raw = X_raw[:, [0, 5, 11]]
    print("Number of selected numeric features:", X_raw.shape[1])

    # Convert to float and fill nan values
    X_raw = X_raw.astype(float)
    nan_mask = np.isnan(X_raw)
    col_mean = np.nanmean(X_raw, axis=0)
    X_raw[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # Standardization
    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    n_samples = X.shape[0]
    X = np.hstack([np.ones((n_samples, 1)), X])  # add bias column

    # Convert target label: yes->1, no->0
    y = np.where(y_raw == 'yes', 1, 0)

    model = RegularizedLogisticGD(lr=0.03, lam=0.1, max_iter=5000)
    model.fit(X, y)

    print("Optimal weight w (bias + features):", model.w)
