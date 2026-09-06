import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


class Loss:

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        n = y_true.shape[0]
        loss_val = np.sum((y_pred - y_true) ** 2) / n
        return loss_val

    @staticmethod
    def compute_gradient(x: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: float):
        """
        :param x: (N, feature_dim)
        :param y_true: (N, )
        :param w: (feature_dim, )
        :param b: scalar
        :return: grad_w, grad_b
        """
        n = x.shape[0]
        y_pred = x @ w + b
        diff = y_pred - y_true

        grad_w = (2 / n) * x.T @ diff
        grad_b = (2 / n) * np.sum(diff)
        return grad_w, grad_b


class GradientDescent:
    """optimizer class"""
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def update(self, w: np.ndarray, b: float, grad_w: np.ndarray, grad_b: float):
        w_new = w - self.lr * grad_w
        b_new = b - self.lr * grad_b
        return w_new, b_new


def main():
    concrete_compressive_strength = fetch_ucirepo(id=165)

    x_raw = concrete_compressive_strength.data.features.to_numpy()
    y_raw = concrete_compressive_strength.data.targets.to_numpy().ravel()

    x_train, x_test, y_train, y_test = train_test_split(
        x_raw, y_raw, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    n_features = x_train.shape[1]

    w = np.zeros(n_features)
    b = 0.0
    lr = 0.03
    epoch_num = 800

    optimizer = GradientDescent(learning_rate=lr)

    for epoch in range(epoch_num):
        grad_w, grad_b = Loss.compute_gradient(x_train, y_train, w, b)
        w, b = optimizer.update(w, b, grad_w, grad_b)

        if epoch % 50 == 0:
            y_train_pred = x_train @ w + b
            train_loss = Loss.mse(y_train_pred, y_train)
            print(f"Epoch {epoch:3d} | Train MSE Loss: {train_loss:.4f}")

    y_test_pred = x_test @ w + b
    test_mse = Loss.mse(y_test_pred, y_test)
    print("\n===== Training Finished =====")
    print(f"Test Set MSE Loss：{test_mse:.4f}")
    print(f"Final weights w:\n{np.round(w, 3)}")
    print(f"Final bias b: {np.round(b, 3)}")


if __name__ == "__main__":
    main()
