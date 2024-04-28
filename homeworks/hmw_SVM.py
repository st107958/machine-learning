import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Инициализация параметров
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Градиентный спуск для оптимизации параметров
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = np.where(linear_model >= 0, 1, -1)

            # Градиенты
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + 2 * self.lambda_param * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Обновление параметров
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.where(linear_model >= 0, 1, -1)


# Пример использования
X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array([-1, -1, 1, 1, -1, 1])

svm = SVM()
svm.fit(X_train, y_train)

X_test = np.array([[1, 3], [8, 9], [0, 1], [3, 4]])
print(svm.predict(X_test))
