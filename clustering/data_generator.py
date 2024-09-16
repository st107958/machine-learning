from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


X1, y1 = make_circles(n_samples=1000, factor=0.5, noise=0.2)

X2 = X1 *1.5+3
y2 = np.ones_like(y1) * 2
X = np.vstack([X1, X2])
y = np.concatenate([y1, y2])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.show()

np.save('X_circles.npy', X1)
np.save('y_circles.npy', y1)


X2, y2 = make_blobs(n_samples=500, centers=3, cluster_std=4, random_state=42)

noise = np.random.normal(scale=0.5, size=X2.shape)
X_noisy = X2 + noise

# plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y2, cmap='viridis')
# plt.show()

np.save('X_overlapping.npy', X_noisy)
np.save('y_overlapping.npy', y2)


X3, y3 = make_moons(n_samples=500, noise=0.2, random_state=42)

# plt.scatter(X3[:, 0], X3[:, 1], c=y3, cmap='winter')
# plt.show()

np.save('X_moons.npy', X3)
np.save('y_moons.npy', y3)


n_points = 500
n_rows, n_cols = 4, 4

X4 = np.random.rand(n_points, 2) * [n_cols, n_rows]

y4 = ((X4[:, 0] // 1) + (X4[:, 1] // 1)) % 2

# plt.scatter(X4[:, 0], X4[:, 1], c=y4)
# plt.show()

np.save('X_chessboard.npy', X4)
np.save('y_chessboard.npy', y4)

