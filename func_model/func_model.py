import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error



np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.exp(X) + np.random.normal(scale=0.5, size=X.shape)

X = X.reshape(-1, 1)


poly_features = PolynomialFeatures(degree=5)
poly_model = make_pipeline(poly_features, LinearRegression())
poly_model.fit(X, y)
poly_predictions = poly_model.predict(X)

# Вычисление MSE
poly_mse = mean_squared_error(y, poly_predictions)

print(f'MSE: {poly_mse}')


plt.scatter(X, y, color='blue', label='Data', s=1)
plt.plot(X, poly_predictions, color='red', label='Polynomial Regression')
plt.legend()
plt.show()
