import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) 
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Linear regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
