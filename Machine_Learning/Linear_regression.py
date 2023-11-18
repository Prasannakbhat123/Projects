import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    return data


def preprocess_data(data):
    X = data[['AGE', 'FEMALE', 'LOS', 'RACE', 'APRDRG']].values
    Y = data['TOTCHG'].values

    # Standardize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Add an intercept term
    X = np.column_stack((np.ones(len(X)), X))

    return X, Y


def linear_regression(X, Y, learning_rate, iterations):
    m, n = X.shape

    coefficients = np.ones(n)

    print("m=",coefficients)

    cost_function = []

    for _ in range(iterations):
        predictions = (X @ coefficients)
        error = predictions - Y
        gradient = (1 / m) * (X.T @ error)
        coefficients -= learning_rate * gradient
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_function.append(cost)

    return coefficients, cost_function



data = load_data('linear_regression_dataset.csv')
X, Y = preprocess_data(data)

learning_rate = 0.001
iterations = 10000

coefficients, cost_function = linear_regression(X, Y, learning_rate, iterations)

print("Coefficients:", coefficients)
print("Final Cost:", cost_function[-1])

sample_data = X[4]
actual_pred = Y[4]
print("Predictions:", np.dot(sample_data, coefficients), "Actual Value:", actual_pred)
predictions=X@coefficients
print("Accuaracy:",r2_score(Y, predictions))

# aprdrg = np.array(data['APRDRG'])
# totchg = data['TOTCHG']
# plt.scatter(aprdrg, totchg, c='b', marker='o', label='Data Points')
# plt.plot(aprdrg, X @ coefficients, c='r', label='Linear Regression Line')
# plt.title('APRDRG vs TOTCHG')
# plt.xlabel('APRDRG')
# plt.ylabel('TOTCHG')
# plt.legend()
# plt.grid(True)

# Plot the cost function curve
plt.figure(figsize=(8, 6))
plt.plot(range(iterations), cost_function, c='g', label='Cost Function')
plt.title('Cost Function over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

plt.show()



