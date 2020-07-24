from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load data
iris = datasets.load_iris()
X=iris.data[0:99,:2]
y=iris.target[0:99]
# Plot the training points
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


# Function for Stochastic Gradient Descent
def Stochastic_GD(Learning_Rate, num_iterations, X, y):
    # Step 1: Initial Parameter
    N = len(X)
    w = np.zeros((X.shape[1], 1))
    b = 0
    costs = []
    # Starting two layer of loops
    for i in range(num_iterations):
        for j in range(N):
            # Choose 1 record
            XX = X[j, :]
            yy = y[j]

            # Step 2: Apply Sigmoid Function and get y prediction
            Z = np.dot(w.T, XX.T) + b
            y_pred = 1 / (1 + 1 / np.exp(Z))

            # Step 3: Calculate Loss Function
            cost = -(yy * np.log(y_pred) + (1 - yy) * np.log(1 - y_pred))

            # Step 4: Calculate Gradient
            dw = np.multiply(XX, (y_pred - yy)).reshape((2, 1))
            db = y_pred - yy

            # Step 5: Update w & b
            w = w - Learning_Rate * dw
            b = b - Learning_Rate * db

        # Step 6: Calculate Loss Function
        Z_full = np.dot(w.T, X.T) + b
        y_pred_full = 1 / (1 + 1 / np.exp(Z_full))
        cost = -(1 / N) * np.sum(y * np.log(y_pred_full) + (1 - y) * np.log(1 - y_pred_full))

        # Records cost
        if i % 100 == 0:
            costs.append(cost)
            print(cost)

    return (w, b, costs)


# Run a function
Result_Stoc_GD = Stochastic_GD(Learning_Rate=0.01, num_iterations=2000, X=X, y=y)