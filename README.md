# Gradient-Descent

Implementation of Gradient Descent in logistic regression to identify 2 kinds of iris from Iris dataset.

The repository contains 2 types of gradient descent algorithms:
1. Batch Gradient Descent: Batch Gradient Descent uses the whole batch of training data at every step. It calculates the error for each record and takes an average to determine the gradient. The advantage of Batch Gradient Descent is that the algorithm is more computational efficient and it produces a stable learning path, so it is easier to convergence. However, Batch Gradient Descent takes longer time when the training set is large.

2. Stochastic Gradient Descent: Stochastic Gradient Descent just picks one instance from training set at every step and update gradient only based on that single record. The advantage of Stochastic Gradient Descent is that the algorithm is much faster at every iteration, which remediate the limitation in Batch Gradient Descent. However, the algorithm produces less regular and stable learning path compared to Batch Gradient Descent. Instead of decreasing smoothly, the cost function will bounce up and down. After rounds of iterations, the algorithm may find a good parameter, but the final result is not necessary global optimal.
