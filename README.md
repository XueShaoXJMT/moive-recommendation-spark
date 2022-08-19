# moive-recommendation-spark
Spark-based moive recommendation model using Alternative least square algorithm.

# Collaborative filtering
I used collaborative filtering to build the recommendation model, collaborative filtering is based on the wisdom of the crowd approach, if an item is not ranked by a user, it uses the other related users’ rank on it to generate the probability of the target user’s interest.
To know the probability of how the user will like the item, we used matrix factorization, that is firstly build a user-item matrix, each value in the matrix is a user’s rank towards an item. then, decomposed it into 2 matrices with lower dimension, a user matrix and an item matrix. 

# ALS
ALS works by iteratively solving a series of least squares regression problems. In each iteration, one of the user or item matrices is treated as fixed, while the other one is updated using its factor. Then, the other matrix will be updated in the same way. This process continues until the model has converged (or for a fixed number of iterations).

For this project, I used ALS(Alternating Least Square) algorithm to do matrix factorization as it is an easy way for parallel solution on large scale collaborative filtering problems, it also doing a good job at solving scalability and sparseness of large Ratings dataset.
