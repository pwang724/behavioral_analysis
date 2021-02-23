import numpy as np
import cv2

# X = cv2.triangulatePoints(P1, P2, x1, x2)
# 	P1: Camera projection from X to x1; x1 = dot(P1,X)
# 	P2: Camera projection from X to x2; x2 = dot(P2,X)
# 	x1: 2xN normalized points
# 	x2: 2xN normalized points


# Camera projection matrices
P1 = np.eye(4)
P1 = P1[:3]
P2 = np.array([[0.878, -0.01, 0.479, -1.995],
            [0.01, 1., 0.002, -0.226],
            [-0.479, 0.002, 0.878, 0.615]])

# homogeneous coordinates
a3xN = np.array([[0.091, 0.167, 0.231, 0.083, 0.154],
              [0.364, 0.333, 0.308, 0.333, 0.308]])
b3xN = np.array([[0.42, 0.537, 0.645, 0.431, 0.538],
              [0.389, 0.375, 0.362, 0.357, 0.345]])
X = cv2.triangulatePoints(P1, P2, a3xN, b3xN)
X /= X[3]
# Recover the origin arrays from PX
x1 = np.dot(P1, X)
x2 = np.dot(P2, X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]

print(X)
print(x1)
print(x2)