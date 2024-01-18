import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.spatial


def g(x):
    return -np.sin(6 * np.pi * x)**2 + 6 * x**2 - 5 * x**4 + 3 / 2


def periodic_kernel(x, x_, v, p, l):
    dists = scipy.spatial.distance.cdist(x, x_, metric="sqeuclidean")
    s = np.sin(np.pi * dists / p) / l
    return v * np.exp(-2 * s**2)


def linear_kernel(x, x_, a, b):
    return b + a * np.outer(x, x_)


l = 101
w = np.ones(l) / (l - 1)
w[0] = 1 / (2 * l - 2)
w[-1] = 1 / (2 * l - 2)

S = (np.arange(l) / (l - 1)).reshape(-1, 1)
kernel = lambda x, x_: linear_kernel(x, x_, 1, 1) + periodic_kernel(x, x_, 1.5, 0.167, 4)
K = kernel(S, S)

A = np.vstack((w, np.eye(l)))
Sigma = A @ K @ A.T

q = int(sys.argv[1])
mu_cond = Sigma[1:, 0] / Sigma[0, 0] * q
# Sigma_cond = K - Sigma[1:, 0].reshape(-1, 1) @ Sigma[0, 1:].reshape(1, -1) / Sigma[0, 0]
Sigma_cond = K - K @ w.reshape(-1, 1) @ w.reshape(1, -1) @ K / (w.reshape(1, -1) @ K @ w.reshape(-1, 1))

for _ in range(5):
    samples = np.random.multivariate_normal(mu_cond, Sigma_cond)
    print("Approximated integral:", w @ samples)
    plt.plot(S, samples)

plt.title(f"q = {q}")
plt.xlabel("x")
plt.ylabel("f")
plt.savefig(f"B2_q_{q}.png", dpi=300)


xs = np.array([0, 0.25, 0.5])
ys = np.array([1.46, 0.93, 2.76])

B = np.vstack((
    np.hstack((np.zeros(3), w)),
    np.hstack((np.eye(3), np.zeros((3, l)))),
    np.hstack((np.zeros((l, 3)), np.eye(l)))
))
s = np.vstack((xs.reshape(-1, 1), S))
K = kernel(s, s)
Sigma = B @ K @ B.T + 0.1 * np.diag(np.hstack((0, 1, 1, 1, np.zeros(l))))

mu_cond = Sigma[4:, :4] @ np.linalg.inv(Sigma[:4, :4]) @ np.hstack((q, ys)).reshape(-1, 1)
Sigma_cond = Sigma[4:, 4:] - Sigma[4:, :4] @ np.linalg.inv(Sigma[:4, :4]) @ Sigma[:4, 4:]

print(np.diag(Sigma_cond))

plt.figure()
plt.plot(S, mu_cond)
# plt.plot(S, g(S), c="gray")
plt.fill_between(
    S.flatten(),
    mu_cond.flatten() - 1.96 * np.sqrt(np.abs(np.diag(Sigma_cond))),
    mu_cond.flatten() + 1.96 * np.sqrt(np.abs(np.diag(Sigma_cond))),
    color="C0",
    alpha=0.3,
)
plt.scatter(xs, ys)
plt.savefig('fitted.png', dpi=600)
