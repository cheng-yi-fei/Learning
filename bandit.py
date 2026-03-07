import numpy as np
k = 10
q_true = np.random.randn(k)
episodes = 1000
Q = np.zeros(k)
N = np.zeros(k)
rewards = []
eps = 0.1
for _ in range(episodes):
    if np.random.rand() < eps:
        a = np.random.randint(k)
    else:
        a = np.argmax(Q)
    r = np.random.normal(q_true[a], 1)
    N[a] += 1
    Q[a] += (r - Q[a]) / N[a]
    rewards.append(r)
print("最终价值估计：", Q)
