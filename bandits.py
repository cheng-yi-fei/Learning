import random
import math
k = 10
q_true = [random.gauss(0, 1) for _ in range(k)]
episodes = 1000
Q = [0.0] * k
N = [0] * k
rewards = []
eps = 0.1
for _ in range(episodes):
    if random.random() < eps:
        a = random.randint(0, k-1)
    else:
        a = Q.index(max(Q))   
    r = random.gauss(q_true[a], 1)    
    N[a] += 1
    Q[a] += (r - Q[a]) / N[a]
    rewards.append(r)
print("最终价值估计：", Q)
