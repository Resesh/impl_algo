# implement k-means algo


import numpy as np
import matplotlib.pyplot as plt
comp = np.random.rand(200,2)*(10-(-10))-10
comp_num, comp_dim = comp.shape

def first_cnt(k):
    first_cnt = np.ndarray((comp.shape))
    np.random.shuffle(first_cnt)
    return first_cnt[:k]

def kmeans(comp, k=3, iter=100):
    """
    k: number of cluster
    comp: two dimensinal points
    iter: number of iteration execution
    """
    cluster = np.ndarray(comp_num)
    centroid = first_cnt(k=3)
    new_cnt = np.ndarray((k, comp_dim))
    for _ in range(iter):
        # calculate distance and decide cluster
        for i in range(comp_num):
            distance = np.sum((centroid - comp[i])**2, axis = 1)
            cluster[i] = np.argmin(distance, axis = 0)
        # renew cluster
        for j in range(comp_dim):
            new_cnt[j] = comp[cluster==j].mean(axis = 0)
    return cluster, new_cnt
cluster, new_cnt = kmeans(comp)
print(cluster, new_cnt)
