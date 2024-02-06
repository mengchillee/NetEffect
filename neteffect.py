import numpy as np
from collections import defaultdict

from scipy.sparse import lil_matrix, csr_matrix
from sklearn.utils.extmath import randomized_svd

import rwcpp
from netest import *


def degree_matrix(A, p=1):
    A = csr_matrix(A)
    diag = np.array((A.power(p) if p > 1 else A).sum(axis=0))[0]
    D = lil_matrix(A.shape)
    D.setdiag(diag)
    return csr_matrix(D)


def random_walk_matrix(A, T, trials):
    neighbors = defaultdict(list)
    for ii, jj in zip(*A.nonzero()):
        if ii < jj:
            neighbors[ii].append(jj)
            neighbors[jj].append(ii)
    walks_arr = rwcpp.random_walks(neighbors, T, trials)

    index = ([], [])
    S = lil_matrix(A.shape)
    for k in walks_arr.keys():
        index[0].append(k[0])
        index[1].append(k[1])
    S[index] = list(walks_arr.values())
    return S


def proximity_matrix(A, T, trials=30):
    A = csr_matrix(A, copy=True)
    S = random_walk_matrix(A, T, trials)
    D = degree_matrix(A).power(-1)
    S = D @ (csr_matrix(S).multiply(A))
    nz = S.nonzero()
    S[nz] = np.log(S[nz])
    return csr_matrix(S)


def euclidean_matrix(A, X):
    S, index, dis_arr = lil_matrix(A.shape), ([], []), []
    for idx1, idx2 in zip(*A.nonzero()):
        if idx1 < idx2:
            dis = -np.linalg.norm(X[idx1] - X[idx2])
            index[0].extend([idx1, idx2])
            index[1].extend([idx2, idx1])
            dis_arr.append(dis)
    S[index] = np.repeat(np.exp(dis_arr), 2)
    return csr_matrix(S)


def emphasis_matrix(A, T=4, trials=10, rank=128, random_state=None):
    W = proximity_matrix(A, T=T, trials=trials)
    X, s, _ = randomized_svd(W, rank, n_iter=2, random_state=random_state)
    for r, ss in enumerate(np.sqrt(s)):
        X[:, r] *= ss
    return euclidean_matrix(A, X)


def NetEffect(A, labels, prior, est=False, em=False, rank=128, T=4, trials=30):
    n, c = A.shape[0], np.max(labels) + 1
    E_h = np.ones((n, c)) * (1 / c)
    for p in prior:
        E_h[p] = 0
        E_h[p, labels[p]] = 1
    E_h = csr_matrix(E_h - 1 / c)

    if em:
        A = emphasis_matrix(A, T=T, trials=trials, rank=rank)
    s_A = randomized_svd(A, 1, random_state=None)[1][0]
    A *= (1 / s_A) * 0.9

    if est:
        H_h = fast_cm(A, E_h, prior)
    else:
        H_h = csr_matrix(np.identity(c) - (1 / c))

    prev_pred = np.zeros(n)
    B_h, eps, eps_min = csr_matrix(E_h), np.inf, 1 / (n * c)
    while eps > eps_min:
        B_h_new = E_h + A @ B_h @ H_h
        B_h_new[prior] = E_h[prior]
        eps = np.abs(B_h_new - B_h).mean()
        B_h = B_h_new

    return B_h.toarray() + 1 / c
