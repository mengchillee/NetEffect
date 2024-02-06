import numpy as np
from collections import defaultdict

from scipy.sparse import csr_matrix, kron
from scipy.stats import chi2_contingency, chi2
from sklearn.linear_model import RidgeCV


def check_cm(H_h):
    H_h = np.array(H_h, copy=True)
    c = H_h.shape[0]
    H_h += 1 / c
    H_h[H_h < 0] = 0
    for i in range(c):
        H_h[i] /= H_h[i].sum()
    return H_h - 1 / c


def fast_cm(A, B_h, prior):
    n, c = B_h.shape
    PQ = csr_matrix(kron(np.identity(c), A @ B_h))
    B_h_v = B_h.toarray().T.reshape(-1)

    nvi = np.array([prior])
    for i in range(1, c):
        nvi = np.concatenate([nvi, [prior + n * i]], axis=0)
    nvi = nvi.T.reshape(-1)

    del_idx = []
    for j, i in enumerate(np.sum(np.abs(PQ[nvi]), axis=1)):
        if i == 0:
            del_idx.append(j)
    nvi = np.delete(nvi, del_idx)
    
    reg = RidgeCV(fit_intercept=False)
    reg.fit(PQ[nvi], B_h_v[nvi])
    
    H_h = reg.coef_.reshape(c, c).T
    return csr_matrix(check_cm(H_h))


def fast_cm_hom(A, B_h, prior, cnum=[]):
    n, c = B_h.shape
    if len(cnum) == c:
        return csr_matrix(np.identity(c) - 1 / c)
    
    PQ = csr_matrix(kron(np.identity(c), A @ B_h))
    B_h_v = B_h.toarray().T.reshape(-1)

    nvi = np.array([prior])
    for i in range(1, c):
        nvi = np.concatenate([nvi, [prior + n * i]], axis=0)
    nvi = nvi.T.reshape(-1)

    del_idx = []
    for j, i in enumerate(np.sum(np.abs(PQ[nvi]), axis=1)):
        if i == 0:
            del_idx.append(j)
    nvi = np.delete(nvi, del_idx)
    
    cnum_arr = np.concatenate([np.arange(i * c, (i + 1) * c) for i in cnum])
    cnum_arr_c = np.concatenate([np.arange(i * c, (i + 1) * c) for i in np.arange(c) if i not in cnum])
    cnum_c = np.array([i for i in range(c) if i not in cnum])
    
    identity = np.identity(c)[:, cnum]
    diff = PQ[nvi][:, cnum_arr] @ identity.T.reshape(-1)
    
    reg = RidgeCV(alphas=np.arange(1, 11) / 1000, fit_intercept=False)
    reg.fit(PQ[nvi][:, cnum_arr_c], B_h_v[nvi] - diff)
    H = reg.coef_.reshape(c - len(cnum), c).T + 1 / c
    
    c1, c2, H_h = 0, 0, []
    for i in range(c):
        if i in cnum_c:
            H_h.append(H[:, c1])
            c1 += 1
        elif i in cnum:
            H_h.append(identity[:, c2])
            c2 += 1
    H_h = np.array(H_h).T - 1 / c
    return csr_matrix(check_cm(H_h))


def net_effect_test(A, labels, prior):
    c = np.max(labels) + 1

    sampled_edges = set()
    prior_set = set(list(prior))
    for e1, e2 in zip(*A.nonzero()):
        if e1 < e2 and e1 in prior_set and e2 in prior_set:
            sampled_edges.add(tuple(np.sort([e1, e2])))
    sampled_edges = np.array(list(sampled_edges))
        
    a, b = defaultdict(list), []
    for i in range(1000):
        pvalues = np.zeros((c, c))
        for c1 in range(c):
            for c2 in range(c1 + 1, c):
                counting = np.zeros((2, 2))
                np.random.shuffle(sampled_edges)
                for e1, e2 in sampled_edges:
                    if labels[e1] == c1 and labels[e2] == c1:
                        counting[1, 1] += 2
                    elif labels[e1] == c1 and labels[e2] == c2:
                        counting[0, 1] += 1
                        counting[1, 0] += 1
                    elif labels[e1] == c2 and labels[e2] == c1:
                        counting[1, 0] += 1
                        counting[0, 1] += 1
                    elif labels[e1] == c2 and labels[e2] == c2:
                        counting[0, 0] += 2
                    if counting.sum() > 500:
                        break
                sta, _, dof, _ = chi2_contingency(counting / 2)
                a[(c1, c2)].append(sta)
                b.append(dof)

    pvalues = np.zeros((c, c))
    for (c1, c2), ss in a.items():
        pvalues[c1, c2] = pvalues[c2, c1] = chi2.sf(np.abs(np.mean(a[c1, c2])), df=np.mean(b))
    
    return pvalues
