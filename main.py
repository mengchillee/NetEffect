import numpy as np
import argparse
import os
import time
from tqdm import tqdm

from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score

from neteffect import NetEffect


def accuracy(labels, prior, pred):
    idx = [i for i in range(len(labels)) if i not in prior]
    return accuracy_score(labels[idx], pred[idx])

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fb',
                    help='Name of input dataset')
parser.add_argument('--T', type=float, default=4,
                    help='Length of random walk')
parser.add_argument('--trials', type=float, default=10,
                    help='Number of random walk trials')
parser.add_argument('--rank', type=int, default=256,
                    help='Number of embedding dimensions')

if __name__ == '__main__':    

    args = parser.parse_args()

    edges = np.loadtxt(os.path.join('data', args.dataset, 'edges.txt'), dtype=int)
    labels = np.loadtxt(os.path.join('data', args.dataset, 'labels.txt'), dtype=int)
    priors = []
    for i in range(1, 6):
        priors.append(np.loadtxt(os.path.join('data', args.dataset, 'labels' + str(i) + '.txt'), dtype=int))

    n, c = len(labels), np.max(labels) + 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)

    acc1, acc2 = [], []
    time1, time2 = [], []
    for idx, prior in enumerate(tqdm(priors)): 
        t1 = time.time()
        ### NetEffect-Hom
        B_h1 = NetEffect(A, labels, prior, est=False, em=True, rank=args.rank, T=args.T, trials=args.trials)
        pred1 = np.argmax(B_h1, axis=1)
        t2 = time.time()
        ### NetEffect
        B_h2 = NetEffect(A, labels, prior, est=True, em=True, rank=args.rank, T=args.T, trials=args.trials)
        pred2 = np.argmax(B_h2, axis=1)
        t3 = time.time()
        
        acc1.append(accuracy(labels, prior, pred1))
        acc2.append(accuracy(labels, prior, pred2))
        time1.append(t2 - t1)
        time2.append(t3 - t2)

    print('NetEffect-Hom')
    print('Accuracy: %.3f +- %.3f' % (np.mean(acc1), np.std(acc1)))
    print('Run Time: %.3f' % np.mean(time1))
    print()
    print('NetEffect')
    print('Accuracy: %.3f +- %.3f' % (np.mean(acc2), np.std(acc2)))
    print('Run Time: %.3f' % np.mean(time2))
    print()

