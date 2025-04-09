import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def cross_corr(X, flat=False):
    """
    Args:
        X: neurons x time
    Returns:
        corr: 1-D vector of length `len(neurons) x (len(neurons) - 1) / 2`
    Example:
    >>> from scipy.spatial import distance
    ... X = rng.random((100, 100))
    ... corr = cross_corr(X)
    ... corr2 = distance.squareform(distance.pdist(X, metric='correlation'))
    ... assert np.isclose(corr, 1 - corr2).all()
    True
    >>> from timeit import timeit
    ... X = rng.random((1000, 1000))
    ... timeit(lambda: cross_corr(X), number=100)
    < 1 sec
    >>> from timeit import timeit
    ... X = rng.random((1000, 1000))
    ... timeit(lambda: distance.pdist(X, metric='correlation'), number=100)
    ~ 20 sec
    """
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    corr = X @ X.T
    assert np.isclose(np.diag(corr), 1).all()
    # set diagonal to 1
    corr[np.diag_indices_from(corr)] = 1
    if flat:
        corr = corr[np.triu_indices_from(corr, k=1)]
    return corr


def pair_corr(X, Y):
    """
    Args:
        X, Y: neurons x time
    Returns:
        corr: 1-D vector of length `len(neurons) x len(neurons)`
    Example:
    write a test to compare with numpy.corrcoef:
    >>> import numpy as np
    ... rng = np.random.default_rng(0)
    ... X = rng.random((100, 200))
    ... Y = rng.random((100, 200))
    ... corr = pair_corr(X, Y)
    ... corr2 = np.corrcoef(X, Y)[100:, :100].diagonal()
    ... assert np.isclose(corr, corr2).all()
    """
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    corr = (X * Y).sum(axis=1)
    return corr

def _pair_corr_by_list(pair_list, X):
    return pair_corr(X[pair_list[:, 0]], X[pair_list[:, 1]])

def pair_corr_by_list(X, pair_list, batch_size=100_000, pbar=True):
    """
    Args:
        X: neurons x time
        pair_list: numpy array of (i, j) pairs
        batch_size: batch size for parallelization across pairs
    Returns:
        corr: 1-D vector of length `len(pair_list)`, each element is the correlation between the corresponding pair
    Example:
    >>> import numpy as np
    ... rng = np.random.default_rng(0)
    ... X = rng.random((100, 200))
    ... pair_list = [(i, j) for i in range(100) for j in range(100)]
    ... corr = pair_corr(X, pair_list)
    ... corr2 = np.corrcoef(X)[np.triu_indices_from(np.corrcoef(X), k=1)]
    ... assert np.isclose(corr, corr2).all()
    """
    edge_split = np.array_split(pair_list, len(pair_list) // batch_size + 1)


    with mp.Pool(mp.cpu_count()) as pool:
        corr = list(
            tqdm(
                pool.imap(
                    partial(_pair_corr_by_list, X=X),
                    edge_split,
                ),
                total=len(edge_split),
                disable=not pbar,
            )
        )
    return np.concatenate(corr)


def spawn_rng(rng, n_children):
    # ref: https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx#L597
    return [
        type(rng)(type(rng.bit_generator)(seed=s))
        for s in rng.bit_generator._seed_seq.spawn(n_children)
    ]
