import numpy as np
import statsmodels.api as sm


def multipletests(p, method="fdr_bh"):
    not_nan = ~np.isnan(p)
    p_adj = np.full(p.shape, np.nan)
    p_adj[not_nan] = sm.stats.multipletests(p[not_nan], method=method)[1]
    return p_adj
