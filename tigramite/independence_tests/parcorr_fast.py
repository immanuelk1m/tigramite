"""Opt-in Gram-matrix acceleration of analytic ParCorr tests.

License: GNU General Public License v3.0 (see license.txt).
"""

import math
from hashlib import sha1

import numpy as np
from scipy.special import stdtr

from tigramite.data_processing import DataFrame
from .parcorr import ParCorr


class ParCorrFast(ParCorr):
    """ParCorr with reusable lagged cross-products for complete float64 data.

    Only ``run_test`` with analytic significance, scalar variables, one dataset
    and ``cut_off='2xtau_max'`` is accelerated. Other calls use ParCorr unchanged.
    Masks, missing values, bootstrap, vector variables, confidence estimation
    and residual recycling use the original path. PCMCI/PCMCI+ need no changes.

    Data and reference points remain writable. Their content is checked before
    reusing cross-products, including in-place edits and rolling-window changes.
    This costs O(T*N) per call; one test instance must not be used concurrently.
    Cache memory for lagged rows and their Gram matrix is limited to 64 MiB.

    Ill-conditioned, near-constant, small-residual and near-threshold tests use
    the original OLS path. Floating-point results may differ; the guards are
    conservative heuristics, not a universal error bound.

    Example: ``PCMCI(dataframe, ParCorrFast(significance='analytic'))``.
    Pass the same ``run_pcmciplus`` arguments as with ordinary ParCorr.
    """

    _MAX_CACHE_BYTES = 64 * 1024**2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = None
        self._fast_cache = {}
        self._gram = None

    def set_dataframe(self, dataframe):
        super().set_dataframe(dataframe)
        self._state = None
        self._fast_cache = {}
        self._gram = None

    def _prepare(self, tau_max):
        """Build cross-products once for exactly the samples used by ParCorr."""
        frame = self.dataframe
        if (type(frame) is not DataFrame or frame.analysis_mode != 'single'
                or frame.M != 1 or len(frame.values) != 1
                or frame.mask is not None or frame.missing_flag is not None
                or frame.data_type is not None or frame.has_vector_data
                or frame.bootstrap is not None or self.mask_type is not None
                or self.significance != 'analytic' or self.confidence is not None
                or self.recycle_residuals or self.verbosity != 0
                or getattr(self.get_dependence_measure, '__func__', None) is not ParCorr.get_dependence_measure
                or getattr(self.get_analytic_significance, '__func__', None) is not ParCorr.get_analytic_significance):
            return False
        dataset = next(iter(frame.values))
        values, refs = frame.values[dataset], frame.reference_points
        if (values.dtype != np.float64 or values.ndim != 2
                or frame.N != values.shape[1] or frame.T[dataset] != len(values)
                or frame.time_offsets[dataset] != 0
                or not isinstance(refs, np.ndarray) or refs.ndim != 1
                or not np.issubdtype(refs.dtype, np.integer)):
            return False
        # Check content, not just object identity: DataFrame is publicly mutable.
        state = (tau_max, dataset, values.shape, refs.shape, refs.dtype.str,
                 sha1(np.ascontiguousarray(values)).digest(),
                 sha1(np.ascontiguousarray(refs)).digest())
        if state == self._state:
            return self._gram is not None
        self._state, self._gram, self._fast_cache = state, None, {}
        n = np.count_nonzero((refs >= 2*tau_max) & (refs < len(values)))
        size = frame.N * (2*tau_max + 1)
        # Include temporary lagged rows, centering and the persistent Gram matrix.
        if (n < 3 or 8 * (4*size*n + size*size) > self._MAX_CACHE_BYTES
                or not np.isfinite(values).all()):
            return False
        nodes = [(i, -lag) for i in range(frame.N)
                 for lag in range(2*tau_max + 1)]
        raw, _, _ = frame.construct_array(X=nodes, Y=[], Z=[], tau_max=tau_max)
        means = raw.mean(axis=1, keepdims=True)
        centered = raw - means
        scale = centered.std(axis=1)
        self._unsafe = ((raw.std(axis=1) == 0.)
                        | (np.linalg.norm(centered, axis=1)
                           < 1e-13 * np.abs(means[:, 0])))
        np.divide(centered, scale[:, None], out=centered,
                  where=scale[:, None] != 0.)
        self._gram = centered @ centered.T
        self._node_index = dict(zip(nodes, range(size)))
        self._n_samples = raw.shape[1]
        self._indices = {k: v.copy() for k, v in frame.use_indices_dataset_dict.items()}
        return True

    def _partial_correlation(self, ids):
        """Schur complement equals the OLS residual cross-product matrix."""
        g = self._gram
        x, y = ids[:2]
        xx, yy, xy = g[x, x], g[y, y], g[x, y]
        if len(ids) > 2:
            z = ids[2:]
            zz, za = g[np.ix_(z, z)], g[np.ix_(z, [x, y])]
            try:
                eig = np.linalg.eigvalsh(zz)
                if eig[0] <= 0 or eig[-1]/eig[0] > 1e5:
                    return None
                solved = za / zz[0, 0] if len(z) == 1 else np.linalg.solve(zz, za)
            except np.linalg.LinAlgError:
                return None
            adjustment = za.T @ solved
            xx, yy, xy = xx-adjustment[0, 0], yy-adjustment[1, 1], xy-adjustment[0, 1]
        if min(xx, yy) < 1e-6 * self._n_samples:
            return None
        r = float(xy / math.sqrt(xx*yy))
        # Perfect/near-perfect fits are sensitive to cancellation; preserve OLS.
        if not math.isfinite(r) or abs(r) >= 1. - 1e-12:
            return None
        return r

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max', alpha_or_thres=None):
        """Return the usual ParCorr tuple, preserving validation and fallbacks."""
        fallback = super().run_test
        if (cut_off != '2xtau_max' or len(X) != 1 or len(Y) != 1
                or (Z is not None and not isinstance(Z, (list, tuple)))
                or not isinstance(tau_max, (int, np.integer)) or tau_max < 0
                or not self._prepare(tau_max)):
            return fallback(X, Y, Z, tau_max, cut_off, alpha_or_thres)
        nodes = list(X) + list(Y) + (list(Z) if Z is not None else [])
        if any(not isinstance(node, tuple) or len(node) != 2
               or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
                      for v in node) for node in nodes):
            return fallback(X, Y, Z, tau_max, cut_off, alpha_or_thres)
        z = [node for node in dict.fromkeys(nodes[2:]) if node not in nodes[:2]]
        try:
            ids = [self._node_index[node] for node in nodes[:2] + z]
        except KeyError:
            return fallback(X, Y, Z, tau_max, cut_off, alpha_or_thres)
        dof = self._n_samples - len(ids)
        if dof < 1 or self._unsafe[ids].any():
            return fallback(X, Y, Z, tau_max, cut_off, alpha_or_thres)
        key = (tuple(sorted(ids[:2])), tuple(sorted(ids[2:])))
        if key not in self._fast_cache:
            r = self._partial_correlation(ids)
            if r is None:
                return fallback(X, Y, Z, tau_max, cut_off, alpha_or_thres)
            t = r * math.sqrt(dof / (1. - r*r))
            self._fast_cache[key] = (r, float(2. * stdtr(dof, -abs(t))))
        r, p = self._fast_cache[key]
        # Check each alpha, even on a cache hit. Do not cache a decision.
        if alpha_or_thres is not None and abs(p - alpha_or_thres) < 1e-8:
            return fallback(X, Y, Z, tau_max, cut_off, alpha_or_thres)
        dependent = None if alpha_or_thres is None else p <= alpha_or_thres
        self.ci_results[(tuple(X), tuple(Y), tuple(z))] = (r, p, dependent)
        self.dataframe.use_indices_dataset_dict = {k: v.copy() for k, v in self._indices.items()}
        return (r, p) if alpha_or_thres is None else (r, p, dependent)
