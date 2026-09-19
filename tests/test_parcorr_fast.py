"""Regression checks for the opt-in analytic ParCorr fast path."""

from copy import deepcopy

import numpy as np
import pytest

from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_fast import ParCorrFast
from tigramite.pcmci import PCMCI


def compare_test(reference, fast, **kwargs):
    a, b = reference.run_test(**kwargs), fast.run_test(**kwargs)
    np.testing.assert_allclose(a[:2], b[:2], atol=1e-10, rtol=0, equal_nan=True)
    if len(a) == 3:
        assert a[2] == b[2]
    return a, b


def attached(values, frame_kwargs=None, test_kwargs=None):
    frame_kwargs, test_kwargs = frame_kwargs or {}, test_kwargs or {}
    a, b = ParCorr(**test_kwargs), ParCorrFast(**test_kwargs)
    a.set_dataframe(DataFrame(values.copy(), **deepcopy(frame_kwargs)))
    b.set_dataframe(DataFrame(values.copy(), **deepcopy(frame_kwargs)))
    return a, b


def query():
    return dict(X=[(0, -1)], Y=[(1, 0)], Z=[(2, -1), (3, -2)],
                tau_max=2, alpha_or_thres=.01)


@pytest.mark.parametrize('seed', range(6))
def test_pcmciplus_equivalence_and_trace(seed):
    values = np.random.default_rng(seed).normal(size=(54, 4))
    for t in range(2, len(values)):
        values[t] += .4 * values[t-1]
        values[t, 2] += .5*values[t, 0] + .5*values[t, 1]
    for offset in range(3):
        models, outputs, traces = [], [], []
        for cls in (ParCorr, ParCorrFast):
            model = PCMCI(DataFrame(values[offset:offset+52]), cls())
            trace, original = [], model.cond_ind_test.run_test
            def record(*args, **kwargs):
                result = original(*args, **kwargs)
                trace.append((deepcopy((args, kwargs)), result))
                return result
            model.cond_ind_test.run_test = record
            outputs.append(model.run_pcmciplus(tau_max=6, pc_alpha=.01))
            models.append(model)
            traces.append(trace)
        a, b = outputs
        np.testing.assert_array_equal(a['graph'], b['graph'])
        for key in ('p_matrix', 'val_matrix'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-10, rtol=0)
        for key in ('sepsets', 'ambiguous_triples', 'conf_matrix'):
            assert a[key] == b[key]
        assert len(traces[0]) == len(traces[1])
        for (qa, ra), (qb, rb) in zip(*traces):
            assert qa == qb
            np.testing.assert_allclose(ra[:2], rb[:2], atol=1e-10, rtol=0)
            assert ra[2] == rb[2]
        assert models[1].cond_ind_test._fast_cache


@pytest.mark.parametrize('case', ['constant_x', 'constant_z', 'singular', 'near_singular',
                                  'tiny_residual', 'near_constant', 'perfect', 'short'])
def test_degenerate_queries(case):
    values = np.random.default_rng(91).normal(size=(52, 4))
    q = dict(X=[(0, 0)], Y=[(1, 0)], Z=[(2, 0), (3, 0)],
             tau_max=0, alpha_or_thres=.01)
    if case == 'constant_x': values[:, 0] = 0.
    if case == 'constant_z': values[:, 2] = 0.
    if case == 'singular': values[:, 3] = values[:, 2]
    if case == 'near_singular': values[:, 3] = values[:, 2] + 1e-9*values[:, 3]
    if case == 'tiny_residual': values[:, 0] = values[:, 2] + 1e-10*values[:, 0]
    if case == 'near_constant': values[:, 0] = 1e10 + 1e-5*values[:, 0]
    if case == 'perfect': values[:, 1] = values[:, 0]
    if case == 'short': values = values[:3]
    a, b = attached(values)
    compare_test(a, b, **q)
    assert not b._fast_cache


@pytest.mark.parametrize('case', ['mask', 'missing', 'multiple', 'vector', 'bootstrap',
                                  'float32', 'shuffle', 'fixed', 'confidence', 'recycle',
                                  'cutoff', 'dtype_mask', 'memory_limit', 'verbose'])
def test_fallbacks(case):
    values = np.random.default_rng(12).normal(size=(52, 4))
    fk, tk, q = {}, {}, query()
    if case == 'mask':
        mask = np.zeros_like(values, dtype=bool); mask[20, 1] = True
        fk['mask'], tk['mask_type'] = mask, 'y'
    if case == 'missing': values[20, 1] = 999.; fk['missing_flag'] = 999.
    if case == 'multiple': values = np.stack([values, values*2]); fk['analysis_mode'] = 'multiple'
    if case == 'vector': fk['vector_vars'] = {i: [(i, 0)] for i in range(4)}
    if case == 'float32': values = values.astype(np.float32)
    if case == 'shuffle': tk.update(significance='shuffle_test', sig_samples=10, sig_blocklength=2)
    if case == 'fixed': tk['significance'] = 'fixed_thres'
    if case == 'confidence': tk['confidence'] = 'analytic'
    if case == 'recycle': tk['recycle_residuals'] = True
    if case == 'cutoff': q['cut_off'] = 'max_lag'
    if case == 'dtype_mask': fk['data_type'] = np.zeros_like(values, dtype=bool)
    if case == 'verbose': tk['verbosity'] = 1
    a, b = attached(values, fk, tk)
    if case == 'bootstrap':
        for test in (a, b):
            test.dataframe.bootstrap = dict(boot_blocklength=2, random_state=np.random.default_rng(7))
    if case == 'memory_limit': b._MAX_CACHE_BYTES = 1
    compare_test(a, b, **q)
    assert not b._fast_cache


@pytest.mark.parametrize('change', ['values', 'replace_values', 'refs', 'replace_refs',
                                   'tau', 'dataframe'])
def test_cache_invalidation_without_freezing_input(change):
    a, b = attached(np.random.default_rng(8).normal(size=(52, 4)))
    q = query()
    compare_test(a, b, **q)
    old_gram = b._gram
    assert b.dataframe.values[0].flags.writeable
    assert b.dataframe.reference_points.flags.writeable
    for test in (a, b):
        if change == 'values': test.dataframe.values[0][-20:, 1] += 7.
        if change == 'replace_values': test.dataframe.values[0] = test.dataframe.values[0][::-1].copy()
        if change == 'refs': test.dataframe.reference_points[:20] = 20
        if change == 'replace_refs': test.dataframe.reference_points = np.array([40, 41, 12, 13, 20, 30, 20])
        if change == 'dataframe': test.set_dataframe(DataFrame(np.random.default_rng(19).normal(size=(60, 4))))
    if change == 'tau': q['tau_max'] = 3
    compare_test(a, b, **q)
    assert b._gram is not old_gram
    for key in a.dataframe.use_indices_dataset_dict:
        np.testing.assert_array_equal(a.dataframe.use_indices_dataset_dict[key],
                                      b.dataframe.use_indices_dataset_dict[key])


def test_cache_hits_deduplication_and_alpha_guard():
    a, b = attached(np.random.default_rng(20).normal(size=(52, 4)))
    q = query(); q['Z'] += [q['Z'][0], q['X'][0], q['Y'][0]]
    ra, _ = compare_test(a, b, **q)
    assert len(b._fast_cache) == 1
    q['alpha_or_thres'] = ra[1]
    aa, bb = compare_test(a, b, **q)
    assert aa == bb  # Exact original OLS result at the decision boundary.
    q['alpha_or_thres'] = None
    compare_test(a, b, **q)
    q['X'], q['Y'] = q['Y'], q['X']
    compare_test(a, b, **q)
    assert len(b._fast_cache) == 1


@pytest.mark.parametrize('change', ['index', 'lag', 'multivariate', 'negative_tau'])
def test_invalid_arguments_retain_original_exception(change):
    a, b = attached(np.random.default_rng(8).normal(size=(52, 4)))
    q = query()
    if change == 'index': q['X'] = [(4, -1)]
    if change == 'lag': q['X'] = [(0, 1)]
    if change == 'multivariate': q['X'] = [(0, -1), (1, -1)]
    if change == 'negative_tau': q['tau_max'] = -1
    try:
        a.run_test(**q)
    except Exception as exc:
        with pytest.raises(type(exc)):
            b.run_test(**q)
    else:
        compare_test(a, b, **q)


def test_interleaved_cutoffs_restore_sample_indices():
    a, b = attached(np.random.default_rng(3).normal(size=(52, 4)))
    for cutoff in ('2xtau_max', 'max_lag', '2xtau_max'):
        compare_test(a, b, **query(), cut_off=cutoff)
        np.testing.assert_array_equal(a.dataframe.use_indices_dataset_dict[0],
                                      b.dataframe.use_indices_dataset_dict[0])


@pytest.mark.parametrize('method', ['get_dependence_measure', 'get_analytic_significance'])
def test_custom_estimators_retain_original_dispatch(method):
    a, b = attached(np.random.default_rng(15).normal(size=(52, 4)))
    for test in (a, b):
        setattr(test, method, lambda *args, **kwargs: .3)
    compare_test(a, b, **query())
    assert not b._fast_cache
