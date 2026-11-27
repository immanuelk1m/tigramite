"""Regression checks for RPCMCI output and prediction contracts."""

from copy import deepcopy
import struct

import numpy as np
import pytest

pytest.importorskip("ortools")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

from joblib import parallel_backend
from ortools.linear_solver import linear_solver_pb2, pywraplp
from scipy import stats
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.rpcmci import RPCMCI


def assert_identical(actual, expected):
    """Compare nested results, including floating-point bytes and scalar types."""
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(actual, np.generic):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(actual, float):
        assert struct.pack("!d", actual) == struct.pack("!d", expected)
    elif isinstance(actual, dict):
        assert tuple(actual) == tuple(expected)
        for key in actual:
            assert_identical(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_identical(left, right)
    else:
        assert actual == expected


@pytest.fixture
def values():
    data = np.random.default_rng(42).normal(size=(48, 2))
    for t in range(1, len(data)):
        data[t] += .7 * data[t - 1]
        data[t, 1] += .4 * data[t - 1, 0]
    return data


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


def run(model, **kwargs):
    options = dict(num_regimes=1, max_transitions=2, num_iterations=3,
                   max_anneal=2, switch_thres=0., n_jobs=1)
    options.update(kwargs)
    return model.run_rpcmci(**options)


@pytest.mark.parametrize("tau_max", [1, 2])
def test_single_regime_matches_pcmci_and_convergence(values, tau_max):
    result = run(RPCMCI(DataFrame(values.copy()), seed=17), tau_max=tau_max)
    expected = PCMCI(DataFrame(values.copy(), mask=np.zeros_like(values, dtype=bool)),
                     ParCorr(mask_type="y")).run_pcmci(
                         tau_min=1, tau_max=tau_max, pc_alpha=.2, alpha_level=.01)

    assert result is not None
    assert set(result) == {"regimes", "causal_results", "diff_g_f",
                           "error_free_annealings"}
    assert_identical(result["regimes"], np.ones((1, len(values))))
    assert_identical(result["causal_results"], {0: expected})
    assert result["error_free_annealings"] == 2
    all_differences, best_differences = result["diff_g_f"]
    expected_differences = []
    for seed in [17, 18]:
        initial = np.random.default_rng(seed).uniform(0, 1, size=(1, len(values)))
        expected_differences.append([np.sum(np.abs(1. - initial)), np.float64(0.)])
    assert_identical(all_differences, expected_differences)
    assert_identical(best_differences, all_differences[0])


@pytest.mark.parametrize("tau_max", [1, 2])
def test_parallel_annealings_preserve_entire_result(values, tau_max):
    options = dict(num_regimes=2, num_iterations=1, tau_max=tau_max,
                   switch_thres=.05)
    serial = run(RPCMCI(DataFrame(values.copy()), seed=17), **options)
    with parallel_backend("loky", inner_max_num_threads=1):
        parallel = run(RPCMCI(DataFrame(values.copy()), seed=17),
                       n_jobs=2, **options)
    assert serial is not None
    assert_identical(parallel, serial)


def test_repeated_calls_preserve_entire_result_and_previous_result(values):
    model = RPCMCI(DataFrame(values.copy()), seed=17)
    first = run(model)
    snapshot = deepcopy(first)
    second = run(model)
    assert first is not None
    assert_identical(second, snapshot)
    assert_identical(first, snapshot)


def test_empty_parent_sets_produce_finite_results(values):
    result = run(RPCMCI(DataFrame(values.copy()), seed=17), alpha_level=0.)
    assert result is not None
    causal = result["causal_results"][0]
    assert not np.any(causal["graph"] != "")
    assert np.isfinite(causal["val_matrix"]).all()
    assert np.isfinite(causal["p_matrix"]).all()
    assert_identical(result["regimes"], np.ones((1, len(values))))


@pytest.mark.parametrize("failure", ["too_few_samples", "missing_values"])
def test_failed_annealings_return_none(values, failure, capsys):
    options = {}
    if failure == "too_few_samples":
        options["switch_thres"] = 1.
        frame = DataFrame(values.copy())
    else:
        # Prediction drops missing observations while RPCMCI currently expects
        # T - tau_max residual rows. Preserve its existing failure behavior.
        data = values.copy()
        data[20, :] = 999.
        frame = DataFrame(data, missing_flag=999.)
    result = run(RPCMCI(frame, seed=17), **options)
    assert result is None
    assert "No annealings have converged. Run failed." in capsys.readouterr().out


def test_custom_estimators_receive_every_regime_fit_and_test(values):
    class RecordingParCorr(ParCorr):
        calls = []

        def run_test(self, *args, **kwargs):
            self.calls.append((deepcopy(args), deepcopy(kwargs)))
            return super().run_test(*args, **kwargs)

    class RecordingRegression(LinearRegression):
        fits = []
        predictions = []

        def fit(self, X, y, **kwargs):
            self.fits.append((X.copy(), y.copy()))
            return super().fit(X, y, **kwargs)

        def predict(self, X):
            self.predictions.append(X.copy())
            return super().predict(X)

    model = RPCMCI(DataFrame(values.copy()), cond_ind_test=RecordingParCorr(),
                   prediction_model=RecordingRegression(), seed=17)
    result = run(model, max_anneal=1, alpha_level=1.)
    expected = run(RPCMCI(DataFrame(values.copy()), seed=17),
                   max_anneal=1, alpha_level=1.)
    assert result is not None
    assert_identical(result, expected)
    iterations = len(result["diff_g_f"][0][0])
    assert len(RecordingRegression.fits) == iterations * values.shape[1]
    assert len(RecordingRegression.predictions) == iterations * values.shape[1]
    assert len(RecordingParCorr.calls) > 0


@pytest.mark.parametrize("max_transitions", [2, np.int64(2), float("inf"), 10**100])
def test_lp_preserves_original_model_and_solution(values, monkeypatch, max_transitions):
    """Keep the original ordered LP, including duplicate persistence rows."""
    original_solve = pywraplp.Solver.Solve
    checked_statuses = []
    num_regimes, length = 2, len(values)

    def checked_solve(solver, *args, **kwargs):
        model = linear_solver_pb2.MPModelProto()
        solver.ExportModelToProto(model)
        reference = pywraplp.Solver.CreateSolver("GLOP")
        variables = [reference.NumVar(0, 1, f"x_{i}")
                     for i in range(num_regimes * length)]
        variables += [reference.NumVar(0, reference.infinity(), f"eta_{i}")
                      for i in range(num_regimes * length - 1)]
        reference.Minimize(sum(model.variable[i].objective_coefficient * variables[i]
                               for i in range(num_regimes * length)))
        for t in range(length):
            reference.Add(sum(variables[k * length + t]
                              for k in range(num_regimes)) == 1)
        for k in range(num_regimes):
            for t in range(length - 1):
                left, right = variables[k * length + t:k * length + t + 2]
                eta = variables[k * length + t + num_regimes * length]
                reference.Add(right - left - eta <= 0)
                reference.Add(-right + left - eta <= 0)
                reference.Add(sum(variables[k * length + time + num_regimes * length]
                                  for time in range(length - 1)) <= max_transitions)
        expected_model = linear_solver_pb2.MPModelProto()
        reference.ExportModelToProto(expected_model)
        assert model == expected_model
        expected_status = original_solve(reference, *args, **kwargs)
        status = original_solve(solver, *args, **kwargs)
        assert status == expected_status
        if status == pywraplp.Solver.OPTIMAL:
            assert_identical(solver.Objective().Value(), reference.Objective().Value())
            assert_identical(np.array([v.solution_value() for v in solver.variables()]),
                             np.array([v.solution_value() for v in reference.variables()]))
        checked_statuses.append(status)
        return status

    monkeypatch.setattr(pywraplp.Solver, "Solve", checked_solve)
    result = run(RPCMCI(DataFrame(values.copy()), seed=17), num_regimes=num_regimes,
                 max_transitions=max_transitions, num_iterations=1, max_anneal=1)
    assert len(checked_statuses) == 1
    assert (result is not None) == (checked_statuses[0] == pywraplp.Solver.OPTIMAL)


@pytest.mark.parametrize("dtype", [float, np.float32, np.float64, np.longdouble])
def test_parcorr_analytic_significance_matches_distribution(dtype):
    test = ParCorr()
    one, zero = dtype(1), dtype(0)
    correlations = [-one, -np.nextafter(one, zero), dtype(-0.45), zero,
                    dtype(0.45), np.nextafter(one, zero), one, dtype(np.nan)]
    for value in correlations:
        for dof in (0, 1, 17, 498, 2**64):
            if dof < 1:
                expected = np.nan
            elif abs(value) == 1:
                expected = 0.0
            else:
                try:
                    t = value * np.sqrt(dof / (1. - value * value))
                    expected = stats.t.sf(np.abs(t), dof) * 2
                except TypeError as error:
                    with pytest.raises(TypeError) as raised:
                        test.get_analytic_significance(value, T=dof + 2, dim=2, xyz=None)
                    assert str(raised.value) == str(error)
                    continue
            actual = test.get_analytic_significance(value, T=dof + 2, dim=2, xyz=None)
            assert_identical(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(2, 2), (2, 120), (5, 120)])
def test_parcorr_statistic_matches_scipy_residual_correlation(dtype, shape):
    test = ParCorr()
    data = np.random.default_rng(31).normal(size=shape).astype(dtype)
    reference = data.copy()
    x = test._get_single_residuals(reference, 0)
    y = test._get_single_residuals(reference, 1)
    expected = stats.pearsonr(x, y).statistic
    assert_identical(test.get_dependence_measure(data, xyz=None), expected)
    assert_identical(data, reference)


def test_parcorr_preserves_scipy_dispatch_and_near_constant_warning(monkeypatch):
    test = ParCorr()
    x = 1. + np.spacing(1.) * np.arange(8)
    y = np.arange(8, dtype=float)[::-1]
    inputs = (x, y)
    monkeypatch.setattr(test, "_get_single_residuals",
                        lambda array, target_var: inputs[target_var])
    with pytest.warns(stats.NearConstantInputWarning):
        expected = stats.pearsonr(x, y).statistic
    with pytest.warns(stats.NearConstantInputWarning):
        actual = test.get_dependence_measure(np.empty((2, 8)), xyz=None)
    assert_identical(actual, expected)

    monkeypatch.setattr(stats, "pearsonr", lambda x, y: (np.float64(0.125), None))
    assert_identical(test.get_dependence_measure(np.empty((2, 8)), xyz=None), np.float64(0.125))


@pytest.mark.parametrize("constant", [False, True])
def test_ci_array_preserves_constant_row_handling_and_copy(values, constant):
    data = values.copy(order="F")
    if constant:
        data[:, 0] = 1.
    test = ParCorr()
    test.set_dataframe(DataFrame(data, data_type=np.zeros_like(data, dtype=int)))
    result = test._get_array([(0, -1)], [(1, 0)], [(1, -1)],
                             tau_max=1, remove_constant_data=True)
    array, xyz, nodes, data_type, clean, clean_xyz, clean_nodes, clean_type = result
    dropped = np.flatnonzero(array.std(axis=1) == 0)
    assert dropped.size == int(constant)
    expected_array = np.delete(array, dropped, axis=0)
    expected_type = np.delete(data_type, dropped, axis=0)
    assert_identical(clean, expected_array)
    assert_identical(clean_xyz, np.delete(xyz, dropped))
    assert_identical(clean_type, expected_type)
    assert_identical(clean_nodes, (nodes[0] if not constant else [], nodes[1], nodes[2]))
    assert clean.strides == expected_array.strides
    assert clean_type.strides == expected_type.strides
    assert not np.shares_memory(clean, array)
    assert not np.shares_memory(clean_xyz, xyz)
    assert not np.shares_memory(clean_type, data_type)
