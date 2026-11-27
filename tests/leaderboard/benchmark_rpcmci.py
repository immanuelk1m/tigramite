"""Reproduce RPCMCI ablations in isolated copies of the baseline package."""

import argparse
import contextlib
import hashlib
from importlib.metadata import version
import io
import json
import marshal
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time


BASELINE = "9cdba1343fd4763da72d2b04e9687c75d4318051"
PATCH_FILE = "rpcmci_variants.json"
CASES = ("tutorial", "multivariate", "repeat")
LP_CASE_COUNT = 217
THREAD_LIMITS = {
    name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
}


def fingerprint(value):
    """Serialize the nested result with type, shape, and exact numeric bits."""
    import numpy as np

    if isinstance(value, (np.ndarray, np.generic)):
        return ("numpy", type(value).__name__, value.dtype.str,
                value.shape, value.tobytes())
    if isinstance(value, float):
        return ("float", struct.pack("!d", value))
    if isinstance(value, dict):
        return ("dict", [(fingerprint(k), fingerprint(v)) for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [fingerprint(item) for item in value])
    return (type(value).__name__, value)


def digest(value):
    return hashlib.sha256(marshal.dumps(fingerprint(value), 2)).hexdigest()


def environment_details():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "ortools", "joblib")},
        "thread_limits": THREAD_LIMITS,
    }


def make_case(case):
    import numpy as np

    config = dict(num_regimes=2, max_transitions=4, max_anneal=1, n_jobs=1)
    if case in ("tutorial", "parallel", "missing"):
        data = np.random.RandomState(43).randn(500, 3)
        for t in range(1, len(data)):
            winter = (t % 365) < 90 or (t % 365) > 240
            data[t, 0] += 0.4 * data[t - 1, 0]
            if winter:
                data[t, 1] += 0.3 * data[t - 1, 1] + 0.7 * data[t - 1, 0]
                data[t, 2] += 0.3 * data[t - 1, 2] + 0.7 * data[t - 1, 0]
            else:
                data[t, 1] += 0.3 * data[t - 1, 1] - 0.7 * data[t - 1, 0]
                data[t, 2] += 0.3 * data[t - 1, 2]
        if case == "parallel":
            config.update(max_anneal=2, n_jobs=2)
        if case == "missing":
            data[88, 1] = -999.0
    elif case == "multivariate":
        data = np.random.default_rng(43).normal(size=(160, 8))
        for t in range(1, len(data)):
            data[t] += 0.55 * data[t - 1] + (0.65 if t < 80 else -0.65) * np.roll(data[t - 1], 1)
        config.update(max_anneal=2, tau_max=2, num_iterations=1)
    elif case in ("repeat", "short"):
        data = np.random.default_rng(28).normal(size=(96, 3))
        config.update(num_regimes=1, switch_thres=0.0)
        if case == "short":
            data = data[:5]
    else:
        raise ValueError(case)
    return data, config


def lp_solver():
    """Expose the existing nested solver for model-level parity checks."""
    import ast
    import inspect
    import textwrap
    from tigramite import rpcmci

    method = ast.parse(textwrap.dedent(inspect.getsource(rpcmci.RPCMCI.run_rpcmci))).body[0]
    solver = next(node for node in method.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_optimize_gamma")
    solver.args.args.insert(0, ast.arg(arg="self"))
    namespace = vars(rpcmci).copy()
    module = ast.fix_missing_locations(ast.Module(body=[solver], type_ignores=[]))
    exec(compile(module, "<RPCMCI LP check>", "exec"), namespace)
    return namespace["_optimize_gamma"]


def lp_check():
    import numpy as np
    from ortools.linear_solver import linear_solver_pb2, pywraplp
    from types import SimpleNamespace

    solver = lp_solver()
    original_solve = pywraplp.Solver.Solve
    models = []

    def checked_solve(self, *args, **kwargs):
        proto = linear_solver_pb2.MPModelProto()
        self.ExportModelToProto(proto)
        models.append(hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest())
        return original_solve(self, *args, **kwargs)

    pywraplp.Solver.Solve = checked_solve
    try:
        rng = np.random.default_rng(8)
        shapes = [(k, t, tr) for k in (2, 3, 4) for t in (8, 19, 50)
                  for tr in (0, 1, 2, 4)]
        cases = [(np.zeros((k, t)), tr) for k, t, tr in shapes]
        cases += [(rng.integers(0, 3, size=(k, t)).astype(float), tr)
                  for k, t, tr in shapes for _ in range(5)]
        cases += [(rng.integers(0, 3, size=(2, 500)).astype(float), 4)]
        owner = SimpleNamespace(verbosity=0)
        outputs = [digest(solver(owner, cost, bound)) for cost, bound in cases]
        return {"models": models, "outputs": outputs}
    finally:
        pywraplp.Solver.Solve = original_solve


def cache_trace():
    """Count hit branches in an untimed RPCMCI run."""
    import inspect
    from tigramite.rpcmci import RPCMCI

    source, start = inspect.getsourcelines(RPCMCI.run_rpcmci)
    markers = {
        "result, ci_result, used_indices = request_cache[key]": "ci",
        "residuals[k, int(tau_max):, :] = prediction_residuals[prediction_key]": "prediction",
    }
    branches = {start + i: markers[line.strip()] for i, line in enumerate(source)
                if line.strip() in markers}
    hits = dict.fromkeys(markers.values(), 0)

    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_name not in ("cached_run_test", "one_annealing_step"):
            return None
        if event == "line" and frame.f_lineno in branches:
            hits[branches[frame.f_lineno]] += 1
        return trace

    return hits, trace


def worker():
    from tigramite.data_processing import DataFrame
    from tigramite.rpcmci import RPCMCI

    for command in sys.stdin:
        case = json.loads(command)
        if case == "lp":
            record = lp_check()
        else:
            count_hits = case.startswith("hits:")
            case = case.removeprefix("hits:")
            data, config = make_case(case)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if count_hits:
                    hits, tracer = cache_trace()
                    sys.settrace(tracer)
                try:
                    start = time.perf_counter()
                    result = RPCMCI(
                        DataFrame(data, missing_flag=-999.0 if case == "missing" else None),
                        seed=43,
                    ).run_rpcmci(**config)
                    elapsed = time.perf_counter() - start
                finally:
                    if count_hits:
                        sys.settrace(None)
            record = hits if count_hits else {
                "seconds": elapsed,
                "digest": digest(result),
                "successful": 0 if result is None else result["error_free_annealings"],
            }
        print("RESULT " + json.dumps(record), flush=True)


def patches_from(path):
    return {name: "\n".join(lines) + "\n"
            for name, lines in json.loads(path.read_text()).items()}


class Variant:
    def __init__(self, directory):
        environment = dict(os.environ, **THREAD_LIMITS, PYTHONPATH=str(directory))
        self.process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"], cwd=directory,
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )

    def evaluate(self, case):
        self.process.stdin.write(json.dumps(case) + "\n")
        self.process.stdin.flush()
        for line in self.process.stdout:
            if line.startswith("RESULT "):
                return json.loads(line.removeprefix("RESULT "))
        raise RuntimeError(f"worker exited during {case} ({self.process.wait()})")

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.wait(timeout=30)


def summary(records, names):
    import numpy as np

    result = {}
    for case, rows in records.items():
        result[case] = {"samples": rows, "variants": {}}
        baseline = np.array([row["base"]["seconds"] for row in rows])
        rng = np.random.default_rng(2026)
        resamples = rng.integers(0, len(rows), size=(10000, len(rows)))
        for name in names:
            elapsed = np.array([row[name]["seconds"] for row in rows])
            gain = 100 * (1 - elapsed / baseline)
            quartiles = np.quantile(elapsed, (0.25, 0.75))
            result[case]["variants"][name] = {
                "median_s": float(np.median(elapsed)),
                "iqr_s": float(quartiles[1] - quartiles[0]),
                "speedup": float(np.median(baseline) / np.median(elapsed)),
                "gain_pct": float(np.median(gain)),
                "gain_ci": np.quantile(np.median(gain[resamples], axis=1), (0.025, 0.975)).tolist(),
                "equal": all(row[name]["digest"] == row["base"]["digest"] for row in rows),
            }
    return result


def main(args):
    root = Path(__file__).resolve().parents[2]
    patches = patches_from(args.patches)
    names = ["base", *args.variants]
    if args.current:
        names.append("current")
    with tempfile.TemporaryDirectory(prefix="rpcmci-ablation-") as temp:
        temp = Path(temp)
        archive = subprocess.check_output(["git", "archive", BASELINE, "tigramite"], cwd=root)
        base = temp / "base"
        base.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as git_archive:
            git_archive.extractall(base, filter="data")
        with contextlib.ExitStack() as workers:
            variants = {}
            for name in names:
                if name == "current":
                    directory = root
                elif name == "base":
                    directory = base
                else:
                    directory = temp / name
                    shutil.copytree(base, directory)
                    for patch_name in name.split("+"):
                        subprocess.run(["git", "apply"],
                                       input=patches[patch_name], cwd=directory, text=True, check=True)
                variant = Variant(directory)
                workers.callback(variant.close)
                variants[name] = variant

            checks = {}
            exact = True
            if not args.skip_checks:
                for case in ("lp", *CASES, "parallel", "missing", "short"):
                    reference = variants["base"].evaluate(case)
                    if case in CASES and not reference["successful"]:
                        raise RuntimeError(f"expected successful baseline in {case}")
                    if case == "short" and reference["successful"]:
                        raise RuntimeError("expected failed baseline in short")
                    checks[case] = {}
                    for name in names:
                        output = reference if name == "base" else variants[name].evaluate(case)
                        if case == "lp":
                            checks[case][name] = {
                                field: sum(a == b for a, b in zip(reference[field], output[field]))
                                for field in ("models", "outputs")
                            }
                            exact &= all(
                                len(reference[field]) == len(output[field]) == count == LP_CASE_COUNT
                                for field, count in checks[case][name].items()
                            )
                        else:
                            checks[case][name] = {
                                "equal": output["digest"] == reference["digest"],
                                "successful": output["successful"],
                            }
                            exact &= checks[case][name]["equal"]
                    print(case, json.dumps(checks[case]), file=sys.stderr, flush=True)

            records = {case: [] for case in CASES}
            if args.rounds and exact:
                for case in CASES:
                    # Warm the worker and numerical libraries before taking timings.
                    warm = [variants[name].evaluate(case) for name in names]
                    if any(not row["successful"] for row in warm):
                        raise RuntimeError(f"timing {case} requires successful runs")
                    for index in range(args.rounds):
                        order = names[index % len(names):] + names[:index % len(names)]
                        if (index // len(names)) % 2:
                            order.reverse()
                        records[case].append({name: variants[name].evaluate(case) for name in order})
                        print(f"{case} round {index + 1}/{args.rounds}", file=sys.stderr, flush=True)

            hits = {case: {name: variants[name].evaluate(f"hits:{case}")
                           for name in names if "4" in name}
                    for case in CASES} if any("4" in name for name in names) else {}

        timings = summary(records, names) if args.rounds and exact else {}
        exact &= all(variant["equal"] for case in timings.values()
                     for variant in case["variants"].values())
        output = {"baseline": BASELINE, "environment": environment_details(),
                  "exact": exact, "checks": checks, "hits": hits, "timings": timings}
        json.dump(output, sys.stdout, indent=2)
        print()
        if not exact:
            raise SystemExit("RPCMCI parity check failed")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--patches", type=Path, default=Path(__file__).with_name(PATCH_FILE))
        parser.add_argument("--variants", nargs="*", default=["1", "2", "3", "4ci", "4pred"])
        parser.add_argument("--rounds", type=int, default=10)
        parser.add_argument("--current", action="store_true")
        parser.add_argument("--skip-checks", action="store_true")
        options = parser.parse_args()
        if options.rounds < 0:
            parser.error("--rounds must be nonnegative")
        main(options)
