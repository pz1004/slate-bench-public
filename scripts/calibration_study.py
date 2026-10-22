"""Calibration of the audit's inference: coverage of known procedure-level contrasts.

The audit's intervals resample completed dataset-level differences with every fitted choice
held fixed, although each dataset also trains the procedures evaluated on the others. This
script measures, in simulation, how well they cover the contrast they target. It generates
complete score tables shaped like TabArena (the same 44 candidates, task mix and unit
counts, heterogeneous validation optimism, family-level dependence, fit-time budgets and
imputed rows), runs the audit's own nested code on them (scripts/score_level_policy_benchmark.py
with the 8-row pool of configs/score_level_tabarena_shrink.yaml and the reported
conventions), and records, for each replicate, the audit's estimate and three 95% intervals
for six contrasts: the dataset bootstrap, the paired t interval and the corrected resampled
t interval of scripts/corrected_resampled_test.py.

The target is theta = E[L(S) - L(R)] for procedures fitted on 38 training datasets with the
audit's task mix and evaluated on a fresh dataset, averaged over the artifact's task mix
(30/8/13). A held-out evaluation of one task is unbiased for that task's contrast, so the
task-weighted mean over a split's held-out datasets is unbiased for theta; theta is estimated
from separate truth runs: one outer split each, with 38 fresh training datasets and 51 fresh
held-out datasets, on seeds disjoint from the coverage replicates. Theta depends on the inner
fit through nested selection, so the bridge, which uses the shared inner fit, is scored
against truth runs with the shared inner fit on the same tables.

At 20 splits each dataset is held out about five times, so the weights J/(N n_d) of the
audit's corrected test (scripts/corrected_resampled_test.py), which make the mean of the split
statistics equal the dataset-level mean, vary several-fold and add noise of their own. The
replicates therefore also report the corrected test on split statistics with fixed task
weights; the audit's own version is calibrated at 100 splits on the bridge.

Stages (``--stage``): ``estimate`` fits the generator's parameters to TabArena's measured
rows; ``pilot`` checks the base scenario against TabArena's qualitative facts, checks that the
forced fast leave-one-dataset-out path gives identical rows, and times splits; ``truth``,
``truth_shared`` and ``replicates`` run the simulation in resumable parts; ``bridge`` reruns
base tables with the shared inner fit at 100 splits; ``summarize`` writes coverage with Wilson
intervals, widths, rejection rates and standard-error ratios with intervals over replicates,
and the labels of the adequacy rule of the config.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

import tabular_sota_benchmark.artifacts as artifacts  # noqa: E402
from tabular_sota_benchmark.artifacts import _candidate_rows_from_tabarena, _method_family, _method_variant  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, bootstrap_mean_p, holm  # noqa: E402
from tabular_sota_benchmark.ranking import set_rank_rule  # noqa: E402


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_module("score_level_policy_benchmark_calibration", "score_level_policy_benchmark.py")
review = _load_module("score_level_review_calibration", "score_level_review.py")
crt = _load_module("corrected_resampled_test_calibration", "corrected_resampled_test.py")
esc = _load_module("error_space_correction_calibration", "error_space_correction.py")
provref = _load_module("provenance_reference_calibration", "provenance_reference.py")

TASKS = {"binary": 30, "multiclass": 8, "regression": 13}
TRAIN_TASKS = {"binary": 22, "multiclass": 6, "regression": 10}
N_DATASETS = sum(TASKS.values())
# Dataset names end in the task's first three letters (generate_table).
TASK_SHARE = {task[:3]: count / N_DATASETS for task, count in TASKS.items()}
METRICS = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
SMALL_UNITS, LARGE_UNITS = 30, 9
NESTED = "nested_policy_selected"
VB = "meta_selector_single"
FP = "meta_selector_family_prior_budgeted"
TSF = "train_selected_fixed_reference__declared_fixed"
VB_DEFAULT = "reference_validation_best_default"
DTT = "reference_defaults_then_tuned"
CONTRASTS = [
    ("C1", NESTED, VB, "fitted"),
    ("C2", NESTED, FP, "fitted"),
    ("C3", NESTED, TSF, "fitted"),
    ("C4", FP, VB, "fitted"),
    ("C5", DTT, VB, "no fit"),
    ("C6", VB_DEFAULT, VB, "no fit"),
]
HOLM_SET = ["C1", "C2", "C3"]
# Interval methods and the point estimate each is centred on.
METHODS = ["boot", "t", "nb", "nbt"]
CENTRE = {"boot": "estimate", "t": "estimate", "nb": "nb_estimate", "nbt": "nbt_estimate"}


# ----------------------------------------------------------------------------- estimation


def _normal_scores(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby(["dataset", "fold", "seed"])[column]
    ranks = grouped.rank(method="average")
    size = grouped.transform("size").astype(float)
    return pd.Series(stats.norm.ppf((ranks - 0.5) / size), index=frame.index)


def _cell_means(values: pd.DataFrame, task: pd.Series, size: pd.Series) -> dict[str, float]:
    """Mean of a dataset-by-method table by task, dataset size and method, with fallbacks to
    the task and then to all datasets where a cell has no measured value."""
    out: dict[str, float] = {}
    for t in TASKS:
        for s in ["small", "large"]:
            rows = values.loc[[d for d in values.index if task[d] == t and size[d] == s]]
            task_rows = values.loc[[d for d in values.index if task[d] == t]]
            for m in values.columns:
                for candidate in (rows[m], task_rows[m], values[m]):
                    if candidate.notna().any():
                        out[f"{t}|{s}|{m}"] = float(candidate.mean())
                        break
    return out


def estimate_world(results: pd.DataFrame, metadata: pd.DataFrame, *, shrinkage: float = 0.05) -> dict[str, Any]:
    """Generator parameters from TabArena's measured rows (the imputed rows removed).

    A synthetic dataset's profile is the vector of every candidate's dataset-level mean test
    and validation normal score (88 values). Its mean is TabArena's by task, dataset size and
    candidate; its deviation is drawn from the covariance of TabArena's deviations, estimated
    from pairwise-complete datasets with each size class rescaled to a common spread, shrunk
    slightly toward its diagonal and made positive semidefinite. The covariance carries the
    dependence between candidates, between a candidate's test and validation scores (how far
    validation tracks test) and between optimism and quality. Unit-level noise is added on
    top, with a part shared by a family's variants.
    """
    frame = results.copy()
    frame["family"] = frame["method"].map(_method_family)
    frame["variant"] = frame["method"].map(_method_variant)
    frame["tuned"] = frame["variant"].ne("default").astype(float)
    frame["zt"] = _normal_scores(frame, "metric_error")
    frame["zv"] = _normal_scores(frame, "metric_error_val")
    units = frame.groupby("dataset")["fold"].nunique()
    size = units.map(lambda n: "small" if n >= SMALL_UNITS else "large")
    task = frame.groupby("dataset")["problem_type"].first()
    frame["size"] = frame["dataset"].map(size)
    methods = sorted(frame["method"].unique())
    family = pd.Series({m: _method_family(m) for m in methods})

    zt = frame.groupby(["dataset", "method"])["zt"].mean().unstack("method").reindex(columns=methods)
    zv = frame.groupby(["dataset", "method"])["zv"].mean().unstack("method").reindex(columns=methods)
    mu_t, mu_v = _cell_means(zt, task, size), _cell_means(zv, task, size)
    cell = lambda d, m, mu: mu[f"{task[d]}|{size[d]}|{m}"]  # noqa: E731
    dt = pd.DataFrame({m: [zt.loc[d, m] - cell(d, m, mu_t) for d in zt.index] for m in methods}, index=zt.index)
    dv = pd.DataFrame({m: [zv.loc[d, m] - cell(d, m, mu_v) for d in zv.index] for m in methods}, index=zv.index)
    profile = pd.concat([dt.add_prefix("t|"), dv.add_prefix("v|")], axis=1)
    spread = {s: float(np.sqrt(np.nanmean(profile.loc[size[size.eq(s)].index].to_numpy() ** 2))) for s in ["small", "large"]}
    pooled = float(np.sqrt(np.nanmean(profile.to_numpy() ** 2)))
    scale = {s: spread[s] / pooled for s in spread}
    standardized = profile.div(profile.index.map(lambda d: scale[size[d]]), axis=0)
    cov = standardized.cov(min_periods=10).to_numpy()
    cov = np.where(np.isfinite(cov), cov, 0.0)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 0.0, None)
    root = vectors * np.sqrt(values)

    frame = frame.join(
        frame.groupby(["dataset", "method"])[["zt", "zv"]].mean().rename(columns={"zt": "zt_mean", "zv": "zv_mean"}),
        on=["dataset", "method"],
    )
    frame["et"] = frame["zt"] - frame["zt_mean"]
    frame["ev"] = frame["zv"] - frame["zv_mean"]
    # Residuals from a dataset's own mean shrink the variance by (n - 1) / n.
    inflate = {"small": SMALL_UNITS / (SMALL_UNITS - 1), "large": LARGE_UNITS / (LARGE_UNITS - 1)}
    noise = {
        s: {
            "test_var": float(group["et"].var()) * inflate[s],
            "val_var": float(group["ev"].var()) * inflate[s],
            "rho": float(np.corrcoef(group["et"], group["ev"])[0, 1]),
        }
        for s, group in frame.groupby("size")
    }

    # Unit noise shared within a learner family (tuned and ensembled variants share folds).
    def family_share(column: str) -> float:
        wide = frame.pivot_table(index=["dataset", "fold", "seed"], columns="method", values=column)
        corr = wide.corr()
        pairs = [corr.loc[a, b] for a in corr.columns for b in corr.columns if a < b and family[a] == family[b]]
        return float(np.clip(np.nanmean(pairs), 0.0, 0.95))

    family_noise = {"test": family_share("et"), "val": family_share("ev")}

    lt = np.log(frame["time_train_s"].clip(lower=1e-3))
    frame["lt"] = lt
    default_rows = frame[frame["variant"].eq("default")]
    a_family = default_rows.groupby("family")["lt"].mean()
    zeta = (default_rows["lt"] - default_rows["family"].map(a_family)).groupby(default_rows["dataset"]).mean()
    tuned_rows = frame[frame["variant"].eq("tuned")]
    ratio = (
        tuned_rows.groupby(["family", "dataset"])["lt"].mean() - default_rows.groupby(["family", "dataset"])["lt"].mean()
    ).groupby("family").mean().dropna()  # families without a tuned variant have no ratio
    fit_resid = frame["lt"] - frame["family"].map(a_family) - frame["dataset"].map(zeta) - frame["tuned"] * frame["family"].map(ratio).fillna(0.0)
    lp = np.log(frame["time_infer_s"].clip(lower=1e-4))
    predict_base = (lp - frame["dataset"].map(zeta)).groupby(frame["method"]).mean()
    predict_sd = float((lp - frame["dataset"].map(zeta) - frame["method"].map(predict_base)).std())
    zeta_by_size = {s: {"mean": float(zeta[size.eq(s)].mean()), "sd": float(zeta[size.eq(s)].std())} for s in ["small", "large"]}

    unit_stats = results.groupby(["dataset", "fold", "seed"])["metric_error"].agg(
        median="median", iqr=lambda v: float(np.subtract(*np.percentile(v, [75, 25])))
    )
    raw = unit_stats.groupby("dataset").median()
    scale_pairs = {
        t: [[float(raw.loc[d, "median"]), float(raw.loc[d, "iqr"] / 1.349)] for d in task.index[task.eq(t)]]
        for t in TASKS
    }

    meta = metadata.set_index("dataset_name")
    flags = {}
    for name, column in [("TABPFNV2", "can_run_tabpfnv2"), ("TABICL", "can_run_tabicl")]:
        table = {}
        for t in TASKS:
            for s in ["small", "large"]:
                ds = [d for d in task.index if task[d] == t and size[d] == s]
                table[f"{t}|{s}"] = float(np.mean([not bool(meta.loc[d, column]) for d in ds])) if ds else 0.0
        flags[name] = table
    small_share = {t: float(size[task.eq(t)].eq("small").mean()) for t in TASKS}
    return {
        "methods": methods,
        "mu_test": mu_t,
        "mu_val": mu_v,
        "profile_root": root.tolist(),
        "profile_scale": scale,
        "shrinkage": shrinkage,
        "tuning_sd": 0.0,
        "noise": noise,
        "family_noise_share": family_noise,
        "fit": {
            "family_log_default": {f: float(v) for f, v in a_family.items()},
            "family_log_ratio": {f: float(v) for f, v in ratio.items()},
            "zeta": zeta_by_size,
            "resid_sd": float(fit_resid.std()),
            "predict_base": {m: float(v) for m, v in predict_base.items()},
            "predict_sd": predict_sd,
        },
        "scale_pairs": scale_pairs,
        "imputation": flags,
        "small_share": small_share,
    }


# ----------------------------------------------------------------------------- generation


def scenario_world(world: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Apply a scenario's knobs to the estimated world."""
    out = json.loads(json.dumps(world))
    if "small_share_factor" in scenario:
        out["small_share"] = {t: v * float(scenario["small_share_factor"]) for t, v in out["small_share"].items()}
    if "small_gap_factor" in scenario:
        # Amplify the mean validation optimism (test minus validation score) on small datasets.
        factor = float(scenario["small_gap_factor"])
        for key in list(out["mu_val"]):
            if "|small|" in key:
                t = out["mu_test"][key]
                out["mu_val"][key] = t - factor * (t - out["mu_val"][key])
    if "tuning_sd" in scenario:
        out["tuning_sd"] = float(scenario["tuning_sd"])
    if "family_noise_share" in scenario:
        out["family_noise_share"] = {k: float(scenario["family_noise_share"]) for k in out["family_noise_share"]}
    return out


def generate_table(world: dict[str, Any], rng: np.random.Generator, tasks: dict[str, int], prefix: str = "d") -> pd.DataFrame:
    """One synthetic results file in TabArena's columns."""
    methods = world["methods"]
    k = len(methods)
    family = np.array([_method_family(m) for m in methods])
    tuned = np.array([_method_variant(m) != "default" for m in methods], dtype=float)
    root = np.asarray(world["profile_root"], dtype=float)
    fit = world["fit"]
    rf_default = methods.index("RF (default)")
    families = sorted(set(family))
    column = np.array([families.index(f) for f in family])
    rows = []
    index = 0
    for task, count in tasks.items():
        for _ in range(int(count)):
            name = f"{prefix}{index:03d}_{task[:3]}"
            index += 1
            small = rng.random() < world["small_share"][task]
            size = "small" if small else "large"
            n_units = SMALL_UNITS if small else LARGE_UNITS
            profile = (root @ rng.normal(0.0, 1.0, root.shape[1])) * float(world["profile_scale"][size])
            mean_t = np.array([world["mu_test"][f"{task}|{size}|{m}"] for m in methods]) + profile[:k]
            mean_v = np.array([world["mu_val"][f"{task}|{size}|{m}"] for m in methods]) + profile[k:]
            if world["tuning_sd"] > 0:
                # A tuning effect on test that validation does not see.
                mean_t = mean_t + rng.normal(0.0, world["tuning_sd"]) * tuned
            noise = world["noise"][size]
            s_t, s_v, rho = math.sqrt(noise["test_var"]), math.sqrt(noise["val_var"]), float(noise["rho"])
            # Unit noise: a part shared by a family's variants and a part of each candidate,
            # with the test-validation correlation rho at both levels.
            share_t, share_v = world["family_noise_share"]["test"], world["family_noise_share"]["val"]
            shared_t = rng.normal(0.0, 1.0, (n_units, len(families)))[:, column]
            own_t = rng.normal(0.0, 1.0, (n_units, k))
            shared_v = rho * shared_t + math.sqrt(max(1 - rho**2, 0.0)) * rng.normal(0.0, 1.0, (n_units, len(families)))[:, column]
            own_v = rho * own_t + math.sqrt(max(1 - rho**2, 0.0)) * rng.normal(0.0, 1.0, (n_units, k))
            z_test = mean_t + s_t * (math.sqrt(share_t) * shared_t + math.sqrt(1 - share_t) * own_t)
            z_val = mean_v + s_v * (math.sqrt(share_v) * shared_v + math.sqrt(1 - share_v) * own_v)
            centre, spread = world["scale_pairs"][task][int(rng.integers(len(world["scale_pairs"][task])))]
            spread = max(spread, 1e-4)
            test = centre + spread * z_test
            val = centre + spread * z_val
            low = min(test.min(), val.min())
            if low <= 0:
                test, val = test - low + 0.01 * spread, val - low + 0.01 * spread
            zeta = rng.normal(fit["zeta"][size]["mean"], fit["zeta"][size]["sd"])
            log_fit = np.array([fit["family_log_default"][f] for f in family]) + zeta
            log_fit = log_fit + tuned * np.array([fit["family_log_ratio"].get(f, 0.0) for f in family])
            fit_seconds = np.exp(log_fit[None, :] + rng.normal(0.0, fit["resid_sd"], (n_units, k)))
            for j, m in enumerate(methods):
                if _method_variant(m) == "tuned + ensemble":
                    fit_seconds[:, j] = fit_seconds[:, methods.index(m.replace("tuned + ensemble", "tuned"))]
            predict = np.exp(np.array([fit["predict_base"][m] for m in methods])[None, :] + zeta + rng.normal(0.0, fit["predict_sd"], (n_units, k)))
            imputed = set()
            for model, table in world["imputation"].items():
                if rng.random() < table[f"{task}|{size}"]:
                    imputed |= {j for j, m in enumerate(methods) if _method_family(m) == model}
            for j in imputed:
                test[:, j], val[:, j], predict[:, j] = test[:, rf_default], val[:, rf_default], predict[:, rf_default]
                fit_seconds[:, j] = fit_seconds[:, rf_default] * (1.0 if _method_variant(methods[j]) == "default" else 201.0)
            for u in range(n_units):
                for j, m in enumerate(methods):
                    rows.append((name, u, 0, m, test[u, j], fit_seconds[u, j], predict[u, j], val[u, j], task, METRICS[task]))
    return pd.DataFrame(
        rows,
        columns=["dataset", "fold", "seed", "method", "metric_error", "time_train_s", "time_infer_s", "metric_error_val", "problem_type", "metric"],
    )


# ----------------------------------------------------------------------------- the audit


def force_fast_lodo() -> None:
    """Use the benchmark's exact fast leave-one-dataset-out path at every size (the default
    switches to it at 100 000 rows); ``--stage pilot`` checks that the rows are identical."""
    if getattr(artifacts._calibrated_selector, "_forced_fast", False):
        return
    patched = functools.partial(artifacts._calibrated_selector, fast_path_min_rows=0)
    patched._forced_fast = True  # type: ignore[attr-defined]
    artifacts._calibrated_selector = patched


def build_artifact(results: pd.DataFrame, selector: dict[str, Any], inner_source_fit: str) -> dict[str, Any]:
    candidates = _candidate_rows_from_tabarena(results)
    selector = {**selector, "inner_source_fit": inner_source_fit}
    derived, policy_methods, reference_methods = bench._selector_grid(selector, "tabarena_results")
    static = bench.build_static_selector_rows(
        candidates,
        runtime_weight=float(selector.get("runtime_weight", 0.0)),
        selector_method_name=str(selector.get("selector_method_name", VB)),
        derived_selectors=derived,
        ensemble_repository=None,
    )
    return {
        "name": "Synthetic",
        "kind": "tabarena_results",
        "source": {},
        "candidates": candidates,
        "repository": None,
        "selector": selector,
        "derived_selectors": derived,
        "policy_methods": policy_methods,
        "reference_methods": reference_methods,
        "dataset_descriptors": None,
        "static_selector_rows": static,
    }


def evaluate_split(artifact: dict[str, Any], split: tuple[int, list[str], list[str], str], pool: list[str]) -> dict[str, pd.DataFrame]:
    split_id, train, heldout, mode = split
    _, train_scores, _, units = bench._evaluate_artifact_split(
        artifact, split_id=split_id, train_datasets=train, heldout_datasets=heldout, split_mode=mode
    )
    held = units[units["phase"].eq("heldout")]
    per_dataset = held.groupby(["dataset", "method"], as_index=False).agg(
        regret=("normalized_rank_regret", "mean"), member=("selected_member", "first")
    )
    choice = train_scores[train_scores["method"].isin(pool)].assign(artifact="Synthetic", split_id=split_id)
    picked = review.train_selected_choices(choice, pool_name="declared_fixed", members=pool)
    reference = str(picked["selected_reference_method"].iloc[0])
    tsf = per_dataset[per_dataset["method"].eq(reference)].assign(method=TSF)
    nested_policy = str(held.loc[held["method"].eq(NESTED), "original_method"].iloc[0]) if "original_method" in held else ""
    per_dataset = pd.concat([per_dataset, tsf], ignore_index=True).assign(split_id=split_id)
    return {"per_dataset": per_dataset, "choice": pd.DataFrame([{"split_id": split_id, "nested": nested_policy, "reference": reference}])}


def no_fit_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset regrets of the rules that fit nothing, by the review's rank regret."""
    frame = esc.prepare(_candidate_rows_from_tabarena(results)[lambda f: f["method_type"].eq("candidate")])
    choices = provref.unit_choices(frame)
    return choices.groupby("dataset")[[DTT, VB, VB_DEFAULT]].mean()


# ----------------------------------------------------------------------------- inference


def task_weighted_split_means(per_split: pd.DataFrame, left: str, right: str) -> np.ndarray:
    """Per split, the held-out mean difference with each task weighted by its share of the
    artifact's datasets: an unbiased draw of theta, with no weights that depend on how often
    a dataset is held out."""
    frame = per_split[per_split["method"].isin([left, right])].pivot_table(
        index=["split_id", "dataset"], columns="method", values="regret"
    )
    diff = (frame[left] - frame[right]).dropna().rename("diff").reset_index()
    by_task = diff.assign(task=diff["dataset"].str[-3:]).groupby(["split_id", "task"])["diff"].mean().unstack("task")
    if by_task.isna().any().any() or set(by_task.columns) != set(TASK_SHARE):
        raise ValueError("A split does not hold out every task.")
    return (by_task * pd.Series(TASK_SHARE)[by_task.columns]).sum(axis=1).to_numpy(dtype=float)


def intervals(per_split: pd.DataFrame, left: str, right: str, *, n_train: int, n_test: int, resampler: ArtifactResampler) -> dict[str, float]:
    """The 95% intervals and tests of the mean for one contrast: the dataset bootstrap, the
    paired t interval, the audit's corrected resampled t interval and the corrected interval
    on split statistics with fixed task weights."""
    by_dataset = per_split.groupby(["dataset", "method"])["regret"].mean().unstack("method")
    diff = (by_dataset[left] - by_dataset[right]).dropna()
    values = resampler.align(diff)
    n = values.size
    mean = float(values.mean())
    boot = values[resampler.indices].mean(axis=1)
    b_low, b_high = np.quantile(boot, [0.025, 0.975])
    sd = float(values.std(ddof=1))
    se_t = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, n - 1))
    split_frame = per_split.rename(columns={"regret": "mean_normalized_rank_regret"})
    split = crt.split_means(split_frame[split_frame["method"].isin([left, right])])
    nb = crt.corrected_test((split[left] - split[right]).dropna().to_numpy(dtype=float), n_train=n_train, n_test=n_test, margins=[])
    j = int(nb["n_splits"])
    nb_crit = float(stats.t.ppf(0.975, j - 1))
    flat = task_weighted_split_means(per_split, left, right)
    nbt = crt.corrected_test(flat, n_train=n_train, n_test=n_test, margins=[])
    nbt_crit = float(stats.t.ppf(0.975, int(nbt["n_splits"]) - 1))
    zero = bool(np.all(np.abs(values) <= 1e-12))
    return {
        "estimate": mean,
        "boot_low": float(b_low), "boot_high": float(b_high), "boot_se": float(np.std(boot, ddof=1)),
        "boot_p": 1.0 if zero else float(bootstrap_mean_p(boot)),
        "t_low": mean - t_crit * se_t, "t_high": mean + t_crit * se_t, "t_se": se_t,
        "t_p": 1.0 if zero or se_t == 0 else float(2 * stats.t.sf(abs(mean) / se_t, n - 1)),
        "nb_estimate": float(nb["split_mean_difference"]),
        "nb_low": float(nb["split_mean_difference"] - nb_crit * nb["nb_se"]),
        "nb_high": float(nb["split_mean_difference"] + nb_crit * nb["nb_se"]),
        "nb_se": float(nb["nb_se"]), "nb_p": 1.0 if zero else float(nb["nb_p"]),
        "nbt_estimate": float(nbt["split_mean_difference"]),
        "nbt_low": float(nbt["split_mean_difference"] - nbt_crit * nbt["nb_se"]),
        "nbt_high": float(nbt["split_mean_difference"] + nbt_crit * nbt["nb_se"]),
        "nbt_se": float(nbt["nb_se"]), "nbt_p": 1.0 if zero else float(nbt["nb_p"]),
        "skew": float(stats.skew(values)) if sd > 0 else 0.0,
        "zero_share": float(np.mean(np.abs(values) <= 1e-12)),
        "degenerate": zero,
    }


# ----------------------------------------------------------------------------- tasks

_STATE: dict[str, Any] = {}


def _replicate_seed(config: dict[str, Any], scenario_index: int, kind: str, replicate: int) -> list[int]:
    # Bridge tables are the coverage replicates' own tables, rerun with more splits.
    # Shared-fit truth runs are the per-target truth runs' own tables, with the shared inner fit.
    kinds = {"replicate": 0, "bridge": 0, "truth": 1, "truth_shared": 1, "check": 2}
    return [int(config["seed"]), int(scenario_index), kinds[kind], int(replicate)]


def _scenario_list(config: dict[str, Any]) -> list[str]:
    return list(config["scenarios"].keys())


def run_block(task: dict[str, Any]) -> dict[str, Any]:
    """One block of outer splits of one replicate (or one truth run)."""
    config, world = _STATE["config"], _STATE["world"]
    selector, pool = _STATE["selector"], _STATE["pool"]
    set_rank_rule("consistent")
    if task.get("fast", True):
        force_fast_lodo()
    scenario = task["scenario"]
    s_index = _scenario_list(config).index(scenario)
    world_s = scenario_world(world, config["scenarios"][scenario] or {})
    rng = np.random.default_rng(_replicate_seed(config, s_index, task["kind"], task["replicate"]))
    started = time.time()
    if task["kind"] in ("truth", "truth_shared"):
        train_table = generate_table(world_s, rng, TRAIN_TASKS, prefix="t")
        held_table = generate_table(world_s, rng, TASKS, prefix="h")
        results = pd.concat([train_table, held_table], ignore_index=True)
        artifact = build_artifact(results, selector, task["inner"])
        split = (0, sorted(train_table["dataset"].unique()), sorted(held_table["dataset"].unique()), "truth")
        out = evaluate_split(artifact, split, pool)
        nofit = no_fit_rows(held_table)
        per_dataset = out["per_dataset"]
        by_dataset = per_dataset.pivot_table(index="dataset", columns="method", values="regret")
        for column in [DTT, VB_DEFAULT]:
            by_dataset[column] = nofit[column].reindex(by_dataset.index)
        gap = float((by_dataset[VB] - nofit[VB].reindex(by_dataset.index)).abs().max())
        record = {name: float((by_dataset[left] - by_dataset[right]).mean()) for name, left, right, _ in CONTRASTS}
        record.update({"scenario": scenario, "replicate": task["replicate"], "nofit_vb_gap": gap, "seconds": time.time() - started})
        return {"kind": "truth", "record": record}
    table = generate_table(world_s, rng, TASKS)
    artifact = build_artifact(table, selector, task["inner"])
    splits = bench._split_datasets(
        artifact["candidates"], n_splits=int(task["n_splits"]), heldout_fraction=0.25,
        seed=int(config["seed"]) + 1000 * s_index + int(task["replicate"]),
    )
    frames, choices = [], []
    for split in splits:
        if split[0] not in set(task["split_ids"]):
            continue
        out = evaluate_split(artifact, split, pool)
        frames.append(out["per_dataset"])
        choices.append(out["choice"])
    nofit = None
    if task["split_ids"][0] == 0:
        nofit = no_fit_rows(table).reset_index()
    return {
        "kind": task["kind"],
        "key": task["key"],
        "per_dataset": pd.concat(frames, ignore_index=True),
        "choices": pd.concat(choices, ignore_index=True),
        "nofit": nofit,
        "seconds": time.time() - started,
    }


def _part_path(output_dir: Path, task: dict[str, Any]) -> Path:
    return output_dir / "parts" / task["kind"] / f"{task['key']}.parquet"


def run_tasks(config: dict[str, Any], world: dict[str, Any], tasks: list[dict[str, Any]], jobs: int, output_dir: Path) -> None:
    global _STATE
    pending = [task for task in tasks if not _part_path(output_dir, task).exists()]
    print(f"{len(tasks) - len(pending)} done, {len(pending)} pending", flush=True)
    if not pending:
        return
    bench_config = yaml.safe_load(Path(config["benchmark_config"]).read_text(encoding="utf-8"))
    review_config = yaml.safe_load(Path(config["review_config"]).read_text(encoding="utf-8"))
    _STATE = {
        "config": config,
        "world": world,
        "selector": dict(bench_config["artifacts"][0]["selector"]),
        "pool": [str(m) for m in review_config["fixed_reference_pools"]["declared_fixed"]],
    }
    context = multiprocessing.get_context("fork")
    started = time.time()
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=context) as pool:
        futures = {pool.submit(run_block, task): task for task in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            result = future.result()
            path = _part_path(output_dir, task)
            path.parent.mkdir(parents=True, exist_ok=True)
            if result["kind"] == "truth":
                pd.DataFrame([result["record"]]).to_parquet(path, index=False)
            else:
                frame = result["per_dataset"].assign(block_seconds=result["seconds"])
                frame.to_parquet(path, index=False)
                result["choices"].to_parquet(path.with_suffix(".choices.parquet"), index=False)
                if result["nofit"] is not None:
                    result["nofit"].to_parquet(path.with_suffix(".nofit.parquet"), index=False)
            if done % 10 == 0 or done == len(pending):
                elapsed = time.time() - started
                print(f"{done}/{len(pending)} blocks, {elapsed / 3600:.2f} h, about {elapsed / done * (len(pending) - done) / 3600:.2f} h left", flush=True)


def replicate_tasks(config: dict[str, Any], *, kind: str, scenarios: dict[str, int], n_splits: int, block: int, inner: str, fast: bool) -> list[dict[str, Any]]:
    tasks = []
    for scenario, count in scenarios.items():
        for replicate in range(int(count)):
            for start in range(0, n_splits, block):
                ids = list(range(start, min(start + block, n_splits)))
                tasks.append(
                    {
                        "kind": kind, "scenario": scenario, "replicate": replicate, "n_splits": n_splits,
                        "split_ids": ids, "inner": inner, "fast": fast,
                        "key": f"{scenario}_r{replicate:03d}_s{ids[0]:03d}",
                    }
                )
    return tasks


def truth_tasks(config: dict[str, Any], kind: str = "truth") -> list[dict[str, Any]]:
    runs, inner = (config["truth_runs"], "per_target") if kind == "truth" else (config["truth_shared_runs"], "shared")
    return [
        {"kind": kind, "scenario": scenario, "replicate": r, "inner": inner, "fast": True, "key": f"{scenario}_t{r:03d}"}
        for scenario, count in runs.items()
        for r in range(int(count))
    ]


# ----------------------------------------------------------------------------- summary


def assemble(output_dir: Path, kind: str, n_splits: int, *, first_splits: int | None = None) -> pd.DataFrame:
    """Per replicate and contrast: the estimate and the three intervals (on the first
    ``first_splits`` splits only, if given)."""
    parts = sorted((output_dir / "parts" / kind).glob("*_s000.parquet"))
    rows = []
    for first in parts:
        stem = first.name[: -len("_s000.parquet")]
        blocks = sorted((output_dir / "parts" / kind).glob(f"{stem}_s[0-9][0-9][0-9].parquet"))
        per_split = pd.concat([pd.read_parquet(path) for path in blocks], ignore_index=True)
        if per_split["split_id"].nunique() < n_splits:
            continue
        if first_splits is not None:
            per_split = per_split[per_split["split_id"] < int(first_splits)]
        nofit = pd.read_parquet(first.with_suffix(".nofit.parquet")).set_index("dataset")
        vb_gap = float(
            (per_split[per_split["method"].eq(VB)].groupby("dataset")["regret"].mean() - nofit[VB]).abs().max()
        )
        held = per_split[["split_id", "dataset"]].drop_duplicates()
        extra = [held.assign(method=column, regret=held["dataset"].map(nofit[column])) for column in [DTT, VB_DEFAULT]]
        per_split = pd.concat([per_split, *extra], ignore_index=True)
        choices = pd.concat(
            [pd.read_parquet(path.with_suffix(".choices.parquet")) for path in blocks], ignore_index=True
        )
        if first_splits is not None:
            choices = choices[choices["split_id"] < int(first_splits)]
        datasets = sorted(per_split["dataset"].unique())
        n_test = int(held.groupby("split_id")["dataset"].nunique().iloc[0])
        # A dataset that no split holds out (possible at 20 splits) drops out of the estimate;
        # the training and held-out sizes of the corrected test stay those of the design.
        resampler = ArtifactResampler(datasets, seed=20260608, n_resamples=10_000)
        scenario, rep = stem.rsplit("_r", 1)
        shares = choices["nested"].value_counts(normalize=True)
        for name, left, right, family in CONTRASTS:
            result = intervals(per_split, left, right, n_train=N_DATASETS - n_test, n_test=n_test, resampler=resampler)
            rows.append(
                {
                    "scenario": scenario, "replicate": int(rep), "contrast": name, "kind": family, **result,
                    "n_datasets_heldout": len(datasets),
                    "vb_reproduction_gap": vb_gap, "nested_modal_share": float(shares.iloc[0]),
                    "nested_distinct": int(choices["nested"].nunique()),
                    "reference_modal_share": float(choices["reference"].value_counts(normalize=True).iloc[0]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Holm adjustment within the three contrasts of nested selection, as in the audit's family.
    for method in METHODS:
        table[f"{method}_holm_p"] = np.nan
        for _, group in table[table["contrast"].isin(HOLM_SET)].groupby(["scenario", "replicate"]):
            table.loc[group.index, f"{method}_holm_p"] = holm(group[f"{method}_p"].tolist())
    return table


def wilson(covered: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The 95% Wilson interval of a proportion."""
    p = covered / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def detectable_coverage(total: int) -> float:
    """The largest coverage that R replicates label anti-conservative (Wilson upper end below 0.925)."""
    hits = [k for k in range(total + 1) if wilson(k, total)[1] < 0.925]
    return max(hits) / total if hits else float("nan")


def coverage_table(replicates: pd.DataFrame, truth: pd.DataFrame, label: str) -> pd.DataFrame:
    theta = truth.groupby("scenario")[[c for c, *_ in CONTRASTS]].agg(["mean", "std", "count"])
    rng = np.random.default_rng(20260626)
    rows = []
    for (scenario, contrast), group in replicates.groupby(["scenario", "contrast"]):
        mean, sd, count = (float(theta.loc[scenario, (contrast, s)]) for s in ("mean", "std", "count"))
        theta_se = sd / math.sqrt(count)
        r = len(group)
        draws = rng.integers(0, r, size=(2000, r))
        for method in METHODS:
            centre = group[CENTRE[method]].to_numpy(dtype=float)
            se = group[f"{method}_se"].to_numpy(dtype=float)
            empirical_sd = float(np.std(centre, ddof=1))
            covered = (group[f"{method}_low"] <= mean) & (mean <= group[f"{method}_high"])
            k = int(covered.sum())
            p = k / r
            low, high = wilson(k, r)
            spread = np.std(centre[draws], axis=1, ddof=1)
            ratios = np.where(spread > 0, se[draws].mean(axis=1) / np.where(spread > 0, spread, 1.0), np.nan)
            rejects = group[f"{method}_p"] < 0.05
            holm_rejects = group[f"{method}_holm_p"] < 0.05
            rows.append(
                {
                    "design": label, "scenario": scenario, "contrast": contrast, "kind": group["kind"].iloc[0], "method": method,
                    "replicates": r, "theta": mean, "theta_mc_se": theta_se, "mean_estimate": float(centre.mean()),
                    "bias": float(centre.mean()) - mean, "empirical_sd": empirical_sd,
                    "theta_se_over_sd": theta_se / empirical_sd if empirical_sd > 0 else float("nan"),
                    "coverage": p, "coverage_low": low, "coverage_high": high,
                    "coverage_mc_se": math.sqrt(max(p * (1 - p), 1e-12) / r),
                    "miss_low": float((group[f"{method}_high"] < mean).mean()), "miss_high": float((group[f"{method}_low"] > mean).mean()),
                    "mean_width": float((group[f"{method}_high"] - group[f"{method}_low"]).mean()),
                    "se_ratio": float(se.mean()) / empirical_sd if empirical_sd > 0 else float("nan"),
                    "se_ratio_low": float(np.nanpercentile(ratios, 2.5)), "se_ratio_high": float(np.nanpercentile(ratios, 97.5)),
                    "rms_se_ratio": float(np.sqrt((se**2).mean())) / empirical_sd if empirical_sd > 0 else float("nan"),
                    "reject_rate": float(rejects.mean()),
                    "reject_rate_holm": float(holm_rejects.mean()) if group[f"{method}_holm_p"].notna().any() else float("nan"),
                    "degenerate_share": float(group["degenerate"].mean()),
                    "mean_skew": float(group["skew"].mean()), "nested_modal_share": float(group["nested_modal_share"].mean()),
                    "min_datasets_heldout": int(group["n_datasets_heldout"].min()),
                }
            )
    table = pd.DataFrame(rows)
    # The adequacy rule of configs/calibration_study.yaml, applied in this order.
    se_ok = table["se_ratio"] >= 0.9
    table["verdict"] = np.select(
        [table["coverage_high"] < 0.925, table["coverage_low"] > 0.975, se_ok & (table["coverage"] >= 0.925)],
        ["anti-conservative", "conservative", "adequate"],
        default="inconclusive",
    )
    # The two earlier wordings (adequacy_rule_history), reported alongside; both used the
    # Wald interval of the coverage.
    wald_upper = table["coverage"] + 1.96 * table["coverage_mc_se"]
    table["verdict_first"] = np.select([wald_upper < 0.925, se_ok], ["anti-conservative", "adequate"], default="inconclusive")
    table["verdict_second"] = np.select(
        [wald_upper < 0.925, se_ok & table["coverage"].between(0.925, 0.975), se_ok & (table["coverage"] > 0.975)],
        ["anti-conservative", "adequate", "conservative"],
        default="inconclusive",
    )
    return table


# ----------------------------------------------------------------------------- stages


def stage_estimate(config: dict[str, Any], output_dir: Path) -> None:
    results = pd.read_csv(config["source_results"])
    metadata = pd.read_csv(config["source_metadata"])
    world = estimate_world(results, metadata)
    (output_dir / "world.json").write_text(json.dumps(world, indent=2), encoding="utf-8")
    print(json.dumps({k: world[k] for k in ["profile_scale", "shrinkage", "noise", "family_noise_share", "small_share"]}, indent=1))


def table_facts(results: pd.DataFrame) -> dict[str, float]:
    """The qualitative TabArena facts a world should reproduce, from one results table."""
    frame = esc.prepare(_candidate_rows_from_tabarena(results))
    choices = provref.unit_choices(frame)
    units = results.groupby("dataset")["fold"].nunique()
    per_dataset = choices.groupby("dataset")[[VB, VB_DEFAULT, DTT]].mean()
    small = per_dataset.index.map(lambda d: units[d] >= SMALL_UNITS)
    keyed = results.assign(variant=results["method"].map(_method_variant))
    unit = ["dataset", "fold", "seed"]
    best = keyed.loc[keyed.groupby(unit)["metric_error"].idxmin(), "variant"].value_counts(normalize=True)
    score = keyed["metric_error_val"] + 1e-6 * np.log1p(keyed["time_train_s"].clip(lower=0))
    picked = keyed.loc[score.groupby([keyed[c] for c in unit]).idxmin(), "variant"]
    return {
        "vb_regret": float(per_dataset[VB].mean()),
        "vb_regret_large": float(per_dataset.loc[~small, VB].mean()),
        "vb_regret_small": float(per_dataset.loc[small, VB].mean()),
        "vb_default_regret": float(per_dataset[VB_DEFAULT].mean()),
        "defaults_rule_regret": float(per_dataset[DTT].mean()),
        "vb_tuned_ensemble_share": float(picked.eq("tuned + ensemble").mean()),
        "test_best_default": float(best.get("default", 0.0)),
        "test_best_tuned": float(best.get("tuned", 0.0)),
        "test_best_tuned_ensemble": float(best.get("tuned + ensemble", 0.0)),
    }


def stage_facts(config: dict[str, Any], world: dict[str, Any], output_dir: Path) -> None:
    """TabArena's facts against the mean over 10 tables of each scenario."""
    set_rank_rule("consistent")
    rows = [{"source": "TabArena (measured rows)", **table_facts(pd.read_csv(config["source_results"]))}]
    for s_index, scenario in enumerate(_scenario_list(config)):
        world_s = scenario_world(world, config["scenarios"][scenario] or {})
        facts = [table_facts(generate_table(world_s, np.random.default_rng([int(config["seed"]), s_index, 3, r]), TASKS)) for r in range(10)]
        rows.append({"source": scenario, **pd.DataFrame(facts).mean().to_dict()})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "world_facts.csv", index=False)
    print(table.round(3).to_string(index=False))


def stage_pilot(config: dict[str, Any], world: dict[str, Any], output_dir: Path, jobs: int) -> None:
    """Base-scenario facts on 3 tables, fast-path identity, and the time per split."""
    bench_config = yaml.safe_load(Path(config["benchmark_config"]).read_text(encoding="utf-8"))
    review_config = yaml.safe_load(Path(config["review_config"]).read_text(encoding="utf-8"))
    selector = dict(bench_config["artifacts"][0]["selector"])
    pool = [str(m) for m in review_config["fixed_reference_pools"]["declared_fixed"]]
    set_rank_rule("consistent")
    rng = np.random.default_rng([int(config["seed"]), 99, 0])
    report: dict[str, Any] = {}
    table = generate_table(world, rng, TASKS)
    frame = esc.prepare(_candidate_rows_from_tabarena(table))
    choices = provref.unit_choices(frame)
    vb_class = choices["deployed_variant"]
    best = frame.loc[frame.groupby("unit")["primary_metric"].idxmin()]
    report["test_best_variant_share"] = best["method"].map(_method_variant).value_counts(normalize=True).round(3).to_dict()
    vb_pick = frame.loc[(frame["val_primary_metric"] + frame["tie"]).groupby(frame["unit"]).idxmin()]
    report["vb_tuned_ensemble_share"] = float(vb_pick["method"].map(_method_variant).eq("tuned + ensemble").mean())
    frame = frame.assign(variant=frame["method"].map(_method_variant))
    ranked = frame.assign(
        vr=frame.groupby("unit")["val_primary_metric"].rank(method="average"),
        tr=frame.groupby("unit")["primary_metric"].rank(method="average"),
    )
    ranked["gap"] = (ranked["tr"] - ranked["vr"]) / 43.0
    report["rank_gap_by_variant"] = ranked.groupby("variant")["gap"].mean().round(3).to_dict()
    report["vb_minus_dtt"] = float(choices.groupby("dataset")[VB].mean().mean() - choices.groupby("dataset")[DTT].mean().mean())
    del vb_class
    fits = table.assign(variant=table["method"].map(_method_variant)).groupby("variant")["time_train_s"].median().round(1).to_dict()
    report["median_fit_seconds"] = fits

    # Fast-path identity on two splits of one table.
    artifact = build_artifact(table, selector, "per_target")
    splits = bench._split_datasets(artifact["candidates"], n_splits=2, heldout_fraction=0.25, seed=1)
    timing = {}
    started = time.time()
    normal = [evaluate_split(artifact, split, pool)["per_dataset"] for split in splits]
    timing["normal_seconds_per_split"] = (time.time() - started) / len(splits)
    force_fast_lodo()
    started = time.time()
    fast = [evaluate_split(artifact, split, pool)["per_dataset"] for split in splits]
    timing["fast_seconds_per_split"] = (time.time() - started) / len(splits)
    for a, b in zip(normal, fast):
        merged = a.merge(b, on=["split_id", "dataset", "method"], suffixes=("_n", "_f"))
        if len(merged) != len(a) or not np.allclose(merged["regret_n"], merged["regret_f"], rtol=0, atol=1e-12) or not (merged["member_n"] == merged["member_f"]).all():
            raise ValueError("The forced fast path does not reproduce the normal path.")
    report["fast_path_identical"] = True
    report["timing"] = timing
    (output_dir / "pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/calibration_study.yaml"))
    parser.add_argument(
        "--stage", required=True, choices=["estimate", "facts", "pilot", "truth", "truth_shared", "replicates", "bridge", "summarize", "check"]
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "estimate":
        stage_estimate(config, output_dir)
        return
    world = json.loads((output_dir / "world.json").read_text(encoding="utf-8"))
    if args.stage == "facts":
        stage_facts(config, world, output_dir)
    elif args.stage == "pilot":
        stage_pilot(config, world, output_dir, args.jobs)
    elif args.stage == "check":
        tasks = replicate_tasks(config, kind="check", scenarios={"base": int(config["check"]["replicates"])},
                                n_splits=int(config["check"]["splits"]), block=int(config["block"]), inner="per_target", fast=True)
        run_tasks(config, world, tasks, args.jobs, output_dir)
        table = assemble(output_dir, "check", int(config["check"]["splits"]))
        table.to_csv(output_dir / "check_replicates.csv", index=False)
        print(table.groupby("contrast")[["estimate", "boot_se", "nb_se", "skew", "zero_share", "nested_modal_share", "reference_modal_share", "vb_reproduction_gap"]].mean().round(4).to_string())
    elif args.stage in ("truth", "truth_shared"):
        run_tasks(config, world, truth_tasks(config, args.stage), args.jobs, output_dir)
    elif args.stage == "replicates":
        tasks = replicate_tasks(config, kind="replicate", scenarios=config["replicates"], n_splits=int(config["splits"]),
                                block=int(config["block"]), inner="per_target", fast=True)
        run_tasks(config, world, tasks, args.jobs, output_dir)
    elif args.stage == "bridge":
        bridge = config["bridge"]
        tasks = replicate_tasks(config, kind="bridge", scenarios={"base": int(bridge["tables"])}, n_splits=int(bridge["splits"]),
                                block=int(bridge["block"]), inner=str(bridge["inner_source_fit"]), fast=True)
        run_tasks(config, world, tasks, args.jobs, output_dir)
    elif args.stage == "summarize":
        def read_truth(kind: str) -> pd.DataFrame:
            return pd.concat([pd.read_parquet(p) for p in sorted((output_dir / "parts" / kind).glob("*.parquet"))], ignore_index=True)

        truth = read_truth("truth")
        truth.to_csv(output_dir / "truth_runs.csv", index=False)
        tables = []
        replicates = assemble(output_dir, "replicate", int(config["splits"]))
        replicates.to_csv(output_dir / "replicate_contrasts.csv", index=False)
        tables.append(coverage_table(replicates, truth, f"per-target, J={config['splits']}"))
        extra: dict[str, Any] = {}
        bridge_dir = output_dir / "parts" / "bridge"
        if bridge_dir.exists():
            # The bridge uses the shared inner fit, so it is scored against shared-fit truth runs
            # on the per-target truth runs' own tables. Rules that never read the inner fit must
            # give the same truth on both.
            shared = read_truth("truth_shared")
            shared.to_csv(output_dir / "truth_runs_shared.csv", index=False)
            paired = truth.merge(shared, on=["scenario", "replicate"], suffixes=("", "_shared"))
            unaffected = [name for name, *_ in CONTRASTS if name not in HOLM_SET]
            gap = float(max((paired[c] - paired[f"{c}_shared"]).abs().max() for c in unaffected))
            if gap > 1e-12:
                raise SystemExit(f"Shared-fit truth runs change contrasts that do not read the inner fit (gap {gap:.3g}).")
            extra["truth_shared_runs"] = shared.groupby("scenario").size().to_dict()
            extra["max_unaffected_truth_gap"] = gap
            extra["inner_fit_theta_shift"] = {c: float((paired[f"{c}_shared"] - paired[c]).mean()) for c in HOLM_SET}
            bridge = assemble(output_dir, "bridge", int(config["bridge"]["splits"]))
            bridge.to_csv(output_dir / "bridge_contrasts.csv", index=False)
            tables.append(coverage_table(bridge, shared, f"shared, J={config['bridge']['splits']}"))
            short = assemble(output_dir, "bridge", int(config["bridge"]["splits"]), first_splits=int(config["splits"]))
            short.to_csv(output_dir / "bridge_contrasts_short.csv", index=False)
            tables.append(coverage_table(short, shared, f"shared, J={config['splits']}"))
            same = replicates[replicates["scenario"].eq("base") & replicates["replicate"].isin(short["replicate"].unique())]
            tables.append(coverage_table(same, truth, f"per-target, J={config['splits']}, bridge tables"))
        coverage = pd.concat(tables, ignore_index=True)
        coverage.to_csv(output_dir / "coverage.csv", index=False)
        dropped = replicates.drop_duplicates(["scenario", "replicate"])
        (output_dir / "metadata.json").write_text(
            json.dumps({"config": str(args.config), "generator": "world.json", "adequacy_rule": config["adequacy_rule"],
                        "adequacy_rule_history": config.get("adequacy_rule_history", []),
                        "detectable_coverage": {str(r): detectable_coverage(r) for r in sorted(coverage["replicates"].unique().tolist())},
                        "truth_runs": truth.groupby("scenario").size().to_dict(),
                        "replicates": replicates.groupby("scenario")["replicate"].nunique().to_dict(),
                        "replicates_with_a_dataset_never_held_out": int((dropped["n_datasets_heldout"] < N_DATASETS).sum()),
                        "max_nofit_vb_gap": float(max(truth["nofit_vb_gap"].max(), replicates["vb_reproduction_gap"].max())),
                        **extra, "command": sys.argv}, indent=2),
            encoding="utf-8",
        )
        shown = ["design", "scenario", "contrast", "method", "theta", "coverage", "coverage_low", "coverage_high", "se_ratio", "reject_rate", "verdict"]
        print(coverage[shown].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
