"""Collect measured facts for report prose from saved run artifacts.

Everything here is read from files written by training/analysis. Nothing is
refitted, and absent artifacts (older runs) simply leave facts empty so the
prose layer can omit the corresponding statements rather than invent them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOSSES = ("mae", "mse", "rmse", "brier")
ALIASES = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}

# Families support one hedged structural reading of the model comparison.
FAMILIES = {
    "ModelLRRegressor": "linear", "ModelLRClassifier": "linear",
    "ModelSVRRegressor": "kernel", "ModelSVCClassifier": "kernel",
    "ModelGPRRegressor": "kernel",
    "ModelKNeighborsRegressor": "neighbors", "ModelKNeighborsClassifier": "neighbors",
    "ModelDecisionTreesClassifier": "tree",
    "ModelRandomForestRegressor": "bagging", "ModelRandomForestClassifier": "bagging",
    "ModelExtraTreeRegressor": "bagging", "ModelBaggingRegressor": "bagging",
    "ModelGradientBoostingRegressor": "boosting", "ModelLGBMRegressor": "boosting",
    "ModelLGBMClassifier": "boosting", "ModelXGBoostRegressor": "boosting",
    "ModelXGBoostClassifier": "boosting", "ModelCatBoostRegressor": "boosting",
    "ModelAdaBoostRegressor": "boosting", "ModelAdaBoostClassifier": "boosting",
    "ModelNaiveBayesClassifier": "probabilistic",
}
NONLINEAR = {"kernel", "neighbors", "tree", "bagging", "boosting"}

# Thresholds for flagging data issues; documented in references/pipeline.md.
SHIFT_FLAG_SD = 0.5
MISSING_FLAG = 0.10
RANGE_FLAG = 0.10
CORRELATION_FLAG = 0.7


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def num(value):
    return float(value) if finite(value) else None


def read_csv(path, **kwargs):
    path = Path(path)
    return pd.read_csv(path, **kwargs) if path.is_file() else None


def load_context(directory):
    """Optional researcher-supplied report context (report-context.json)."""
    path = Path(directory) / "report-context.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_context(data)


def validate_context(data):
    if not isinstance(data, dict):
        raise ValueError("report context must be a JSON object")
    allowed = {"title", "question", "background", "target_label", "target_unit", "tolerance",
               "feature_labels", "class_labels", "population", "notes"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown report context keys: {sorted(unknown)}; allowed: {sorted(allowed)}")
    for key in ("feature_labels", "class_labels"):
        if key in data and not (isinstance(data[key], dict) and
                                all(isinstance(v, str) for v in data[key].values())):
            raise ValueError(f"report context {key} must map names to strings")
    if "tolerance" in data and not (finite(data["tolerance"]) and float(data["tolerance"]) > 0):
        raise ValueError("report context tolerance must be a positive number in target units")
    for key in ("title", "question", "background", "target_label", "target_unit", "population", "notes"):
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"report context {key} must be a string")
    return data


def oriented_gain(metric, model, baseline):
    if model is None or baseline is None:
        return None
    return baseline - model if metric in LOSSES else model - baseline


def relative_change(metric, model, baseline):
    """Fractional improvement over a reference (positive = better); None if undefined."""
    if model is None or baseline is None or abs(baseline) < 1e-12:
        return None
    if metric in LOSSES:
        return (baseline - model) / abs(baseline)
    return None


def comparison_facts(directory, s):
    cv = read_csv(directory / "cv-scores.csv")
    base_cv = read_csv(directory / "baseline-cv.csv")
    oof = read_csv(directory / "oof-predictions.csv", dtype={"sample_id": str, "observed": str, "predicted": str})
    sign = -1 if s["direction"] == "lower" else 1
    base_by_fold = (dict(zip(base_cv.fold.astype(int), base_cv.score.astype(float)))
                    if base_cv is not None and {"fold", "score"} <= set(base_cv) else {})
    rows = []
    for item in s.get("comparison") or []:
        tag = item["model"]
        folds = cv[cv.model == tag].sort_values("fold") if cv is not None else pd.DataFrame()
        row = {"model": tag, "family": FAMILIES.get(tag, "other"), "mean": num(item["mean"]),
               "sd": num(item.get("sd")), "folds": int(item.get("folds") or len(folds)),
               "fold_scores": [], "fold_table": []}
        if len(folds):
            row["fold_scores"] = folds.score.astype(float).tolist()
            row["min"], row["max"] = float(folds.score.min()), float(folds.score.max())
            if "train_score" in folds and folds.train_score.notna().all():
                row["train_mean"] = float(folds.train_score.mean())
                row["train_gap"] = float(sign * (folds.train_score - folds.score).mean())
            gains = [sign * (float(r.score) - base_by_fold[int(r.fold)])
                     for r in folds.itertuples() if int(r.fold) in base_by_fold]
            if gains and len(gains) == len(folds):
                row["baseline_gain_mean"] = float(np.mean(gains))
                row["baseline_gain_positive_folds"] = int(sum(g > 0 for g in gains))
            secondary = {}
            for column in folds.columns:
                if column.startswith("validation_") and column != "validation_n":
                    values = pd.to_numeric(folds[column], errors="coerce")
                    if values.notna().all():
                        secondary[column.removeprefix("validation_")] = float(values.mean())
            row["secondary"] = secondary
            for r in folds.itertuples():
                row["fold_table"].append({
                    "fold": int(r.fold), "score": float(r.score),
                    "train_score": num(getattr(r, "train_score", None)),
                    "baseline": base_by_fold.get(int(r.fold)),
                    "train_n": int(r.train_n) if finite(getattr(r, "train_n", None)) else None,
                    "validation_n": int(r.validation_n) if finite(getattr(r, "validation_n", None)) else None})
        if oof is not None and "model" in oof:
            part = oof[oof.model == tag]
            if len(part):
                row["oof_n"] = int(len(part))
        rows.append(row)
    selected = next((r for r in rows if r["model"] == s.get("selected_model")), None)
    for row in rows:
        row["standing"] = standing(row, selected, rows, s)
        if selected is not None and row is not selected and row["mean"] is not None:
            margin = sign * (selected["mean"] - row["mean"])
            row["margin"] = margin
            pooled = pooled_sd(selected, row)
            row["margin_sd"] = margin / pooled if pooled else None
            if s["metric"] in LOSSES and abs(row["mean"]) > 1e-12:
                # Selected loss relative to this candidate's loss: "x% lower error".
                row["relative_margin"] = margin / abs(row["mean"])
    return rows, selected


def pooled_sd(a, b):
    values = [v for v in (a.get("sd"), b.get("sd")) if finite(v)]
    if not values:
        return None
    value = math.sqrt(sum(v * v for v in values) / len(values))
    return value if value > 1e-12 else None


def standing(row, selected, rows, s):
    if len(rows) <= 1:
        return "sole"
    if selected is None:
        return "unranked"
    if row is selected:
        return "selected"
    sign = -1 if s["direction"] == "lower" else 1
    margin = sign * (selected["mean"] - row["mean"])
    pooled = pooled_sd(selected, row)
    # Measured in between-fold SD units so the label tracks validation noise.
    if pooled is None:
        return "near" if abs(margin) < 1e-12 else "weaker"
    ratio = margin / pooled
    if ratio <= 1:
        return "near"
    if ratio <= 3:
        return "weaker"
    return "clearly_weaker"


def test_facts(directory, s):
    intervals = read_csv(directory / "metric-intervals.csv")
    rows = {}
    if intervals is not None:
        for r in intervals.astype(object).where(pd.notna(intervals), None).to_dict("records"):
            rows[r["metric"]] = {k: (num(v) if k != "metric" else v) for k, v in r.items()}
    for key, value in (s.get("test_metrics") or {}).items():
        rows.setdefault(key, {"metric": key, "estimate": num(value)})
    baseline = (s.get("baseline") or {}).get("test_metrics") or {}
    for key, row in rows.items():
        if row.get("baseline") is None and finite(baseline.get(key)):
            row["baseline"] = float(baseline[key])
            row["gain"] = oriented_gain(key, row.get("estimate"), row["baseline"])
        row["relative"] = relative_change(key, row.get("estimate"), row.get("baseline"))
    primary = ALIASES.get(s["metric"], s["metric"])
    return rows, rows.get(primary, {"metric": primary, "estimate": None})


def regression_facts(directory, s, context):
    pred = read_csv(directory / "test-predictions.csv", dtype={"sample_id": str})
    if pred is None or s["task"] != "regression":
        return {}
    y = pd.to_numeric(pred.observed, errors="coerce")
    p = pd.to_numeric(pred.predicted, errors="coerce")
    residual = y - p
    error = residual.abs()
    facts = {"median_ae": float(error.median()), "p90_ae": float(error.quantile(.9)),
             "max_ae": float(error.max()), "bias": float(residual.mean()),
             "target_sd_test": float(y.std(ddof=1)) if len(y) > 1 else None,
             "over_fraction": float((residual < 0).mean())}
    tolerance = num(context.get("tolerance"))
    if tolerance:
        facts["tolerance"] = tolerance
        facts["within_tolerance"] = float((error <= tolerance).mean())
    bins = read_csv(directory / "regression-bins.csv")
    if bins is not None and len(bins) >= 2:
        worst = bins.loc[bins.mae.idxmax()]
        best = bins.loc[bins.mae.idxmin()]
        facts["worst_bin"] = {"range": str(worst.bin), "mae": float(worst.mae), "n": int(worst.n),
                              "bias": float(worst.bias), "predicted_mean": float(worst.predicted_mean)}
        facts["best_bin_mae"] = float(best.mae)
        lowest, highest = bins.iloc[0], bins.iloc[-1]
        facts["edge_bias"] = {"low": float(lowest.bias), "high": float(highest.bias)}
    return facts


def classification_facts(directory, s):
    if s["task"] != "classification":
        return {}
    facts = {}
    table = read_csv(directory / "class-metrics.csv", dtype={"class": str})
    if table is not None and len(table):
        present = table[table.n > 0]
        facts["classes"] = present.to_dict("records")
        if len(present):
            weakest = present.loc[present.recall.idxmin()]
            strongest = present.loc[present.recall.idxmax()]
            facts["weakest"] = {"class": str(weakest["class"]), "recall": float(weakest.recall),
                                "precision": float(weakest.precision), "n": int(weakest.n)}
            facts["strongest"] = {"class": str(strongest["class"]), "recall": float(strongest.recall),
                                  "n": int(strongest.n)}
            facts["recall_spread"] = float(present.recall.max() - present.recall.min())
    pred = read_csv(directory / "test-predictions.csv", dtype={"sample_id": str, "observed": str, "predicted": str},
                    keep_default_na=False)
    if pred is not None:
        wrong = pred[pred.observed != pred.predicted]
        facts["error_n"] = int(len(wrong))
        if len(wrong):
            pair = wrong.groupby(["observed", "predicted"]).size().sort_values(ascending=False)
            (obs, prd), count = pair.index[0], int(pair.iloc[0])
            facts["top_confusion"] = {"observed": obs, "predicted": prd, "n": count,
                                      "share": count / len(wrong)}
        counts = pred.observed.value_counts()
        facts["test_counts"] = {str(k): int(v) for k, v in counts.items()}
        if len(counts):
            facts["majority_share"] = float(counts.iloc[0] / counts.sum())
    calibration = read_csv(directory / "calibration.csv")
    if calibration is not None and len(calibration):
        facts["ece"] = float(np.average((calibration.observed_fraction - calibration.mean_probability).abs(),
                                        weights=calibration.n))
        signed = float(np.average(calibration.observed_fraction - calibration.mean_probability,
                                  weights=calibration.n))
        facts["calibration_direction"] = "under" if signed > 0.02 else "over" if signed < -0.02 else "neutral"
    return facts


def importance_facts(directory):
    imp = read_csv(directory / "importance.csv", dtype={"feature": str}, keep_default_na=False)
    if imp is None or not len(imp):
        return {}
    imp = imp.sort_values("mean", ascending=False)
    positive = imp[imp["mean"] > 0]
    total = float(positive["mean"].sum())
    ranked = [{"feature": r.feature, "mean": float(r.mean), "sd": float(r.sd),
               "share": float(r.mean) / total if total > 0 and r.mean > 0 else 0.0,
               "stable": bool(r.mean - 2 * r.sd > 0)} for r in imp.itertuples()]
    facts = {"ranked": ranked, "n": len(ranked),
             "nonpositive": [r["feature"] for r in ranked if r["mean"] <= 0],
             "negligible": [r["feature"] for r in ranked
                            if r["mean"] <= 0 or (r["share"] < 0.02 and not r["stable"])],
             "unstable": [r["feature"] for r in ranked if r["mean"] > 0 and not r["stable"]]}
    cumulative, leaders = 0.0, []
    for r in ranked:
        if r["mean"] <= 0:
            break
        leaders.append(r["feature"])
        cumulative += r["share"]
        if cumulative >= 0.8 or len(leaders) >= 5:
            break
    facts["leaders"], facts["leader_share"] = leaders, cumulative
    corr = read_csv(directory / "feature-correlations.csv", dtype={"feature": str})
    if corr is not None and len(corr) and "feature" in corr:
        corr = corr.set_index("feature")
        corr.columns = [str(c) for c in corr.columns]
        pairs, names = [], list(corr.columns)
        top = set(r["feature"] for r in ranked[:10])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                value = corr.loc[a, b] if a in corr.index else None
                if finite(value) and abs(value) >= CORRELATION_FLAG and (a in top or b in top):
                    pairs.append({"a": a, "b": b, "rho": float(value)})
        facts["correlated_pairs"] = sorted(pairs, key=lambda x: -abs(x["rho"]))[:5]
    return facts


def shift_facts(directory):
    shift = read_csv(directory / "feature-shift.csv", dtype={"feature": str}, keep_default_na=False)
    if shift is None or not len(shift):
        return {}
    for column in ("standardized_mean_shift", "development_missing", "test_missing", "outside_development_range"):
        shift[column] = pd.to_numeric(shift[column], errors="coerce")
    flagged_shift = shift[shift.standardized_mean_shift.abs() >= SHIFT_FLAG_SD]
    missing_delta = (shift.test_missing - shift.development_missing)
    flagged_missing = shift[missing_delta.abs() >= MISSING_FLAG]
    flagged_range = shift[shift.outside_development_range >= RANGE_FLAG]
    return {
        "mean_shift": [{"feature": r.feature, "shift": float(r.standardized_mean_shift)}
                       for r in flagged_shift.sort_values("standardized_mean_shift", key=abs, ascending=False).itertuples()],
        "missing": [{"feature": r.feature, "development": float(r.development_missing),
                     "test": float(r.test_missing)} for r in flagged_missing.itertuples()],
        "out_of_range": [{"feature": r.feature, "fraction": float(r.outside_development_range)}
                         for r in flagged_range.sort_values("outside_development_range", ascending=False).itertuples()],
        "max_shift": float(shift.standardized_mean_shift.abs().max()) if shift.standardized_mean_shift.notna().any() else None,
        "any_missing": bool((shift.development_missing > 0).any() or (shift.test_missing > 0).any()),
        "development_missing_max": float(shift.development_missing.max()),
    }


def strata_facts(directory):
    strata = read_csv(directory / "strata-metrics.csv", dtype={"stratum": str})
    if strata is None or not len(strata):
        return {}
    measure = str(strata.metric.iloc[0])
    worst = strata.loc[strata.value.idxmax() if measure == "mae" else strata.value.idxmin()]
    return {"measure": measure, "n_strata": int(len(strata)), "min": float(strata.value.min()),
            "max": float(strata.value.max()), "worst": str(worst.stratum), "worst_value": float(worst.value),
            "worst_n": int(worst.n), "n_min": int(strata.n.min()), "n_max": int(strata.n.max())}


def feature_names(directory):
    shift = read_csv(directory / "feature-shift.csv", dtype={"feature": str}, keep_default_na=False)
    return shift.feature.tolist() if shift is not None and "feature" in shift else []


def target_facts(s):
    summary = s.get("target_summary")
    return summary if isinstance(summary, dict) else {}


def evidence_strength(s, primary, comparison, selected):
    """Coarse, transparent reasons that strengthen or weaken the main conclusion."""
    notes = {"strengths": [], "weaknesses": []}
    n = s.get("test_n") or 0
    if n < 50:
        notes["weaknesses"].append(("small_test", n))
    elif n >= 200:
        notes["strengths"].append(("large_test", n))
    lo, hi = primary.get("gain_lower"), primary.get("gain_upper")
    if lo is not None and hi is not None:
        notes["strengths" if lo > 0 else "weaknesses"].append(("gain_interval", (lo, hi)))
    estimate, lower, upper = primary.get("estimate"), primary.get("lower"), primary.get("upper")
    if estimate and lower is not None and upper is not None and abs(estimate) > 1e-12:
        width = (upper - lower) / abs(estimate)
        if width > 0.5:
            notes["weaknesses"].append(("wide_interval", width))
    if selected and len(comparison) > 1:
        near = [r for r in comparison if r["standing"] == "near"]
        if near:
            notes["weaknesses"].append(("close_runner_up", [r["model"] for r in near]))
        else:
            notes["strengths"].append(("clear_selection", None))
        if selected.get("baseline_gain_positive_folds") is not None and \
                selected["baseline_gain_positive_folds"] == selected["folds"]:
            notes["strengths"].append(("all_folds_beat_baseline", selected["folds"]))
    if s.get("external_test"):
        notes["strengths"].append(("external_test", None))
    return notes


def collect(directory, s, analysis):
    directory = Path(directory)
    context = load_context(directory)
    comparison, selected = comparison_facts(directory, s)
    tests, primary = test_facts(directory, s)
    facts = {
        "context": context,
        "comparison": comparison,
        "selected": selected,
        "tests": tests,
        "primary": primary,
        "regression": regression_facts(directory, s, context),
        "classification": classification_facts(directory, s),
        "importance": importance_facts(directory),
        "shift": shift_facts(directory),
        "strata": strata_facts(directory),
        "target": target_facts(s),
        "interval": (analysis or {}).get("interval") or {},
        "features": feature_names(directory),
    }
    facts["strength"] = evidence_strength(s, primary, comparison, selected)
    families = {r["family"] for r in comparison}
    if selected and len(comparison) > 1 and "linear" in families and families & NONLINEAR:
        facts["family_reading"] = "linear_best" if selected["family"] == "linear" else (
            "nonlinear_best" if selected["family"] in NONLINEAR else None)
    return facts
