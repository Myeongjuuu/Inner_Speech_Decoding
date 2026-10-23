"""Participant-level paired differences; one max-T family across all features."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from mne.stats import permutation_t_test


def paired_inference(differences, n_permutations="all", bootstrap_samples=2000, seed=42):
    x = np.asarray(differences, float)
    if x.ndim != 2 or x.shape[0] < 3 or not np.isfinite(x).all():
        raise ValueError("Need finite participant-by-feature differences, at least 3 participants")
    n = len(x)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    variable = sd > 0
    # All-zero differences have t=0 and p=1. Nonzero constants are degenerate, not valid t tests.
    if np.any(~variable & (mean != 0)):
        raise ValueError("Nonzero constant differences: inspect degenerate features")
    t, p = np.zeros(x.shape[1]), np.ones(x.shape[1])
    if variable.any():
        t[variable], p[variable], _ = permutation_t_test(
            x[:, variable], n_permutations=n_permutations, tail=0, seed=seed, verbose="ERROR")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, (bootstrap_samples, n))
    weights = np.stack([np.bincount(row, minlength=n) / n for row in draws])
    low, high = np.empty(x.shape[1]), np.empty(x.shape[1])
    for start in range(0, x.shape[1], 256):
        bootstrap = weights @ x[:, start:start + 256]
        low[start:start + 256], high[start:start + 256] = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return dict(mean_difference=mean, t=t, p_fwer=p, ci_low=low, ci_high=high,
                cohen_dz=np.divide(mean, sd, out=np.zeros_like(mean), where=variable))


def aggregate_subject(directory, sessions):
    arrays, reference = [], None
    for session in sessions:
        path = Path(directory) / session / "connectivity.npz"
        if not path.is_file():
            raise ValueError(f"Missing planned session: {path}. No silent complete-case selection.")
        with np.load(path, allow_pickle=False) as data:
            current = {key: data[key] for key in data.files}
        if reference is None:
            reference = current
        for key in ("channels", "methods", "bands", "edge_i", "edge_j", "analysis_spec"):
            if not np.array_equal(current[key], reference[key]):
                raise ValueError(f"Session layout mismatch: {key}")
        arrays.append(current["task"] - current["rest"])
    # Equal session weighting, then one observation per participant.
    return np.mean(arrays, axis=0), reference


def group_analysis(cfg):
    root = Path(cfg["output_root"])
    differences, reference = [], None
    subjects = cfg["subjects"]
    if len(set(subjects)) != len(subjects) or len(subjects) < cfg["statistics"]["min_subjects"]:
        raise ValueError("Need the configured minimum of unique independent participants")
    for subject in subjects:
        diff, meta = aggregate_subject(root / subject, cfg["sessions"])
        if reference is not None:
            for key in ("channels", "methods", "bands", "edge_i", "edge_j", "analysis_spec"):
                if not np.array_equal(meta[key], reference[key]):
                    raise ValueError(f"Participant layout mismatch: {key}")
        reference = meta
        differences.append(diff)
    x = np.stack(differences)
    flat = x.reshape(len(subjects), -1)
    stat = paired_inference(flat, cfg["statistics"]["n_permutations"],
                             cfg["statistics"]["bootstrap_samples"], cfg["seed"])
    rows = []
    for m, method in enumerate(reference["methods"]):
        for edge, (i, j) in enumerate(zip(reference["edge_i"], reference["edge_j"])):
            for b, band in enumerate(reference["bands"]):
                feature = np.ravel_multi_index((m, edge, b), x.shape[1:])
                row = dict(method=str(method), band=str(band), channel_1=str(reference["channels"][i]),
                           channel_2=str(reference["channels"][j]), n_subjects=len(subjects))
                row.update({key: float(values[feature]) for key, values in stat.items()})
                rows.append(row)
    out = root / "group"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "paired_statistics.csv", index=False)
    np.savez_compressed(out / "participant_differences.npz", differences=x,
                        subjects=subjects, channels=reference["channels"],
                        methods=reference["methods"], bands=reference["bands"],
                        edge_i=reference["edge_i"], edge_j=reference["edge_j"])
    report = dict(n_subjects=len(subjects), n_features=flat.shape[1],
                  significant_fwer_005=int(np.sum(stat["p_fwer"] < 0.05)),
                  family="all configured methods x bands x undirected edges",
                  contrast="task minus inter-trial rest; equal session weights",
                  confidence_intervals="pointwise participant-bootstrap; NOT simultaneous",
                  assumptions="independent participants and sign-symmetric null differences",
                  exploratory=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
