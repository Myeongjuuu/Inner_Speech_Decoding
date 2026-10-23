"""Across-trial spectral estimators; no single-epoch PLV pseudo-replication."""
from pathlib import Path
import json
import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

SUPPORTED = {"wpli2_debiased", "coh", "plv", "ciplv", "ppc"}


def estimate_connectivity(task, rest, settings):
    if task.ch_names != rest.ch_names or task.get_channel_types() != ["eeg"] * len(task.ch_names):
        raise ValueError("Require identical EEG-only channel order")
    if len(task) != len(rest) or task.get_data().shape != rest.get_data().shape:
        raise ValueError("Task and rest must have identical trial counts and sample shapes")
    if task.info["sfreq"] != rest.info["sfreq"]:
        raise ValueError("Sampling rates differ")
    if task.metadata is None or rest.metadata is None or not task.metadata.trial_id.equals(rest.metadata.trial_id):
        raise ValueError("Explicit, aligned paired-trial metadata is required")
    if not task.metadata.trial_id.is_unique:
        raise ValueError("Repeated trial IDs cannot be treated as independent epochs")
    if len(task) < settings["min_trials"]:
        raise ValueError(f"Only {len(task)} pairs; need {settings['min_trials']} for estimation")
    methods = settings["methods"]
    if not methods or not set(methods) <= SUPPORTED:
        raise ValueError("Unsupported or empty connectivity methods")
    n = len(task.ch_names)
    # Unique undirected off-diagonal edges, with no artificial zero triangle.
    indices = np.triu_indices(n, k=1)
    bands = settings["bands"]
    duration = task.get_data().shape[-1] / task.info["sfreq"]
    for band, (low, high) in bands.items():
        if low * duration < settings["min_cycles"]:
            raise ValueError(f"{band}: insufficient cycles")
        if not 0 < low < high < task.info["sfreq"] / 2:
            raise ValueError(f"{band}: invalid frequency range")
    values = []
    for epochs in (task, rest):
        con = spectral_connectivity_epochs(
            epochs, indices=indices, method=methods, mode="multitaper",
            fmin=tuple(v[0] for v in bands.values()), fmax=tuple(v[1] for v in bands.values()),
            faverage=True, mt_bandwidth=settings["mt_bandwidth"], n_jobs=1, verbose="ERROR")
        # A list of estimators returns one object per method, even for a length-1 list.
        if not isinstance(con, list):
            con = [con]
        values.append(np.stack([x.get_data() for x in con]))  # method, edge, band
    return dict(task=values[0], rest=values[1], edge_i=indices[0], edge_j=indices[1],
                channels=np.asarray(task.ch_names), methods=np.asarray(methods),
                bands=np.asarray(list(bands)), n_pairs=np.array(len(task)))


def save_connectivity(directory, settings):
    directory = Path(directory)
    qc_path = directory / "qc.json"
    if not qc_path.is_file():
        raise ValueError("Completed preprocessing QC/provenance required before connectivity")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    task = mne.read_epochs(directory / "task-epo.fif", preload=True, verbose="ERROR")
    rest = mne.read_epochs(directory / "rest-epo.fif", preload=True, verbose="ERROR")
    result = estimate_connectivity(task, rest, settings)
    result["analysis_spec"] = np.array(json.dumps(
        dict(preprocessing=qc["analysis_spec"], connectivity=settings), sort_keys=True))
    np.savez_compressed(directory / "connectivity.npz", **result)
    return {"n_pairs": int(result["n_pairs"]), "n_edges": len(result["edge_i"])}


def edge_matrix(values, edge_i, edge_j, n_channels):
    matrix = np.full((n_channels, n_channels), np.nan)
    matrix[edge_i, edge_j] = values
    matrix[edge_j, edge_i] = values
    return matrix
