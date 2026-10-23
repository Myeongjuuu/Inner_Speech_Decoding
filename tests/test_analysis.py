import numpy as np
import pandas as pd
import mne
import pytest
from inner_speech.events import parse_trials, select_pairs
from inner_speech.preprocessing import extract_pairs
from inner_speech.statistics import paired_inference, aggregate_subject
from inner_speech.classification import evaluate_csp
from inner_speech.connectivity import estimate_connectivity, edge_matrix
from inner_speech.config import load_config, validate_config


def protocol(missing=False):
    codes = [15, 21, 42, 31, 44, 45, 46, 16, 15, 22, 42, 32, 44, 45, 46, 42, 33, 44, 45, 46, 16]
    times = [0, .1, 1, 1.5, 2, 4.5, 5.5, 7, 8, 8.1, 9, 9.5, 10, 12.5, 13.5, 15, 15.5, 16, 18.5, 19.5, 21]
    e = np.column_stack((np.rint(np.array(times)*1024).astype(int), np.zeros(len(codes), int), codes))
    return e[np.arange(len(e)) != 13] if missing else e


def paired_epochs(n=30, sfreq=128):
    rng = np.random.default_rng(18)
    data = rng.normal(size=(n, 3, sfreq)) * 1e-6
    info = mne.create_info(["A1", "A2", "A3"], sfreq, "eeg")
    meta = pd.DataFrame({"trial_id": np.arange(n)})
    a = mne.EpochsArray(data, info, metadata=meta, verbose="ERROR")
    b = mne.EpochsArray(data.copy(), info, metadata=meta.copy(), verbose="ERROR")
    return a, b


def test_parser_excludes_other_conditions_and_measures_intervals():
    trials, rejected = parse_trials(protocol(), 1024)
    pairs, short = select_pairs(trials, "inner", [.5, 1.5], [.25, 1.25])
    assert len(trials) == 3 and len(pairs) == 2 and len(rejected) == len(short) == 0
    assert pairs.label.tolist() == [32, 33]
    np.testing.assert_allclose(pairs.task_onset, [10.5, 16.5])
    assert pairs.run.nunique() == 1


def test_missing_marker_is_rejected_not_invented():
    trials, rejected = parse_trials(protocol(missing=True), 1024)
    assert len(rejected) == 1 and len(trials) == 2
    assert 32 not in trials.label.to_list()


def test_run_end_prevents_condition_carryover():
    events = protocol()
    events = events[events[:, 2] != 22]
    trials, rejected = parse_trials(events, 1024)
    assert trials.condition.to_list() == ["pronounced"]
    assert len(rejected) == 2


def test_masked_status_and_absolute_samples():
    e = protocol(); e[:, 0] += 500; e[:, 2] |= 0x10000
    table, _ = parse_trials(e, 1024, first_samp=500)
    np.testing.assert_allclose(table.action_start, [2, 10, 16])


def test_short_rest_not_stretched():
    trials, _ = parse_trials(protocol(), 1024)
    pairs, rejected = select_pairs(trials, "inner", [.5, 1.5], [.5, 2.5])
    assert len(pairs) == 0 and len(rejected) == 2


def test_epoch_onsets_use_resampled_rate_and_paired_rejection():
    rng = np.random.default_rng(42)
    raw = mne.io.RawArray(rng.normal(0, 1e-6, (2, 100*256)), mne.create_info(["A1", "A2"], 256, "eeg"), verbose="ERROR")
    raw._data[:, 30*256:31*256] *= 1000
    table = pd.DataFrame(dict(trial_id=[1, 2], run=[1, 2], label=[31, 32],
                              task_onset=[10., 20.], rest_onset=[30., 40.]))
    task, rest, rejected = extract_pairs(raw, table, 1., 200, .01)
    assert len(task) == len(rest) == 1
    assert task.metadata.trial_id.tolist() == [2] == rest.metadata.trial_id.tolist()
    assert task.events[0, 0] == 20*256 and task.get_data().shape[-1] == 256
    assert rejected.trial_id.tolist() == [1]


def test_connectivity_pairs_and_known_identical_signals():
    task, rest = paired_epochs()
    # Exactly duplicate channels: coherence is one, independently of trial phases.
    task._data[:, 1] = task._data[:, 0]
    rest._data[:, 1] = rest._data[:, 0]
    settings = dict(methods=["coh"], bands={"alpha": [8., 13.]}, min_trials=20, min_cycles=5, mt_bandwidth=4)
    result = estimate_connectivity(task, rest, settings)
    assert result["task"].shape == (1, 3, 1)
    np.testing.assert_allclose(result["task"][0, 0], 1., atol=1e-10)
    np.testing.assert_array_equal(result["task"], result["rest"])
    rest.metadata.loc[0, "trial_id"] = 100
    with pytest.raises(ValueError, match="aligned"):
        estimate_connectivity(task, rest, settings)


def test_matrix_is_symmetric_with_nan_diagonal():
    matrix = edge_matrix([.2, .3, .4], [0, 0, 1], [1, 2, 2], 3)
    np.testing.assert_allclose(matrix, matrix.T, equal_nan=True)
    assert np.isnan(matrix.diagonal()).all()


def test_paired_permutation_detects_consistent_difference():
    x = np.column_stack((np.arange(1., 11), np.zeros(10)))
    result = paired_inference(x, bootstrap_samples=200)
    assert result["p_fwer"][0] < .05 and result["p_fwer"][1] == 1
    assert result["mean_difference"][0] == 5.5
    assert result["ci_low"][0] < 5.5 < result["ci_high"][0]
    with pytest.raises(ValueError, match="constant"):
        paired_inference(np.ones((10, 2)), bootstrap_samples=200)


def test_csp_is_fit_only_on_training_runs(monkeypatch):
    import inner_speech.classification as module
    original = module.CSP
    fit_sizes = []
    class RecordingCSP(original):
        def fit(self, X, y):
            fit_sizes.append(len(X))
            return super().fit(X, y)
    monkeypatch.setattr(module, "CSP", RecordingCSP)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 4, 128))
    y = np.tile([31, 32], 12)
    groups = np.repeat(["run1", "run2", "run3"], 8)
    folds, predictions = evaluate_csp(X, y, groups, n_components=2)
    assert fit_sizes == [16, 16, 16]
    assert len(folds) == 3 and sorted(predictions.trial_index) == list(range(24))


def test_missing_planned_session_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="Missing planned session"):
        aggregate_subject(tmp_path, ["ses-01"])


def test_config_rejects_unequal_duration_and_short_low_frequency():
    from pathlib import Path
    cfg = load_config(Path(__file__).parents[1] / "configs/default.yaml")
    cfg["windows"]["rest"] = [0, 2]
    with pytest.raises(ValueError, match="identical duration"):
        validate_config(cfg)
    cfg["windows"]["rest"] = [.25, 1.25]
    cfg["connectivity"]["bands"] = {"delta": [1, 4]}
    with pytest.raises(ValueError, match="too short"):
        validate_config(cfg)


def test_duplicate_trials_cannot_inflate_connectivity_sample_count():
    task, rest = paired_epochs()
    task.metadata.loc[1, "trial_id"] = 0
    rest.metadata.loc[1, "trial_id"] = 0
    settings = dict(methods=["coh"], bands={"alpha": [8., 13.]}, min_trials=20, min_cycles=5, mt_bandwidth=4)
    with pytest.raises(ValueError, match="Repeated"):
        estimate_connectivity(task, rest, settings)


def test_group_statistics_keeps_subject_as_unit_and_corrects_family(tmp_path):
    import json
    from inner_speech.statistics import group_analysis
    subjects = [f"sub-{i:02}" for i in range(1, 11)]
    sessions = ["ses-01", "ses-02"]
    for i, subject in enumerate(subjects):
        for session in sessions:
            folder = tmp_path / subject / session
            folder.mkdir(parents=True)
            delta = np.array([.1 + i * .01, .05 + i * .002, 0.]).reshape(1, 3, 1)
            np.savez(folder / "connectivity.npz", task=delta, rest=np.zeros_like(delta),
                     channels=["A1", "A2", "A3"], methods=["coh"], bands=["alpha"],
                     edge_i=[0, 0, 1], edge_j=[1, 2, 2], analysis_spec=np.array('same-configuration'))
    cfg = dict(output_root=str(tmp_path), subjects=subjects, sessions=sessions, seed=42,
               statistics=dict(min_subjects=5, n_permutations="all", bootstrap_samples=200))
    report = group_analysis(cfg)
    assert report["n_subjects"] == 10 and report["n_features"] == 3
    table = pd.read_csv(tmp_path / "group/paired_statistics.csv")
    assert table.n_subjects.tolist() == [10, 10, 10]
    assert table.p_fwer.tolist()[-1] == 1
    assert table.p_fwer.tolist()[0] < .05
