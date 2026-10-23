from pathlib import Path
import json
import re
import tempfile
import shutil
import warnings
import mne
import numpy as np
from .events import parse_trials, select_pairs


def read_recording(path):
    raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    events = mne.find_events(raw, stim_channel="Status", shortest_event=1,
                             mask=65535, mask_type="and", verbose="ERROR")
    return raw, events


def audit_recording(path, cfg):
    raw, events = read_recording(path)
    trials, malformed = parse_trials(events, raw.info["sfreq"], raw.first_samp)
    pairs, short = select_pairs(trials, cfg["condition"], cfg["windows"]["task"], cfg["windows"]["rest"])
    codes, counts = np.unique(events[:, 2], return_counts=True)
    summary = dict(sfreq=float(raw.info["sfreq"]), n_channels=len(raw.ch_names),
                   duration_s=raw.n_times / raw.info["sfreq"],
                   event_counts={str(c): int(n) for c, n in zip(codes, counts)},
                   condition_counts=trials.condition.value_counts().to_dict(),
                   eligible_pairs=len(pairs), malformed_trials=len(malformed), short_trials=len(short))
    raw.close()
    return trials, pairs, malformed, short, summary


def extract_pairs(raw, table, duration, peak_uv, flat_uv):
    """Round event onsets only AFTER resampling; retain or reject both windows."""
    if len(table) == 0:
        raise ValueError("No eligible paired trials")
    sfreq = raw.info["sfreq"]
    samples = round(duration * sfreq)
    if samples < 2 or not np.isclose(samples / sfreq, duration):
        raise ValueError("Duration must correspond to an integer number of samples")
    epochs = []
    for kind in ("task", "rest"):
        event_samples = np.rint(table[kind + "_onset"].to_numpy() * sfreq).astype(int) + raw.first_samp
        if len(np.unique(event_samples)) != len(event_samples):
            raise ValueError("Resampling produced duplicate trial onsets")
        events = np.column_stack((event_samples, np.zeros(len(table), int), table.label.to_numpy(int)))
        ep = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=(samples - 1) / sfreq,
                        baseline=None, preload=True, picks="eeg", reject_by_annotation=True,
                        metadata=table.copy(), verbose="ERROR")
        epochs.append(ep)
    common = np.intersect1d(epochs[0].selection, epochs[1].selection)
    aligned = [ep[np.isin(ep.selection, common)] for ep in epochs]
    if any(len(ep) == 0 for ep in aligned):
        raise ValueError("No paired trials survived bounds/annotation rejection")
    good = np.ones(len(common), bool)
    for ep in aligned:
        ptp = np.ptp(ep.get_data(copy=True), axis=-1)
        good &= (ptp.max(axis=1) <= peak_uv * 1e-6) & (ptp.min(axis=1) >= flat_uv * 1e-6)
    reasons = {int(i): "bounds_or_annotation" for i in set(range(len(table))) - set(common)}
    reasons.update({int(i): "paired_amplitude_or_flat_rejection" for i in common[~good]})
    rejected = table.iloc[sorted(reasons)].copy()
    rejected["reason"] = [reasons[i] for i in sorted(reasons)]
    if not good.any():
        raise ValueError("All task/rest pairs failed amplitude QC")
    task, rest = (ep[good] for ep in aligned)
    if task.metadata.trial_id.tolist() != rest.metadata.trial_id.tolist():
        raise RuntimeError("Paired trial order mismatch")
    return task, rest, rejected


def preprocess_recording(path, cfg, output):
    """Continuous filtering/ICA before epoching, with a disk-backed buffer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw, events = read_recording(path)
    trials, malformed = parse_trials(events, raw.info["sfreq"], raw.first_samp)
    table, short = select_pairs(trials, cfg["condition"], cfg["windows"]["task"], cfg["windows"]["rest"])
    if len(table) == 0:
        raise ValueError("No valid trials for selected condition")
    prep = cfg["preprocessing"]
    eeg = [c for c in raw.ch_names if re.fullmatch(r"[ABCD](?:[1-9]|[12][0-9]|3[0-2])", c)]
    if len(eeg) != 128:
        raise ValueError(f"Expected 128 BioSemi scalp electrodes, found {len(eeg)}")
    ocular = [c for c in prep["ica"]["eog_channels"] if c in raw.ch_names]
    raw.pick(eeg + ocular)
    raw.set_channel_types({c: "eog" for c in ocular}, verbose="ERROR")
    raw.set_montage("biosemi128", on_missing="ignore", verbose="ERROR")
    bads = prep["bad_channels"]
    if not set(bads) <= set(eeg):
        raise ValueError("Unknown bad EEG channel")
    raw.info["bads"] = list(bads)
    audit = dict(ica_enabled=prep["ica"]["enabled"], bad_channels=bads, ica_excluded=[],
                 source_sfreq=raw.info["sfreq"], seed=cfg["seed"])
    audit["analysis_spec"] = dict(condition=cfg["condition"], windows=cfg["windows"],
                                   preprocessing={k: v for k, v in prep.items() if k != "scratch_dir"},
                                   seed=cfg["seed"])
    # Reading with a memmap path keeps the original 1024-Hz recording off the Python heap.
    raw.close()
    scratch = prep.get("scratch_dir")
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    scratch_base = Path(scratch) if scratch else Path(tempfile.gettempdir())
    required = raw.n_times * (len(eeg) + len(ocular)) * 8 + 256 * 1024**2
    if shutil.disk_usage(scratch_base).free < required:
        raise OSError(f"Need at least {required / 1024**3:.2f} GiB scratch space; "
                      "set preprocessing.scratch_dir to a drive with sufficient space")
    with tempfile.TemporaryDirectory(prefix="inner-speech-", dir=scratch) as tmp:
        raw = mne.io.read_raw_bdf(path, preload=str(Path(tmp) / "continuous.dat"),
                                  include=eeg + ocular, verbose="ERROR")
        raw.set_channel_types({c: "eog" for c in ocular}, verbose="ERROR")
        raw.set_montage("biosemi128", on_missing="ignore", verbose="ERROR")
        raw.info["bads"] = list(bads)
        notches = [f for f in prep["notch"] if f < raw.info["sfreq"] / 2]
        if notches:
            raw.notch_filter(notches, picks="eeg", verbose="ERROR")
        raw.filter(prep["l_freq"], prep["h_freq"], picks=[*eeg, *ocular], verbose="ERROR")
        raw.resample(prep["sfreq"], verbose="ERROR")
        if bads:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        if prep["reference"] != "average":
            raise ValueError("Only the explicitly documented average reference is supported")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        if prep["ica"]["enabled"]:
            if not ocular:
                raise ValueError("ICA requested but ocular channels unavailable")
            ica = mne.preprocessing.ICA(n_components=prep["ica"]["n_components"],
                                         method="fastica", random_state=cfg["seed"],
                                         max_iter=prep["ica"]["max_iter"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Keep ICA decimation's Nyquist safely above the already filtered passband.
                ica_decim = max(1, int(prep["sfreq"] // (2.5 * prep["h_freq"])))
                ica.fit(raw, picks="eeg", decim=ica_decim, verbose="ERROR")
            audit["ica_warnings"] = [str(w.message) for w in caught]
            if any("converg" in str(w.message).lower() for w in caught):
                raise RuntimeError("ICA did not converge; review settings before continuing")
            excluded = set(prep["ica"]["exclude"])
            for channel in ocular:
                found, _ = ica.find_bads_eog(raw, ch_name=channel, verbose="ERROR")
                excluded.update(found)
            ica.exclude = sorted(int(i) for i in excluded)
            ica.apply(raw, verbose="ERROR")
            audit["ica_excluded"] = ica.exclude
            audit["ica_components"] = int(ica.n_components_)
            ica.save(output / "ocular-ica.fif", overwrite=True, verbose="ERROR")
        duration = cfg["windows"]["task"][1] - cfg["windows"]["task"][0]
        task, rest, rejected = extract_pairs(raw, table, duration,
                                             prep["reject_peak_to_peak_uv"], prep["flat_peak_to_peak_uv"])
        task.save(output / "task-epo.fif", overwrite=True, verbose="ERROR")
        rest.save(output / "rest-epo.fif", overwrite=True, verbose="ERROR")
        audit.update(eligible_pairs=len(table), kept_pairs=len(task), sfreq=task.info["sfreq"],
                     n_eeg=len(task.ch_names), duration_s=duration,
                     class_counts=task.metadata.direction.value_counts().to_dict())
        raw.close()
        del raw
    for name, frame in (("trials", trials), ("malformed_trials", malformed),
                        ("short_trials", short), ("rejected_pairs", rejected)):
        frame.to_csv(output / (name + ".csv"), index=False)
    (output / "qc.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
