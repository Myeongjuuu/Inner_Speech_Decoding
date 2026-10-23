"""Strict trigger parser. Missing markers are reported, never synthesized.

Protocol definitions: N-Nieto/Inner_Speech_Dataset, events_analysis.py.
This independent implementation uses measured marker times, not fixed offsets.
"""
import numpy as np
import pandas as pd

CONDITIONS = {21: "pronounced", 22: "inner", 23: "visualized"}
DIRECTIONS = {31: "up", 32: "down", 33: "right", 34: "left"}
TRIAL_COLUMNS = ["trial_id", "run", "condition", "label", "direction",
                 "trial_start", "cue", "action_start", "action_end", "rest_start", "rest_end"]


def parse_trials(events, sfreq, first_samp=0):
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 3 or sfreq <= 0:
        raise ValueError("Expected MNE events (n, 3) and positive sampling rate")
    if np.any(np.diff(events[:, 0]) < 0):
        raise ValueError("Events must be sorted by sample")
    # BioSemi upper status bits are not protocol codes.
    codes = events[:, 2].astype(int) & 0xFFFF
    valid = codes != 0
    events, codes = events[valid], codes[valid]
    times = (events[:, 0] - first_samp) / float(sfreq)
    condition, run = None, 0
    rows, rejected = [], []
    for i, code in enumerate(codes):
        if code == 15:
            run += 1
            condition = None
        elif code == 16:
            condition = None
        elif code in CONDITIONS:
            condition = CONDITIONS[code]
        elif code == 42:
            reason = None
            block = codes[i:i + 5].tolist()
            if condition is None or run == 0:
                reason = "missing_run_or_condition"
            elif len(block) < 5 or block[1] not in DIRECTIONS or block[2:] != [44, 45, 46]:
                reason = "incomplete_trial_sequence"
            else:
                end = i + 5
                if end >= len(codes) or codes[end] not in {42, 16, 17}:
                    reason = "unbounded_or_unexpected_rest_end"
                elif np.any(np.diff(times[i:end + 1]) <= 0):
                    reason = "nonpositive_interval"
            if reason:
                rejected.append({"event_index": i, "time_s": times[i], "run": run,
                                 "condition": condition, "reason": reason})
                continue
            rows.append(dict(trial_id=int(events[i, 0]), run=run, condition=condition,
                             label=int(codes[i + 1]), direction=DIRECTIONS[codes[i + 1]],
                             trial_start=times[i], cue=times[i + 1], action_start=times[i + 2],
                             action_end=times[i + 3], rest_start=times[i + 4], rest_end=times[i + 5]))
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS), pd.DataFrame(rejected)


def select_pairs(trials, condition, task_window, rest_window):
    """Keep only trials whose full windows fit inside their measured intervals."""
    table = trials.loc[trials.condition == condition].copy()
    for kind, window in (("task", task_window), ("rest", rest_window)):
        anchor = "action_start" if kind == "task" else "rest_start"
        table[kind + "_onset"] = table[anchor] + window[0]
        table[kind + "_stop"] = table[anchor] + window[1]
    good = ((table.task_stop <= table.action_end) & (table.rest_stop <= table.rest_end))
    rejected = table.loc[~good].copy()
    rejected["reason"] = "window_exceeds_measured_interval"
    return table.loc[good].reset_index(drop=True), rejected
