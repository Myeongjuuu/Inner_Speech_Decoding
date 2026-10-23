from pathlib import Path
import copy
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    for key in ("dataset_root", "output_root"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
    cfg["config_file"] = str(path)
    return validate_config(cfg)


def validate_config(config):
    cfg = copy.deepcopy(config)
    for key in ("subjects", "sessions"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must contain unique, nonempty identifiers")
    if cfg["condition"] not in {"inner", "pronounced", "visualized"}:
        raise ValueError("Select an explicit experimental condition")
    prep = cfg["preprocessing"]
    if not 0 < prep["l_freq"] < prep["h_freq"] < prep["sfreq"] / 2:
        raise ValueError("Require 0 < l_freq < h_freq < target Nyquist")
    task, rest = cfg["windows"]["task"], cfg["windows"]["rest"]
    if min(*task, *rest) < 0 or task[1] <= task[0] or rest[1] <= rest[0]:
        raise ValueError("Windows must have nonnegative onsets and positive duration")
    if abs((task[1] - task[0]) - (rest[1] - rest[0])) > 1e-9:
        raise ValueError("Task/rest windows must have identical duration")
    duration = task[1] - task[0]
    for name, (lo, hi) in cfg["connectivity"]["bands"].items():
        if not prep["l_freq"] <= lo < hi <= prep["h_freq"]:
            raise ValueError(f"{name}: band must lie inside preprocessing passband")
        if lo * duration < cfg["connectivity"]["min_cycles"]:
            raise ValueError(f"{name}: window too short for minimum cycle requirement")
    return cfg
