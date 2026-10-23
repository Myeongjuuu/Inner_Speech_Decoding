import argparse
import json
import platform
import importlib.metadata
from pathlib import Path
from .config import load_config


def recording_path(cfg, subject, session):
    return Path(cfg["dataset_root"]) / subject / session / "eeg" / f"{subject}_{session}_task-innerspeech_eeg.bdf"


def run_stage(cfg, stage):
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    # Save resolved config and installed versions for every invocation.
    versions = {}
    for name in ("numpy", "scipy", "mne", "mne-connectivity", "scikit-learn", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    provenance = {"stage": stage, "config": cfg, "python": platform.python_version(), "versions": versions}
    (out / f"provenance_{stage}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if stage == "statistics":
        from .statistics import group_analysis
        return group_analysis(cfg)
    if stage == "classify":
        from .classification import classify_subject
        return {s: classify_subject(cfg, s).to_dict("records") for s in cfg["subjects"]}
    results = {}
    for subject in cfg["subjects"]:
        for session in cfg["sessions"]:
            directory = out / subject / session
            directory.mkdir(parents=True, exist_ok=True)
            key = f"{subject}/{session}"
            if stage == "audit":
                from .preprocessing import audit_recording
                trials, pairs, malformed, short, result = audit_recording(recording_path(cfg, subject, session), cfg)
                for name, frame in (("audit_trials", trials), ("audit_pairs", pairs),
                                    ("audit_malformed", malformed), ("audit_short", short)):
                    frame.to_csv(directory / f"{name}.csv", index=False)
                (directory / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            elif stage == "preprocess":
                from .preprocessing import preprocess_recording
                result = preprocess_recording(recording_path(cfg, subject, session), cfg, directory)
            elif stage == "connectivity":
                from .connectivity import save_connectivity
                result = save_connectivity(directory, cfg["connectivity"])
            else:
                raise ValueError(f"Unknown stage {stage}")
            results[key] = result
            print(key, json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Auditable inner-speech EEG analysis")
    parser.add_argument("stage", choices=["audit", "preprocess", "connectivity", "statistics", "classify"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    result = run_stage(load_config(args.config), args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
