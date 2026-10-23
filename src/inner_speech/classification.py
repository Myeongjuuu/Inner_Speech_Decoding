"""CSP is fitted anew inside every training fold, with entire runs held out."""
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from mne.decoding import CSP


def evaluate_csp(X, y, groups, n_components=4, classifier="lda", min_groups=2):
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if X.ndim != 3 or len(X) != len(y) or len(y) != len(groups) or not np.isfinite(X).all():
        raise ValueError("Expected finite trials x channels x samples and aligned labels/groups")
    if len(np.unique(groups)) < min_groups or len(np.unique(y)) < 2:
        raise ValueError("Insufficient independent runs or classes")
    if n_components < 1 or n_components > X.shape[1]:
        raise ValueError("Invalid CSP component count")
    folds, predictions = [], []
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(X, y, groups)):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Group leakage")
        if set(y[train]) != set(y) or set(y[test]) != set(y):
            raise ValueError("Every held-out run and training set must contain all classes")
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
        if classifier == "lda":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model = make_pipeline(csp, clf)
        elif classifier == "svm":
            model = make_pipeline(csp, StandardScaler(), SVC(kernel="linear", C=1))
        else:
            raise ValueError("Choose lda or svm")
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        folds.append(dict(fold=fold, held_out_group=str(groups[test][0]), n_train=len(train),
                          n_test=len(test), accuracy=accuracy_score(y[test], pred),
                          balanced_accuracy=balanced_accuracy_score(y[test], pred),
                          macro_f1=f1_score(y[test], pred, average="macro")))
        predictions.extend(dict(trial_index=int(i), fold=fold, true=int(t), predicted=int(p))
                           for i, t, p in zip(test, y[test], pred))
    return pd.DataFrame(folds), pd.DataFrame(predictions)


def classify_subject(cfg, subject):
    root = Path(cfg["output_root"])
    xs, ys, groups = [], [], []
    channels, sfreq = None, None
    for session in cfg["sessions"]:
        ep = mne.read_epochs(root / subject / session / "task-epo.fif", preload=True, verbose="ERROR")
        if ep.metadata is None or "run" not in ep.metadata:
            raise ValueError("Run metadata required")
        if channels is not None and (channels != ep.ch_names or sfreq != ep.info["sfreq"]):
            raise ValueError("Channel/sampling layout differs across sessions")
        channels, sfreq = ep.ch_names, ep.info["sfreq"]
        xs.append(ep.get_data(copy=True)); ys.append(ep.events[:, 2])
        groups.extend(session + "/run-" + str(r) for r in ep.metadata.run)
    settings = cfg["classification"]
    folds, predictions = evaluate_csp(np.concatenate(xs), np.concatenate(ys), groups, **settings)
    folds.to_csv(root / subject / "classification_folds.csv", index=False)
    predictions.to_csv(root / subject / "classification_predictions.csv", index=False)
    return folds
