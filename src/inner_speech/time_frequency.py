import numpy as np


def multitaper_power(epochs, frequencies=None):
    """Actual multitaper TFR. No wavelet label or unavailable prestimulus baseline."""
    frequencies = np.arange(8., 36., 2.) if frequencies is None else np.asarray(frequencies)
    return epochs.compute_tfr(method="multitaper", freqs=frequencies, n_cycles=frequencies * 0.25,
                              time_bandwidth=2.0, return_itc=False, average=True,
                              picks="eeg", n_jobs=1, verbose="ERROR")
