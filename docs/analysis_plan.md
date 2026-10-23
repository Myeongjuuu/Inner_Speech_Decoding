# Analysis plan and interpretation limits

This plan follows inspection of existing exploratory outputs. It is **not preregistered** and is not an independent confirmation of those outputs.

## Event and window audit

Use the original 1024-Hz Status channel, masking BioSemi upper bits. Protocol markers: 15/16 run boundaries; 21 pronounced, 22 inner speech, 23 visualization; 42 trial onset; 31-34 directional cues; 44 action onset; 45 action end/relax transition; 46 inter-trial rest onset. Condition is inferred from explicit block markers and reset at each run boundary. A direction marker alone does not identify inner speech.

Accept only the complete ordered sequence `42, direction, 44, 45, 46`, with a following 42, 16, or cognitive-question marker 17 bounding rest. Record and drop incomplete trials instead of imputing missing event times. Store counts, measured interval lengths, and rejection reasons for review. This conservative parser may discard valid but nonstandard sequences and does not implement the dataset authors' ad-hoc repairs.

Default action window: **44 + 0.5 to 1.5 s**. Rest window: **46 + 0.25 to 1.25 s**. Both are half-open 1-second intervals (256 samples at 256 Hz); each must fit within its measured protocol interval. They are NOT offsets from the directional cue. The data descriptor's 1.5-3.5 s interval uses a different reference point. The rest comparator is inter-trial rest, not an independently recorded resting state; order, carry-over, attention and cue differences remain confounds. Do not infer causation or inner-speech specificity from task/rest differences alone.

## Preprocessing and QC

Filter the continuous recording, then resample to 256 Hz. Derive new event samples from onset **seconds** at the new sampling rate; never reuse 1024-Hz sample indices. Use the 128 scalp EEG channels, retain known ocular channels only for ICA detection, exclude Status/EXG from estimates. Apply the same average reference and parameters to both conditions.

ICA is enabled by default, fitted on continuous filtered data with a fixed seed, and excludes components detected by EOG correlation plus explicitly configured indices. Nonconvergence stops processing. This is ocular artifact control, **not a guarantee of artifact-free EEG**. Review ICA, bad channels, EMG and channel spectra before interpreting results. Interpolated channels can distort connectivity; assess sensitivity excluding them with a consistent channel set across participants. Neither automated mouth-EMG exclusion nor source reconstruction is implemented.

Reject an entire task/rest pair if either window is annotated bad, out of bounds, exceeds 200 microvolts peak-to-peak in any EEG channel, or is flatter than 0.1 microvolts in any channel. Thresholds are explicit reviewable starting values. Use actual QC counts, not assumed retention. A temporary memory-mapped file limits initial BDF loading; filtering, resampling and ICA still require substantial RAM/time.

## Connectivity

Default band: alpha 8-13 Hz, an exploratory choice based on the historical analysis. One-second epochs cannot support stable delta-band estimation; configuration validation requires at least five cycles at the band's lower edge. Window length, trial count and estimator are matched across conditions.

Estimate debiased squared wPLI, coherence and PLV with multitaper spectra **across epochs**, pooling directions within the explicitly selected condition. Minimum 20 retained pairs per session is an operational guard, not a power guarantee. Never call the across-epochs estimator separately on a single trial. Debiased squared wPLI is not ordinary wPLI, can be negative, and is not clipped. Lag-sensitive estimates reduce some zero-lag mixing effects but can miss true zero-lag coupling. Sensor connections are not anatomical or causal brain-region connections.

Save unique undirected edges; plots mirror those edges and use NaN on the diagonal. No fake zero upper triangle, arbitrary PLV>0.5 discovery threshold, or unsupported anatomical labels.

## Statistical unit and inference

Compute task-minus-rest differences per session, then average planned sessions **equally** within each participant. Require every configured session and the same channel/method/band layout; missing sessions fail explicitly. The group array has one row per independent participant, not one row per epoch, electrode or overlapping window.

Use a two-sided participant sign-flip t test. The null requires sign-symmetric participant differences; condition order was not randomized and permutation does not remove its confounds. Evaluate all possible signs for the small sample where feasible. All configured methods, bands and 8,128 edges are a **single max-T FWER family**. Report raw mean difference, paired Cohen's dz, corrected p, and participant-bootstrap 95% pointwise intervals. These intervals are not simultaneous or corrected. Do not add selected bands or change parameters after inspecting significance and call that confirmatory. Very small participant counts limit precision; nonsignificance does not establish equivalence. No NBS or direction-specific inferential test is implemented in this version.

## Classification

Use within-participant leave-one-run-out evaluation. Session/run combinations define groups. CSP is fitted inside each training fold, followed by shrinkage LDA or fixed linear SVM; feature scaling for SVM is inside the pipeline. Every train/test fold must contain all configured labels. Report fold accuracy, balanced accuracy, macro-F1 and out-of-fold predictions. No hyperparameter search is conducted. This is NOT subject-independent generalization. Continuous session-level ICA/reference are unsupervised preprocessing shared across folds, so this is not a fully inductive real-time deployment evaluation. A future deployment study should fit learned preprocessing on training data only or independent calibration.

## Primary sources

- Nieto et al., [dataset descriptor](https://doi.org/10.1038/s41597-022-01147-2).
- [Protocol trigger definitions](https://github.com/N-Nieto/Inner_Speech_Dataset/blob/main/Python_Processing/lib/events_analysis.py) and [preprocessing reference](https://github.com/N-Nieto/Inner_Speech_Dataset/blob/main/Python_Processing/InnerSpeech_preprocessing.py). Consulted for protocol, not copied as implementation.
- [MNE across-epochs connectivity estimator](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html).
- Vinck et al., [wPLI and estimator bias](https://doi.org/10.1016/j.neuroimage.2011.01.055).
- [MNE paired permutation/max-T test](https://mne.tools/stable/generated/mne.stats.permutation_t_test.html).
