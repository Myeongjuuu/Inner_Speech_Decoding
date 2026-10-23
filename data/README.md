# Dataset layout

Download OpenNeuro **ds003626**, version **2.1.2**, separately. Do not commit recordings or derivatives.

```
ds003626/
  sub-01/ses-01/eeg/sub-01_ses-01_task-innerspeech_eeg.bdf
  sub-01/ses-02/eeg/sub-01_ses-02_task-innerspeech_eeg.bdf
  ...
```

Copy `configs/default.yaml` to `configs/local.yaml` and set `dataset_root` and `output_root`. Relative paths resolve against the YAML's directory. The local configuration is ignored by Git. Derived FIF files are recreated from raw BDF, not taken from historical notebook outputs. Do not deserialize the dataset's `.dat`/`.pkl` derivatives: some are Python pickles despite their extension.

References: [dataset DOI](https://doi.org/10.18112/openneuro.ds003626.v2.1.2), [data descriptor](https://doi.org/10.1038/s41597-022-01147-2), [authors' protocol implementation](https://github.com/N-Nieto/Inner_Speech_Dataset).
