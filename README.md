# Project Structure – Multimodal Neurophysiological Data Analysis

This repository follows a structured, the goal is to ensure **reproducibility, modularity, and clarity** when working with multimodal experimental data (EEG, eye tracking, pupil diameter, and galvanic skin response).

The project is organized according to the different stages of the data pipeline:

1. Data ingestion
2. Data preprocessing and cleaning
3. Signal synchronization
4. Feature engineering
5. Model development and experimentation
6. Reporting and results

---

# Repository Structure

```
project/

│
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── features/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── synchronization/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── experiments/
│
├── models/
│
├── reports/
│
└── configs/
```

---

# Data Directory

The `data/` directory contains all datasets used in the project.
It is divided into stages reflecting the processing pipeline.

## raw/

Contains **original data exactly as collected** from the experiment.

These files must **never be modified**.

Example:

```
data/raw/

    eeg/
        participant01.txt
        participant02.txt

    tobii/
        data_export.tsv
```

---

## interim/

Intermediate datasets created during the **data ingestion phase**.

Typical operations:

* parsing raw files
* separating Tobii data by participant
* converting files to efficient formats (e.g., Parquet)

Example:

```
data/interim/

    eeg/
        participant01.parquet
        participant02.parquet

    tobii/
        participant01.parquet
        participant02.parquet
```

---

## processed/

Cleaned and **synchronized multimodal datasets**.

At this stage, signals from multiple modalities are aligned in time.

Example:

```
data/processed/

    multimodal/
        participant01.parquet
        participant02.parquet
```

Each dataset may contain:

| timestamp | gaze_x | gaze_y | pupil | gsr | eeg |
| --------- | ------ | ------ | ----- | --- | --- |

---

## features/

Datasets ready for machine learning models.

Feature engineering may include:

* EEG band power
* fixation statistics
* pupil dilation metrics
* GSR peaks
* temporal windows

Example:

```
data/features/

    dataset_v1.parquet
    dataset_v2.parquet
```

---

# Source Code (`src/`)

All reusable project code lives in the `src/` directory.

```
src/

    ingestion/
    preprocessing/
    synchronization/
    features/
    models/
    utils/
```

---

## ingestion/

Responsible for **reading and parsing raw datasets**.

Examples:

* loading EEG text files
* loading Tobii TSV exports

Example files:

```
src/ingestion/

    load_eeg.py
    load_tobii.py
```

---

## preprocessing/

Responsible for **data cleaning and preparation**.

Typical operations:

* artifact removal
* missing data handling
* normalization
* filtering

Example:

```
src/preprocessing/

    clean_eeg.py
    clean_eye_tracking.py
```

---

## synchronization/

Responsible for **aligning multimodal signals in time**.

This stage is critical for multimodal analysis.

Typical operations:

* timestamp alignment
* signal resampling
* window synchronization

Example:

```
src/synchronization/

    align_timestamps.py
    resample_signals.py
```

---

## features/

Feature engineering for machine learning models.

Examples:

* EEG spectral features
* fixation statistics
* pupil dilation metrics
* GSR response features

Example:

```
src/features/

    eeg_features.py
    eye_features.py
    gsr_features.py
```

---

## models/

Definition and training of machine learning and deep learning models.

Examples:

```
src/models/

    cnn_eeg.py
    multimodal_model.py
    train.py
```

Models may include:

* neural networks
* multimodal architectures
* baseline ML models

---

## utils/

Utility functions used across the project.

Examples:

```
src/utils/

    plotting.py
    io.py
    metrics.py
```

---

# Notebooks

The `notebooks/` directory is used for **exploratory analysis and prototyping**.

Example:

```
notebooks/

    01_exploracion_eeg.ipynb
    02_exploracion_tobii.ipynb
    03_sincronizacion.ipynb
    04_feature_engineering.ipynb
    05_modelos.ipynb
```

Guideline:

* exploratory work → notebooks
* reusable code → `src/`

---

# Experiments

The `experiments/` directory stores **controlled model experiments**.

Example:

```
experiments/

    exp01_baseline/
    exp02_cnn/
    exp03_multimodal_model/
```

Each experiment can contain:

```
exp01_baseline/

    config.yaml
    train.py
    results.json
```

---

# Models

Trained model artifacts are stored here.

Example:

```
models/

    baseline.pkl
    cnn_eeg.pt
    multimodal_model.pt
```

---

# Configurations

Centralized configuration files.

```
configs/

    preprocessing.yaml
    model.yaml
    training.yaml
```

This allows experiment parameters to be controlled without modifying code.

---

# Reports

Generated figures, tables, and quality control outputs produced at each stage
of the pipeline. Organized by pipeline stage to ensure traceability between
code and thesis content.

```
reports/

    ingestion/
    preprocessing/
    synchronization/
    features/
    models/
```

Each subdirectory mirrors a stage in `src/` and may contain:

* `figures/`  — plots (.png, .pdf) ready for the thesis
* `tables/`   — CSV or LaTeX exports of summary statistics
* `qc/`       — quality control outputs (e.g. ICA reports, artifact counts)

Example contents per stage:

```
reports/

    preprocessing/
        figures/
            psd_before_after_participant01.png
            ica_components_participant01.png
        qc/
            ica_summary.csv

    synchronization/
        figures/
            timestamp_overlap_participant01.png
        qc/
            sync_offsets.csv

    features/
        figures/
            band_power_distributions.png
            pupil_dilation_by_stimulus.png
        tables/
            feature_statistics.csv

    models/
        figures/
            confusion_matrix_multimodal.png
            roc_curves.png
        tables/
            results_summary.csv
```

---

# Data Processing Pipeline

The complete pipeline follows these stages:

```
RAW DATA
    ↓
DATA INGESTION
    ↓
PREPROCESSING
    ↓
SIGNAL SYNCHRONIZATION
    ↓
FEATURE ENGINEERING
    ↓
MODEL TRAINING
    ↓
RESULTS AND REPORTS
```

---

# Design Principles

This project structure is based on the following principles:

* **Reproducibility**
* **Separation of concerns**
* **Modular code design**
* **Scalability for experiments**
* **Clear data lineage**

Each processing stage produces a **new dataset version**, ensuring that transformations are traceable and reproducible.