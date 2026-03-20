"""
src/preprocessing/clean_eeg.py
================================
EEG preprocessing pipeline for OpenBCI 16-channel recordings.

Pipeline order (methodologically justified):
    1. Channel renaming (OpenBCI → 10-20 standard)
    2. Unit conversion (raw counts → µV → V for MNE)
    3. Notch filter (50 Hz line noise)          ← spectral cleaning first
    4. Bandpass filter (1–40 Hz)
    5. Common Average Reference (CAR)           ← AFTER spectral cleaning
    6. ICA artifact rejection (Extended Infomax + ICLabel)
    7. Timestamp reconstruction and DataFrame export

Note on filter order:
    Notch and Bandpass are applied BEFORE CAR to prevent local 50 Hz artifacts
    from contaminating the virtual average reference and propagating to all 15
    remaining channels before the notch can suppress them locally.
    CAR is then applied post-filtering so that ICLabel's topographic maps are
    computed on a spectrally clean, average-referenced signal, as required by
    its design (Bigdely-Shamlo et al., 2015).
    Reference: Bigdely-Shamlo et al. (2015), "The PREP pipeline", Front. Neuroinformatics.

Computational complexity:
    - Filtering: O(N * filter_len) per channel, where N = n_samples
    - ICA (Extended Infomax): O(N * n_components^2) per iteration, ~20-50 iterations typical
    - ICLabel inference: O(n_components) — negligible
    - Total wall time (125 Hz, 10 min recording): ~30-90 s depending on hardware

Author: Ignacio Negrete Silva
Project: Fondecyt 1231122
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for pipeline runs
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components

from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_eeg")


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def load_eeg_parquet(filepath: Path) -> pd.DataFrame:
    """
    Load an interim EEG parquet file (output of ingestion stage).

    Parameters
    ----------
    filepath : Path
        Path to the .parquet file for a single participant/session.

    Returns
    -------
    pd.DataFrame
        Raw EEG data with original OpenBCI column names.
        Shape: (n_samples, 33) — all original columns preserved.
    """
    logger.info(f"Loading EEG from: {filepath.name}")
    df = pd.read_parquet(filepath)
    logger.info(f"  Loaded {len(df):,} samples")
    return df


def build_mne_raw(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[mne.io.RawArray, pd.Series]:
    """
    Convert a raw OpenBCI DataFrame into an MNE RawArray.

    Unit conversion: OpenBCI delivers data in microvolts (uV).
    MNE internal representation requires Volts -> divide by 1e6.

    Parameters
    ----------
    df : pd.DataFrame
        Raw EEG DataFrame from ingestion stage.
    cfg : dict
        EEG config block from preprocessing.yaml.

    Returns
    -------
    raw : mne.io.RawArray
        MNE Raw object. Shape of underlying data: (n_channels=16, n_times).
    timestamps : pd.Series
        Original wall-clock timestamps (datetime64), preserved for later
        synchronization with Tobii. Shape: (n_times,).
    """
    electrode_map: dict = cfg["electrode_map"]
    sfreq: float = cfg["sfreq"]

    timestamps = pd.to_datetime(df[" Timestamp (Formatted)"])

    ch_names = list(electrode_map.values())
    raw_cols = list(electrode_map.keys())
    data = df[raw_cols].values.T / 1e6  # uV -> V; shape: (n_channels, n_times)

    logger.info(f"  Data matrix shape: {data.shape} (channels x samples)")
    logger.info(
        f"  Recording duration: "
        f"{(timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds():.1f} s"
    )

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, verbose=False)

    return raw, timestamps


def apply_preprocessing(
    raw: mne.io.RawArray,
    cfg: dict,
) -> mne.io.RawArray:
    """
    Apply the full preprocessing pipeline to an MNE Raw object.

    Pipeline order (SOTA — Bigdely-Shamlo et al., 2015):
        1. Notch filter  (50 Hz)      — suppress line noise locally per-channel
        2. Bandpass filter (1–40 Hz)  — remove DC drift and high-freq noise
        3. Common Average Reference (CAR) — spatial filter on the clean spectrum

    Rationale for this order:
        If CAR were applied first, any channel with residual 50 Hz contamination
        would inject that artifact into the virtual average reference and propagate
        it to all 15 remaining channels before the Notch can suppress it locally.
        Filtering first ensures the CAR averages a spectrally clean signal, and
        ICLabel receives topographic maps free of spectral contamination, as
        required by its design assumptions.

    Parameters
    ----------
    raw : mne.io.RawArray
        Unfiltered MNE Raw object. Shape: (n_channels, n_times).
    cfg : dict
        EEG config block from preprocessing.yaml.

    Returns
    -------
    mne.io.RawArray
        Filtered Raw object, ready for ICA. Same shape as input.
    """
    notch_freq: float = cfg["notch_freq"]
    bp_low: float = cfg["bandpass_low"]
    bp_high: float = cfg["bandpass_high"]

    # Step 1 — Notch: suppress 50 Hz line noise locally in sensor space
    logger.info(f"  Applying Notch filter at {notch_freq} Hz...")
    raw_proc = raw.copy()
    raw_proc.notch_filter(freqs=notch_freq, method="fir", phase="zero", verbose=False)

    # Step 2 — Bandpass: remove sub-Hz drift and high-freq muscle noise
    logger.info(f"  Applying bandpass filter [{bp_low}-{bp_high} Hz]...")
    raw_proc.filter(
        l_freq=bp_low,
        h_freq=bp_high,
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    # Step 3 — CAR: applied on spectrally clean signal as required by ICLabel
    logger.info("  Applying Common Average Reference (CAR)...")
    raw_proc.set_eeg_reference("average", projection=False, verbose=False)

    return raw_proc


def run_ica(
    raw_filtered: mne.io.RawArray,
    cfg: dict,
) -> tuple[mne.io.RawArray, dict]:
    """
    Run Extended Infomax ICA and reject artifact components via ICLabel.

    Parameters
    ----------
    raw_filtered : mne.io.RawArray
        Preprocessed Raw object. Shape: (n_channels, n_times).
    cfg : dict
        EEG config block from preprocessing.yaml.

    Returns
    -------
    raw_clean : mne.io.RawArray
        ICA-cleaned Raw object. Same shape as input.
    ica_report : dict
        Keys:
            - 'labels': list of str, ICLabel prediction per component
            - 'probabilities': np.ndarray, shape (n_components,)
            - 'excluded': list of int, indices of removed components
    """
    ica_cfg: dict = cfg["ica"]
    n_components: int = ica_cfg["n_components"]
    method: str = ica_cfg["method"]
    random_state: int = ica_cfg["random_state"]
    artifact_thresholds: dict = ica_cfg["artifact_thresholds"]

    logger.info(f"  Fitting {method} ICA ({n_components} components)...")
    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=dict(extended=True),
        random_state=random_state,
        verbose=False,
    )
    ica.fit(raw_filtered, verbose=False)

    logger.info("  Running ICLabel classification...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ic_labels = label_components(raw_filtered, ica, method="iclabel")

    labels = ic_labels["labels"]
    probs = np.array(ic_labels["y_pred_proba"]).flatten()

    exclude_idx = []
    for i, (label, prob) in enumerate(zip(labels, probs)):
        threshold = artifact_thresholds.get(label)
        if threshold is not None and prob > threshold:
            exclude_idx.append(i)
            logger.info(f"    Excluding IC {i:02d}: {label} (p={prob:.3f} > {threshold})")

    if not exclude_idx:
        logger.warning("  No components excluded — verify ICLabel output manually.")

    ica.exclude = exclude_idx
    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean, verbose=False)

    ica_report = {
        "labels": labels,
        "probabilities": probs,
        "excluded": exclude_idx,
    }

    logger.info(f"  ICA complete: {len(exclude_idx)} components removed.")
    return raw_clean, ica_report


def export_clean_dataframe(
    raw_clean: mne.io.RawArray,
    timestamps: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """
    Convert a cleaned MNE Raw object back to a DataFrame with wall-clock timestamps.

    Parameters
    ----------
    raw_clean : mne.io.RawArray
        ICA-cleaned Raw object. Shape: (n_channels, n_times).
    timestamps : pd.Series
        Original wall-clock timestamps. Shape: (n_times_original,).
    cfg : dict
        EEG config block from preprocessing.yaml.

    Returns
    -------
    pd.DataFrame
        Columns: [Fp1, Fp2, ..., P4, timestamp]. Shape: (n_times_clean, 17).
        EEG values in uV.
    """
    ch_names = list(cfg["electrode_map"].values())

    df_clean = raw_clean.to_data_frame()

    start_ts = timestamps.iloc[0]
    df_clean["timestamp"] = start_ts + pd.to_timedelta(df_clean["time"], unit="s")

    df_out = df_clean[ch_names + ["timestamp"]].copy()

    # MNE stores data in Volts internally — convert back to uV for feature engineering
    df_out[ch_names] = df_out[ch_names] * 1e6

    logger.info(
        f"  Export shape: {df_out.shape} | "
        f"duration: "
        f"{(df_out['timestamp'].max() - df_out['timestamp'].min()).total_seconds():.1f} s"
    )
    return df_out


# ---------------------------------------------------------------------------
# Reporting functions
# ---------------------------------------------------------------------------

def save_psd_report(
    raw_before: mne.io.RawArray,
    raw_after: mne.io.RawArray,
    participant_name: str,
    figures_dir: Path,
) -> None:
    """
    Save a PSD comparison plot (before vs. after preprocessing) to disk.

    Replicates the visual inspection done in the discovery notebook, but
    persisted automatically for every participant.

    Parameters
    ----------
    raw_before : mne.io.RawArray
        Raw signal before any filtering. Shape: (n_channels, n_times).
    raw_after : mne.io.RawArray
        Signal after CAR + notch + bandpass + ICA. Shape: (n_channels, n_times).
    participant_name : str
        Used as filename stem (e.g. 'participant01').
    figures_dir : Path
        Output directory for the figure.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, raw, title in zip(
        axes,
        [raw_before, raw_after],
        ["Before preprocessing", "After preprocessing (CAR + Notch + BP + ICA)"],
    ):
        psd = raw.compute_psd(fmax=62.5)
        psd.plot(axes=ax, average=True, show=False)
        ax.set_title(title, fontsize=11)

    fig.suptitle(f"PSD — {participant_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = figures_dir / f"psd_{participant_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  PSD report saved -> {out_path}")


def save_ica_report(
    ica_report: dict,
    participant_name: str,
    figures_dir: Path,
    qc_dir: Path,
) -> None:
    """
    Save the ICLabel classification table as a CSV and a bar chart.

    The CSV mirrors the DataFrame shown in the discovery notebook.
    The bar chart shows the probability distribution across all components,
    color-coded by artifact type — useful for visual QC in the thesis.

    Parameters
    ----------
    ica_report : dict
        Output of run_ica(). Keys: 'labels', 'probabilities', 'excluded'.
    participant_name : str
        Used as filename stem.
    figures_dir : Path
        Directory for the bar chart figure.
    qc_dir : Path
        Directory for the CSV table.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    labels = ica_report["labels"]
    probs = ica_report["probabilities"]
    excluded = set(ica_report["excluded"])
    n = len(labels)

    # --- CSV table ---
    df_ic = pd.DataFrame({
        "component": range(n),
        "label": labels,
        "probability": probs,
        "excluded": [i in excluded for i in range(n)],
    })
    csv_path = qc_dir / f"iclabel_{participant_name}.csv"
    df_ic.to_csv(csv_path, index=False)
    logger.info(f"  ICLabel CSV saved -> {csv_path}")

    # --- Bar chart ---
    color_map = {
        "brain": "#2ecc71",
        "eye blink": "#e74c3c",
        "muscle artifact": "#e67e22",
        "line noise": "#9b59b6",
        "heart": "#e91e63",
        "channel noise": "#795548",
        "other": "#95a5a6",
    }
    colors = [color_map.get(l, "#bdc3c7") for l in labels]
    edge_colors = ["black" if i in excluded else "none" for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(n), probs, color=colors, edgecolor=edge_colors, linewidth=1.5)

    ax.set_xlabel("Independent Component", fontsize=11)
    ax.set_ylabel("ICLabel Probability", fontsize=11)
    ax.set_title(
        f"ICLabel Classification — {participant_name}\n"
        f"(black border = excluded, n={len(excluded)})",
        fontsize=11,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"IC{i}\n{l}" for i, l in enumerate(labels)], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=0.8, label="threshold 0.70")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    out_path = figures_dir / f"iclabel_{participant_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ICLabel figure saved -> {out_path}")


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def preprocess_participant(
    filepath: Path,
    cfg: dict,
    output_dir: Optional[Path] = None,
    report_dirs: Optional[dict] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline for a single participant's EEG session.

    Parameters
    ----------
    filepath : Path
        Path to the interim .parquet EEG file.
    cfg : dict
        EEG config block from preprocessing.yaml.
    output_dir : Path, optional
        Directory to save the cleaned .parquet. Required if save=True.
    report_dirs : dict, optional
        Report output directories. Expected keys:
            - 'figures': Path  — for PSD and ICLabel plots
            - 'qc': Path       — for ICLabel CSV table
        If None, no reports are saved.
    save : bool, default True
        Whether to persist the cleaned DataFrame to disk.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned EEG DataFrame. Shape: (n_times, 17).
    ica_report : dict
        ICA artifact report for quality control.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {filepath.name}")

    participant_name = filepath.stem

    df_raw = load_eeg_parquet(filepath)
    raw, timestamps = build_mne_raw(df_raw, cfg)
    raw_filtered = apply_preprocessing(raw, cfg)
    raw_clean, ica_report = run_ica(raw_filtered, cfg)
    df_clean = export_clean_dataframe(raw_clean, timestamps, cfg)

    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filepath.name
        df_clean.to_parquet(out_path, index=False)
        logger.info(f"  Saved -> {out_path}")

    if report_dirs is not None:
        figures_dir = Path(report_dirs["figures"])
        qc_dir = Path(report_dirs["qc"])
        save_psd_report(raw, raw_clean, participant_name, figures_dir)
        save_ica_report(ica_report, participant_name, figures_dir, qc_dir)

    return df_clean, ica_report


def preprocess_all(
    eeg_interim_dir: Path,
    eeg_processed_dir: Path,
    cfg: dict,
    report_dirs: Optional[dict] = None,
) -> dict[str, dict]:
    """
    Run the preprocessing pipeline over all participant EEG files.

    Parameters
    ----------
    eeg_interim_dir : Path
        Directory containing interim .parquet EEG files.
    eeg_processed_dir : Path
        Output directory for cleaned .parquet files.
    cfg : dict
        EEG config block from preprocessing.yaml.
    report_dirs : dict, optional
        Report output directories. Same structure as in preprocess_participant.
        If provided, also saves the global ica_summary.csv to report_dirs['qc'].

    Returns
    -------
    dict[str, dict]
        Mapping participant filename -> ICA report.
    """
    eeg_files = sorted(eeg_interim_dir.glob("*.parquet"))

    if not eeg_files:
        raise FileNotFoundError(f"No .parquet files found in {eeg_interim_dir}")

    logger.info(f"Found {len(eeg_files)} EEG files to process.")

    reports = {}
    failed = []

    for filepath in eeg_files:
        try:
            _, ica_report = preprocess_participant(
                filepath=filepath,
                cfg=cfg,
                output_dir=eeg_processed_dir,
                report_dirs=report_dirs,
                save=True,
            )
            reports[filepath.name] = ica_report
        except Exception as e:
            logger.error(f"FAILED: {filepath.name} — {e}")
            failed.append(filepath.name)

    if failed:
        logger.warning(f"{len(failed)} files failed: {failed}")

    logger.info(
        f"Preprocessing complete: {len(reports)} succeeded, {len(failed)} failed."
    )

    # Save global ICA summary across all participants
    if report_dirs is not None and reports:
        qc_dir = Path(report_dirs["qc"])
        qc_dir.mkdir(parents=True, exist_ok=True)
        summary = build_ica_summary(reports)
        summary_path = qc_dir / "ica_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"  Global ICA summary saved -> {summary_path}")

    return reports


def build_ica_summary(reports: dict[str, dict]) -> pd.DataFrame:
    """
    Build a quality control summary table from ICA reports.

    Parameters
    ----------
    reports : dict[str, dict]
        Output of preprocess_all().

    Returns
    -------
    pd.DataFrame
        Shape: (n_participants, 5).
        Columns: n_excluded, eye_blink, muscle, line_noise, other.
    """
    rows = []
    for participant, report in reports.items():
        labels = report["labels"]
        excluded = set(report["excluded"])

        row = {
            "participant": participant,
            "n_excluded": len(excluded),
            "eye_blink": sum(
                1 for i, l in enumerate(labels)
                if l == "eye blink" and i in excluded
            ),
            "muscle": sum(
                1 for i, l in enumerate(labels)
                if l == "muscle artifact" and i in excluded
            ),
            "line_noise": sum(
                1 for i, l in enumerate(labels)
                if l == "line noise" and i in excluded
            ),
            "other": sum(
                1 for i, l in enumerate(labels)
                if l == "other" and i in excluded
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("participant")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Preprocessing Pipeline")
    parser.add_argument(
        "--preprocessing-config",
        type=str,
        default="configs/preprocessing.yaml",
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default="configs/paths.yaml",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Single participant filename. If omitted, processes all.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip saving figures and QC reports (faster, useful for testing).",
    )
    args = parser.parse_args()

    paths = load_config(args.paths_config)
    eeg_cfg = load_config(args.preprocessing_config)["eeg"]

    eeg_interim = Path(paths["data"]["interim"]["eeg"])
    eeg_processed = Path(paths["data"]["processed"]["eeg"])

    report_dirs = None if args.no_reports else {
        "figures": paths["reports"]["preprocessing"]["figures"],
        "qc":      paths["reports"]["preprocessing"]["qc"],
    }

    if args.participant:
        filepath = eeg_interim / args.participant
        preprocess_participant(
            filepath,
            cfg=eeg_cfg,
            output_dir=eeg_processed,
            report_dirs=report_dirs,
            save=True,
        )
    else:
        reports = preprocess_all(
            eeg_interim,
            eeg_processed,
            cfg=eeg_cfg,
            report_dirs=report_dirs,
        )
        summary = build_ica_summary(reports)
        print("\n=== ICA Quality Control Summary ===")
        print(summary.to_string())