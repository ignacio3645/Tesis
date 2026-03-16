"""
src/preprocessing/clean_tobii.py
=================================
Tobii Pro preprocessing pipeline for the Fondecyt 1231122 dataset.

Handles three co-registered signals captured by the same device (shared hardware clock):
    - GSR / EDA  : galvanic skin response — tonic (SCL) / phasic (SCR) decomposition
    - Pupillometry: pupil diameter — blink interpolation + luminance correction placeholder
    - Eye tracking: fixation / saccade events derived from Tobii's IVT classification

Pipeline order:
    1. Timestamp reconstruction   (Recording date + start time + Recording timestamp)
    2. Stream resampling          (irregular ~65-68 Hz → uniform 60 Hz grid)
    3. GSR cleaning               (artifact detection → normalization → SCL/SCR decomposition)
    4. Pupil cleaning             (blink detection → cubic interpolation → normalization)
    5. Eye tracking extraction    (fixations, saccades, valid movement filtering)
    6. Report generation          (QC figures + tables → reports/preprocessing/)

Note on GSR density:
    The Tobii Pro GSR sensor captures at ~8 Hz effective rate within a ~60 Hz stream,
    resulting in ~18% sample density. This is structural, not data loss.
    The sparse GSR samples are resampled onto the uniform 60 Hz grid after decomposition.

Note on synchronization:
    GSR, pupil, and eye tracking share hardware clock — NO inter-modal sync needed.
    EEG ↔ Tobii sync is handled separately in synchronize.py.

Computational complexity:
    - Resampling: O(N) per signal
    - cvxEDA decomposition: O(N^1.5) approximately — dominant cost (~30-120 s per session)
    - Blink interpolation: O(N) — negligible
    - Total wall time: ~1-3 min per participant depending on hardware and GSR length

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
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import zscore

from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_tobii")


# ---------------------------------------------------------------------------
# 1. Timestamp reconstruction
# ---------------------------------------------------------------------------

def reconstruct_timestamps(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Reconstruct absolute wall-clock timestamps for each Tobii sample.

    The Tobii Pro provides three time-related columns:
        - Recording date:       date of recording (mm/dd/yyyy — confirmed for this cohort)
        - Recording start time: exact start time  (HH:MM:SS.ffffff)
        - Recording timestamp:  microseconds elapsed since recording start (int64, starts at 0)

    Absolute timestamp = parse(date + start_time) + timedelta(Recording timestamp, unit='us')

    Note: 'Computer timestamp' is a Windows FILETIME integer that pd.to_datetime()
    misinterprets as nanoseconds from Unix epoch → produces 1970 timestamps. Do NOT use it.

    Parameters
    ----------
    df  : pd.DataFrame — raw Tobii parquet for one participant
    cfg : dict         — tobii block from preprocessing.yaml

    Returns
    -------
    pd.DataFrame with new column 'timestamp' (datetime64[ns])
    """
    cols       = cfg["columns"]
    date_fmt   = cfg["date_format"]

    df = df.copy()
    df["recording_datetime"] = pd.to_datetime(
        df[cols["recording_date"]] + " " + df[cols["recording_start_time"]],
        format=date_fmt,
    )
    df["timestamp"] = (
        df["recording_datetime"]
        + pd.to_timedelta(df[cols["recording_timestamp"]], unit="us")
    )

    assert df["timestamp"].min().year > 2000, (
        f"Timestamp reconstruction failed: year={df['timestamp'].min().year}. "
        "Check date_format in preprocessing.yaml."
    )

    logger.info(
        f"  Timestamps OK: {df['timestamp'].min()} → {df['timestamp'].max()} "
        f"({(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.1f} s)"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Stream resampling
# ---------------------------------------------------------------------------

def build_uniform_index(
    timestamps: pd.Series,
    sfreq_target: float,
) -> pd.DatetimeIndex:
    """
    Build a single uniform DatetimeIndex for a participant session.

    Called ONCE per participant and shared across all signals (GSR, pupil, ET)
    to guarantee that all resampled Series have identical indices — required for
    assembling the final DataFrame in export_clean_dataframe().

    Parameters
    ----------
    timestamps   : pd.Series — raw Tobii timestamps (datetime64) for the session
    sfreq_target : float     — target sampling frequency in Hz

    Returns
    -------
    pd.DatetimeIndex at sfreq_target Hz spanning the full session
    """
    t_start = timestamps.dropna().min()
    t_end   = timestamps.dropna().max()

    if pd.isna(t_start) or pd.isna(t_end):
        raise ValueError(
            "Cannot build uniform index: all timestamps are NaT. "
            "Check timestamp reconstruction for this participant."
        )

    dt_target = pd.Timedelta(seconds=1.0 / sfreq_target)
    return pd.date_range(start=t_start, end=t_end, freq=dt_target)


def resample_to_uniform_grid(
    series: pd.Series,
    timestamps: pd.Series,
    uniform_index: pd.DatetimeIndex,
    gap_max_ms: float,
) -> pd.Series:
    """
    Resample an irregularly-sampled Tobii signal onto a pre-built uniform time grid.

    All signals for a given participant MUST use the same uniform_index (built once
    via build_uniform_index()) so that the resulting Series share an identical index
    and can be assembled into a single DataFrame without reindex mismatches.

    Gaps larger than gap_max_ms are NOT interpolated — they remain NaN in the output.

    Parameters
    ----------
    series        : pd.Series          — signal values (may contain NaN)
    timestamps    : pd.Series          — datetime64 timestamps aligned with series
    uniform_index : pd.DatetimeIndex   — shared uniform grid for this participant
    gap_max_ms    : float              — maximum gap to interpolate (ms); larger → NaN

    Returns
    -------
    pd.Series with uniform_index as its index. Shape: (len(uniform_index),)
    """
    t_start = uniform_index[0]

    # Drop NaN before interpolating.
    # series may have timestamps as its index (DatetimeIndex) while timestamps
    # is a separate Series with a numeric index — .loc would fail.
    # Use a boolean mask applied to both series and timestamps positionally.
    valid_mask = series.notna()
    valid      = series[valid_mask]
    ts_valid   = timestamps[valid_mask.values]  # align by position, not by label

    if len(valid) < 2:
        logger.warning(
            f"  Too few valid samples for '{series.name}' — returning NaN series."
        )
        return pd.Series(np.nan, index=uniform_index, name=series.name)

    ts_seconds      = (ts_valid - t_start).dt.total_seconds().values
    uniform_seconds = (uniform_index - t_start).total_seconds().values

    interp_fn = interp1d(
        ts_seconds, valid.values,
        kind="linear", bounds_error=False, fill_value=np.nan
    )
    resampled = interp_fn(uniform_seconds)

    # Re-introduce NaN for large gaps
    gap_threshold_s = gap_max_ms / 1000.0
    dt_original     = pd.Series(ts_valid.values).diff().dt.total_seconds().fillna(0)
    gap_mask_orig   = dt_original > gap_threshold_s

    if gap_mask_orig.any():
        gap_starts = ts_valid.values[gap_mask_orig.values]
        gap_ends   = ts_valid.values[np.where(gap_mask_orig.values)[0] - 1]
        for g_start, g_end in zip(gap_ends, gap_starts):
            mask = (uniform_index >= g_end) & (uniform_index <= g_start)
            resampled[mask] = np.nan

    return pd.Series(resampled, index=uniform_index, name=series.name)


# ---------------------------------------------------------------------------
# 3. GSR cleaning
# ---------------------------------------------------------------------------

def clean_gsr(
    df: pd.DataFrame,
    cfg: dict,
    uniform_index: pd.DatetimeIndex,
) -> dict:
    """
    Clean and decompose the GSR signal into tonic (SCL) and phasic (SCR) components.

    Pipeline:
        1. Extract valid GSR samples (~18% density — structural, not data loss)
        2. Detect and flag artifacts (abrupt jumps > mean + n_sigma * std of |diff|)
        3. Normalize z-score per subject (controls inter-individual baseline differences)
        4. Decompose into SCL (tonic) and SCR (phasic) via cvxEDA or neurokit2

    Parameters
    ----------
    df            : pd.DataFrame      — Tobii DataFrame with 'timestamp' column
    cfg           : dict              — tobii block from preprocessing.yaml
    uniform_index : pd.DatetimeIndex  — shared uniform grid for this participant

    Returns
    -------
    dict with keys:
        'gsr_raw'       : pd.Series — raw GSR on uniform grid
        'gsr_clean'     : pd.Series — artifact-removed GSR (artifacts → NaN)
        'scl'           : pd.Series — tonic component (Skin Conductance Level)
        'scr'           : pd.Series — phasic component (Skin Conductance Response)
        'artifacts'     : pd.DataFrame — rows flagged as artifacts
        'artifact_thr'  : float — threshold used for artifact detection (µS)
        'n_artifacts'   : int
        'has_gsr'       : bool — False when no valid GSR samples exist (sensor disconnected)
    """
    cols       = cfg["columns"]
    gsr_cfg    = cfg["gsr"]
    gap_max_ms = cfg["gap_interpolation_max_ms"]
    gsr_col    = cols["gsr"]

    logger.info("  Processing GSR...")

    # Extract valid samples
    gsr_raw = df[["timestamp", gsr_col]].dropna(subset=[gsr_col]).copy()
    gsr_raw.columns = ["timestamp", "gsr_raw"]
    n_valid = len(gsr_raw)
    logger.info(
        f"    Valid GSR samples: {n_valid:,} / {len(df):,} "
        f"({100*n_valid/len(df):.1f}%)"
    )

    # Edge case: sensor completely disconnected (P4, P7 pattern)
    if n_valid < 2:
        logger.warning(
            "    No valid GSR samples — sensor likely disconnected. "
            "Returning NaN series. This participant may need to be excluded."
        )
        nan_series = pd.Series(np.nan, index=uniform_index)
        return {
            "gsr_raw":      nan_series.rename("gsr_raw"),
            "gsr_clean":    nan_series.rename("gsr_clean"),
            "scl":          nan_series.rename("scl"),
            "scr":          nan_series.rename("scr"),
            "artifacts":    pd.DataFrame(),
            "artifact_thr": np.nan,
            "n_artifacts":  0,
            "has_gsr":      False,
        }

    # Artifact detection: abrupt jumps (sensor disconnection)
    gsr_diff      = gsr_raw["gsr_raw"].diff().abs()
    n_sigma       = gsr_cfg["artifact_sigma_threshold"]
    artifact_thr  = gsr_diff.mean() + n_sigma * gsr_diff.std()
    artifact_mask = gsr_diff > artifact_thr
    artifacts     = gsr_raw[artifact_mask]
    n_artifacts   = len(artifacts)
    logger.info(
        f"    Artifact threshold ({n_sigma}σ): {artifact_thr:.4f} µS | "
        f"Artifacts detected: {n_artifacts}"
    )

    # Replace artifacts with NaN
    gsr_clean = gsr_raw["gsr_raw"].copy()
    gsr_clean[artifact_mask] = np.nan

    # Z-score normalization per subject
    if gsr_cfg["normalize_per_subject"]:
        valid_vals = gsr_clean.dropna()
        if len(valid_vals) > 1:
            gsr_clean = (gsr_clean - valid_vals.mean()) / valid_vals.std()
            logger.info("    Z-score normalization applied.")

    # Resample to shared uniform grid
    gsr_uniform = resample_to_uniform_grid(
        pd.Series(gsr_clean.values, index=gsr_raw["timestamp"], name="gsr_raw"),
        gsr_raw["timestamp"],
        uniform_index,
        gap_max_ms,
    )

    # Decompose into SCL / SCR
    method = gsr_cfg["decomposition_method"]
    scl, scr = _decompose_gsr(gsr_uniform, sfreq=cfg["sfreq_target"], method=method)

    return {
        "gsr_raw":      gsr_uniform,
        "gsr_clean":    gsr_uniform,
        "scl":          scl,
        "scr":          scr,
        "artifacts":    artifacts,
        "artifact_thr": artifact_thr,
        "n_artifacts":  n_artifacts,
        "has_gsr":      True,
    }


def _decompose_gsr(
    gsr_uniform: pd.Series,
    sfreq: float,
    method: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Decompose normalized GSR into tonic (SCL) and phasic (SCR) components.

    Important: cvxEDA and neurokit operate only on non-NaN samples. We extract
    valid samples, decompose, then reproject back onto the full uniform_index.
    This ensures SCL and SCR always have the same length as gsr_uniform.

    Parameters
    ----------
    gsr_uniform : pd.Series — normalized GSR on uniform time grid (NaN for gaps)
    sfreq       : float     — sampling frequency of gsr_uniform (Hz)
    method      : str       — "cvxeda" | "neurokit"

    Returns
    -------
    scl : pd.Series — tonic component, same index as gsr_uniform
    scr : pd.Series — phasic component, same index as gsr_uniform
    """
    idx = gsr_uniform.index
    n   = len(gsr_uniform)

    # Extract only valid (non-NaN) samples for decomposition
    valid_mask   = gsr_uniform.notna()
    gsr_valid    = gsr_uniform[valid_mask].values
    valid_indices = np.where(valid_mask.values)[0]

    if len(gsr_valid) < 10:
        logger.warning("  Too few valid GSR samples for decomposition — returning NaN.")
        return (
            pd.Series(np.nan, index=idx, name="scl"),
            pd.Series(np.nan, index=idx, name="scr"),
        )

    # Run decomposition on valid samples only
    if method == "cvxeda":
        try:
            import cvxeda
            yn = gsr_valid.reshape(-1, 1)
            [scr_vals, _, scl_vals, _, _, _, _] = cvxeda.cvxEDA(yn, 1.0 / sfreq)
            scl_valid = scl_vals.flatten()
            scr_valid = scr_vals.flatten()
            logger.info("    GSR decomposed via cvxEDA.")
        except ImportError:
            logger.warning(
                "    cvxeda not installed — falling back to neurokit2. "
                "Install with: pip install cvxeda"
            )
            scl_valid, scr_valid = _decompose_gsr_neurokit_values(gsr_valid, sfreq)
    elif method == "neurokit":
        scl_valid, scr_valid = _decompose_gsr_neurokit_values(gsr_valid, sfreq)
    else:
        raise ValueError(f"Unknown GSR decomposition method: '{method}'.")

    # Reproject back onto the full uniform index (NaN where original was NaN)
    scl_full = np.full(n, np.nan)
    scr_full = np.full(n, np.nan)

    # Guard against length mismatch from decomposition algorithm
    out_len = min(len(scl_valid), len(valid_indices))
    scl_full[valid_indices[:out_len]] = scl_valid[:out_len]
    scr_full[valid_indices[:out_len]] = scr_valid[:out_len]

    return (
        pd.Series(scl_full, index=idx, name="scl"),
        pd.Series(scr_full, index=idx, name="scr"),
    )


def _decompose_gsr_neurokit_values(
    gsr_values: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run neurokit2 EDA decomposition on a clean 1D numpy array."""
    import neurokit2 as nk
    eda_signals, _ = nk.eda_process(gsr_values, sampling_rate=int(sfreq))
    return (
        eda_signals["EDA_Tonic"].values,
        eda_signals["EDA_Phasic"].values,
    )


# ---------------------------------------------------------------------------
# 4. Pupil cleaning
# ---------------------------------------------------------------------------

def clean_pupil(
    df: pd.DataFrame,
    cfg: dict,
    uniform_index: pd.DatetimeIndex,
) -> dict:
    """
    Clean the pupil diameter signal.

    Pipeline:
        1. Select preferred column (filtered > left > right)
        2. Detect blinks by abrupt diameter drop (> threshold% in < max_duration_ms)
        3. Interpolate blink windows with cubic spline
        4. Z-score normalize per subject

    Parameters
    ----------
    df            : pd.DataFrame      — Tobii DataFrame with 'timestamp' column
    cfg           : dict              — tobii block from preprocessing.yaml
    uniform_index : pd.DatetimeIndex  — shared uniform grid for this participant

    Returns
    -------
    dict with keys:
        'pupil_raw'    : pd.Series — raw pupil on uniform grid (with NaN for blinks)
        'pupil_clean'  : pd.Series — blink-interpolated and normalized pupil
        'blink_mask'   : pd.Series[bool] — True where blink was detected
        'n_blinks'     : int
        'pct_blinks'   : float — percentage of samples flagged as blinks
    """
    cols       = cfg["columns"]
    pupil_cfg  = cfg["pupil"]
    gap_max_ms = cfg["gap_interpolation_max_ms"]

    logger.info("  Processing pupil...")

    # Select preferred column
    preferred = pupil_cfg["preferred_column"]
    col_map   = {
        "filtered": cols["pupil_filtered"],
        "left":     cols["pupil_left"],
        "right":    cols["pupil_right"],
    }
    pupil_col = col_map.get(preferred, cols["pupil_filtered"])
    if pupil_col not in df.columns:
        fallback = cols["pupil_left"]
        logger.warning(
            f"    Column '{pupil_col}' not found — falling back to '{fallback}'"
        )
        pupil_col = fallback

    pct_nan = df[pupil_col].isna().mean() * 100
    logger.info(
        f"    Using column: '{pupil_col}' | NaN: {pct_nan:.1f}% "
        f"(parpadeos + tracker loss)"
    )

    # Resample raw signal to shared uniform grid
    pupil_uniform = resample_to_uniform_grid(
        df[pupil_col].rename("pupil"),
        df["timestamp"],
        uniform_index,
        gap_max_ms,
    )

    # Blink detection: abrupt drop > threshold% in < max_duration_ms
    thr_pct    = pupil_cfg["blink_detection_threshold_pct"] / 100.0
    max_dur_ms = pupil_cfg["blink_max_duration_ms"]
    dt_ms      = 1000.0 / cfg["sfreq_target"]  # ms per sample on uniform grid

    pct_drop   = pupil_uniform.pct_change(fill_method=None).abs()
    blink_mask = pct_drop > thr_pct
    n_blinks   = blink_mask.sum()
    pct_blinks = 100.0 * n_blinks / max(len(pupil_uniform), 1)
    logger.info(
        f"    Blinks detected: {n_blinks} ({pct_blinks:.2f}% of samples)"
    )

    # Cubic interpolation over blink windows
    interp_method = pupil_cfg["interpolation_method"]
    pupil_interp  = _interpolate_blinks(pupil_uniform, blink_mask, interp_method)

    # Z-score normalization per subject
    if pupil_cfg["normalize_per_subject"]:
        valid = pupil_interp.dropna()
        if len(valid) > 1:
            pupil_interp = (pupil_interp - valid.mean()) / valid.std()
            logger.info("    Z-score normalization applied.")

    return {
        "pupil_raw":   pupil_uniform,
        "pupil_clean": pupil_interp,
        "blink_mask":  blink_mask,
        "n_blinks":    int(n_blinks),
        "pct_blinks":  pct_blinks,
    }


def _interpolate_blinks(
    signal: pd.Series,
    blink_mask: pd.Series,
    method: str = "cubic",
) -> pd.Series:
    """
    Interpolate detected blink windows using the specified method.

    Sets blink samples to NaN, then interpolates using pandas method.
    Samples that are NaN for other reasons (genuine tracker loss) are left as NaN.

    Parameters
    ----------
    signal     : pd.Series — pupil signal on uniform grid
    blink_mask : pd.Series[bool] — True where blink was detected
    method     : str — interpolation method ('cubic', 'linear', 'quadratic')

    Returns
    -------
    pd.Series — signal with blink windows interpolated
    """
    result = signal.copy()
    result[blink_mask] = np.nan
    result = result.interpolate(method=method, limit_direction="both")
    return result


# ---------------------------------------------------------------------------
# 5. Eye tracking extraction
# ---------------------------------------------------------------------------

def extract_eye_tracking(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Extract fixation and saccade events from the Tobii IVT classification.

    The Tobii Pro already classifies each sample as Fixation / Saccade /
    EyesNotFound / Unclassified. We extract event-level summaries:
        - Fixations: centroid coordinates, duration (from Gaze event duration)
        - Saccades: duration (proxy for amplitude given no velocity data available)

    Note: Fixation duration < min_fixation_ms are filtered out as spurious
    tracker instabilities (Salvucci & Goldberg, 2000).

    Parameters
    ----------
    df  : pd.DataFrame — Tobii DataFrame with 'timestamp' column
    cfg : dict         — tobii block from preprocessing.yaml

    Returns
    -------
    dict with keys:
        'fixations'    : pd.DataFrame — one row per fixation event
        'saccades'     : pd.DataFrame — one row per saccade event
        'movement_dist': pd.Series — value_counts of Eye movement type
        'n_fixations'  : int
        'n_saccades'   : int
        'pct_eyes_not_found': float — % samples where tracker lost the eye
    """
    cols     = cfg["columns"]
    et_cfg   = cfg["eyetracking"]

    movement_col  = cols["eye_movement_type"]
    duration_col  = cols["gaze_event_duration"]
    fix_x_col     = cols["fixation_point_x"]
    fix_y_col     = cols["fixation_point_y"]
    evt_idx_col   = cols["eye_movement_type_index"]

    logger.info("  Processing eye tracking...")

    movement_dist      = df[movement_col].value_counts(dropna=False)
    pct_eyes_not_found = 100.0 * (df[movement_col] == "EyesNotFound").sum() / len(df)
    logger.info(f"    Movement type distribution:\n{movement_dist.to_string()}")
    logger.info(f"    EyesNotFound: {pct_eyes_not_found:.1f}% of samples")

    min_fix_ms = et_cfg["min_fixation_ms"]

    # ── Fixations ──
    fix_mask = df[movement_col] == "Fixation"
    fix_df   = (
        df[fix_mask][[
            "timestamp", evt_idx_col, duration_col, fix_x_col, fix_y_col
        ]]
        .dropna(subset=[duration_col])
        .copy()
    )

    # One row per fixation event (group by event index)
    fixations = (
        fix_df.groupby(evt_idx_col)
        .agg(
            onset        = ("timestamp",    "first"),
            duration_ms  = (duration_col,   "first"),   # Gaze event duration is constant per event
            fixation_x   = (fix_x_col,      "mean"),
            fixation_y   = (fix_y_col,      "mean"),
            n_samples    = ("timestamp",    "count"),
        )
        .reset_index(drop=True)
    )

    # Filter spurious short fixations
    fixations = fixations[fixations["duration_ms"] >= min_fix_ms].reset_index(drop=True)
    logger.info(
        f"    Fixations: {len(fixations):,} "
        f"(median dur: {fixations['duration_ms'].median():.0f} ms, "
        f"after >{min_fix_ms} ms filter)"
    )

    # ── Saccades ──
    sac_mask = df[movement_col] == "Saccade"
    sac_df   = (
        df[sac_mask][["timestamp", evt_idx_col, duration_col]]
        .dropna(subset=[duration_col])
        .copy()
    )

    saccades = (
        sac_df.groupby(evt_idx_col)
        .agg(
            onset       = ("timestamp",  "first"),
            duration_ms = (duration_col, "first"),
            n_samples   = ("timestamp",  "count"),
        )
        .reset_index(drop=True)
    )
    logger.info(
        f"    Saccades: {len(saccades):,} "
        f"(median dur: {saccades['duration_ms'].median():.0f} ms)"
    )

    return {
        "fixations":           fixations,
        "saccades":            saccades,
        "movement_dist":       movement_dist,
        "n_fixations":         len(fixations),
        "n_saccades":          len(saccades),
        "pct_eyes_not_found":  pct_eyes_not_found,
    }


# ---------------------------------------------------------------------------
# 6. Report generation
# ---------------------------------------------------------------------------

def save_tobii_reports(
    participant_name: str,
    gsr_result:  dict,
    pupil_result: dict,
    et_result:   dict,
    report_dirs: dict,
) -> None:
    """
    Save QC figures and tables for a single participant to reports/preprocessing/.

    Figures saved (reports/preprocessing/figures/):
        - tobii_gsr_{participant}.png    : GSR raw + SCL + SCR
        - tobii_pupil_{participant}.png  : pupil raw vs clean + blink detection
        - tobii_gaze_{participant}.png   : fixation duration distribution + gaze density

    Tables saved (reports/preprocessing/qc/):
        - tobii_qc_{participant}.csv     : per-participant QC summary row
    """
    figures_dir = Path(report_dirs["figures"])
    qc_dir      = Path(report_dirs["qc"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ── GSR figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    gsr_raw = gsr_result["gsr_raw"].dropna()
    scl     = gsr_result["scl"].dropna()
    scr     = gsr_result["scr"].dropna()

    axes[0].plot(gsr_raw.index, gsr_raw.values, lw=0.6, color="steelblue")
    axes[0].set_title("GSR normalizado (z-score)")
    axes[0].set_ylabel("Conductancia (z)")

    axes[1].plot(scl.index, scl.values, lw=0.8, color="#2ecc71")
    axes[1].set_title("SCL — componente tónico (arousal sostenido)")
    axes[1].set_ylabel("Amplitud (z)")

    axes[2].plot(scr.index, scr.values, lw=0.6, color="#e74c3c")
    axes[2].set_title("SCR — componente fásico (reactividad simpática)")
    axes[2].set_ylabel("Amplitud (z)")
    axes[2].set_xlabel("Tiempo")

    fig.suptitle(
        f"GSR — {participant_name} | "
        f"artefactos: {gsr_result['n_artifacts']} "
        f"(thr={gsr_result['artifact_thr']:.4f} µS)",
        fontsize=11,
    )
    fig.tight_layout()
    out = figures_dir / f"tobii_gsr_{participant_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  GSR figure → {out}")

    # ── Pupil figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    p_raw   = pupil_result["pupil_raw"].dropna()
    p_clean = pupil_result["pupil_clean"].dropna()
    blinks  = pupil_result["blink_mask"]

    axes[0].plot(p_raw.index, p_raw.values, lw=0.5, color="steelblue",
                 label="Raw (con NaN parpadeos)")
    blink_idx = blinks[blinks].index
    if len(blink_idx) > 0:
        blink_y = float(p_raw.dropna().mean()) if not p_raw.dropna().empty else 0.0
        axes[0].scatter(blink_idx, [blink_y] * len(blink_idx),
                        color="red", s=8, zorder=5, label=f"Parpadeo ({len(blink_idx)})")
    axes[0].set_title("Pupila — señal raw")
    axes[0].set_ylabel("Diámetro (mm)")
    axes[0].legend(fontsize=8)

    axes[1].plot(p_clean.index, p_clean.values, lw=0.5, color="#2ecc71")
    axes[1].set_title("Pupila — post interpolación + normalización z-score")
    axes[1].set_ylabel("Diámetro (z)")
    axes[1].set_xlabel("Tiempo")

    fig.suptitle(
        f"Pupilometría — {participant_name} | "
        f"parpadeos: {pupil_result['n_blinks']} ({pupil_result['pct_blinks']:.2f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    out = figures_dir / f"tobii_pupil_{participant_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Pupil figure → {out}")

    # ── Eye tracking figure ──────────────────────────────────────────────────
    fixations = et_result["fixations"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if len(fixations) > 0:
        p99 = fixations["duration_ms"].quantile(0.99)
        fixations[fixations["duration_ms"] < p99]["duration_ms"].hist(
            bins=50, ax=axes[0], color="steelblue", edgecolor="white"
        )
        axes[0].axvline(
            fixations["duration_ms"].median(), color="red", linestyle="--",
            label=f"Mediana: {fixations['duration_ms'].median():.0f} ms"
        )
        axes[0].set_xlabel("Duración fijación (ms)")
        axes[0].set_title("Distribución duraciones de fijación (P99)")
        axes[0].legend()

        hb = axes[1].hexbin(
            fixations["fixation_x"].dropna(),
            fixations["fixation_y"].dropna(),
            gridsize=40, cmap="YlOrRd", mincnt=1,
        )
        axes[1].invert_yaxis()
        axes[1].set_xlabel("X (px)"); axes[1].set_ylabel("Y (px)")
        axes[1].set_title("Mapa de densidad de fijaciones")
        plt.colorbar(hb, ax=axes[1], label="N fijaciones")

    fig.suptitle(
        f"Eye Tracking — {participant_name} | "
        f"fijaciones: {et_result['n_fixations']:,} | "
        f"sacadas: {et_result['n_saccades']:,}",
        fontsize=11,
    )
    fig.tight_layout()
    out = figures_dir / f"tobii_eyetracking_{participant_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Eye tracking figure → {out}")

    # ── QC summary row ───────────────────────────────────────────────────────
    qc_row = pd.DataFrame([{
        "participant":        participant_name,
        "gsr_n_artifacts":    gsr_result["n_artifacts"],
        "gsr_artifact_thr":   round(gsr_result["artifact_thr"], 5),
        "pupil_n_blinks":     pupil_result["n_blinks"],
        "pupil_pct_blinks":   round(pupil_result["pct_blinks"], 3),
        "et_n_fixations":     et_result["n_fixations"],
        "et_n_saccades":      et_result["n_saccades"],
        "et_pct_eyes_lost":   round(et_result["pct_eyes_not_found"], 2),
    }])

    qc_path = qc_dir / f"tobii_qc_{participant_name}.csv"
    qc_row.to_csv(qc_path, index=False)
    logger.info(f"  QC table → {qc_path}")


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------

def export_clean_dataframe(
    df: pd.DataFrame,
    gsr_result:   dict,
    pupil_result: dict,
    et_result:    dict,
    cfg: dict,
) -> pd.DataFrame:
    """
    Assemble the cleaned Tobii signals into a single DataFrame on a uniform time grid.

    Output columns:
        timestamp    : datetime64 — uniform 60 Hz grid
        gsr_raw      : float — normalized GSR (z-score)
        scl          : float — tonic GSR component
        scr          : float — phasic GSR component
        pupil_raw    : float — raw pupil diameter (mm, NaN at blinks)
        pupil_clean  : float — blink-interpolated normalized pupil (z-score)

    Fixations and saccades are NOT included in this DataFrame — they are
    event-based (variable length) and are returned separately in et_result.

    Parameters
    ----------
    df           : pd.DataFrame — original Tobii DataFrame (for timestamp reference)
    gsr_result   : dict — output of clean_gsr()
    pupil_result : dict — output of clean_pupil()
    et_result    : dict — output of extract_eye_tracking()
    cfg          : dict — tobii block from preprocessing.yaml

    Returns
    -------
    pd.DataFrame — shape: (n_samples_uniform, 6)
    """
    # All signals share the same uniform_index built in preprocess_participant.
    # Assign values as numpy arrays — avoids any index alignment issues.
    uniform_idx = gsr_result["gsr_raw"].index

    df_out = pd.DataFrame(index=uniform_idx)
    df_out.index.name = "timestamp"

    df_out["gsr_raw"]     = gsr_result["gsr_raw"].values
    df_out["scl"]         = gsr_result["scl"].values
    df_out["scr"]         = gsr_result["scr"].values
    df_out["pupil_raw"]   = pupil_result["pupil_raw"].values
    df_out["pupil_clean"] = pupil_result["pupil_clean"].values

    df_out = df_out.reset_index()

    logger.info(
        f"  Export shape: {df_out.shape} | "
        f"duration: {(df_out['timestamp'].max() - df_out['timestamp'].min()).total_seconds():.1f} s"
    )
    return df_out


# ---------------------------------------------------------------------------
# 8. High-level entry points
# ---------------------------------------------------------------------------

def preprocess_participant(
    filepath: Path,
    cfg: dict,
    output_dir:   Optional[Path] = None,
    report_dirs:  Optional[dict] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Full Tobii preprocessing pipeline for a single participant.

    Parameters
    ----------
    filepath    : Path — interim .parquet Tobii file
    cfg         : dict — tobii block from preprocessing.yaml
    output_dir  : Path — where to save cleaned .parquet (required if save=True)
    report_dirs : dict — {'figures': Path, 'qc': Path} (optional)
    save        : bool — whether to persist outputs to disk

    Returns
    -------
    df_signals   : pd.DataFrame — continuous signals (gsr, scl, scr, pupil) on uniform grid
    df_fixations : pd.DataFrame — fixation events
    df_saccades  : pd.DataFrame — saccade events
    qc_summary   : dict — quality metrics for this participant
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {filepath.name}")
    participant_name = filepath.stem

    df_raw = pd.read_parquet(filepath)
    logger.info(f"  Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")

    df        = reconstruct_timestamps(df_raw, cfg)

    # Build shared uniform time grid ONCE — all signals must use the same index
    # to guarantee identical Series indices for export_clean_dataframe()
    uniform_index = build_uniform_index(df["timestamp"], cfg["sfreq_target"])
    logger.info(f"  Uniform grid: {len(uniform_index):,} samples @ {cfg['sfreq_target']} Hz")

    gsr_res   = clean_gsr(df, cfg, uniform_index)
    pupil_res = clean_pupil(df, cfg, uniform_index)
    et_res    = extract_eye_tracking(df, cfg)
    df_out    = export_clean_dataframe(df, gsr_res, pupil_res, et_res, cfg)

    if report_dirs is not None:
        save_tobii_reports(
            participant_name, gsr_res, pupil_res, et_res, report_dirs
        )

    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save=True")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Continuous signals
        signals_path = output_dir / filepath.name
        df_out.to_parquet(signals_path, index=False)

        # Fixations and saccades — saved separately (event-based, variable length)
        fix_path = output_dir / filepath.name.replace(".parquet", "_fixations.parquet")
        sac_path = output_dir / filepath.name.replace(".parquet", "_saccades.parquet")
        et_res["fixations"].to_parquet(fix_path, index=False)
        et_res["saccades"].to_parquet(sac_path, index=False)

        logger.info(f"  Saved signals  → {signals_path}")
        logger.info(f"  Saved fixations→ {fix_path}")
        logger.info(f"  Saved saccades → {sac_path}")

    qc = {
        "participant":      participant_name,
        "gsr_n_artifacts":  gsr_res["n_artifacts"],
        "pupil_n_blinks":   pupil_res["n_blinks"],
        "pupil_pct_blinks": pupil_res["pct_blinks"],
        "et_n_fixations":   et_res["n_fixations"],
        "et_n_saccades":    et_res["n_saccades"],
        "et_pct_eyes_lost": et_res["pct_eyes_not_found"],
    }

    return df_out, et_res["fixations"], et_res["saccades"], qc


def preprocess_all(
    tobii_interim_dir: Path,
    tobii_processed_dir: Path,
    cfg: dict,
    report_dirs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run the Tobii pipeline over all participant files.

    Parameters
    ----------
    tobii_interim_dir   : Path — directory with interim .parquet files
    tobii_processed_dir : Path — output directory for cleaned files
    cfg                 : dict — tobii block from preprocessing.yaml
    report_dirs         : dict — optional report output directories

    Returns
    -------
    pd.DataFrame — QC summary table, shape: (n_participants, n_metrics)
    """
    tobii_files = sorted(tobii_interim_dir.glob("*.parquet"))
    if not tobii_files:
        raise FileNotFoundError(f"No .parquet files in {tobii_interim_dir}")

    logger.info(f"Found {len(tobii_files)} Tobii files to process.")

    qc_rows = []
    failed  = []

    for filepath in tobii_files:
        try:
            _, _, _, qc = preprocess_participant(
                filepath=filepath,
                cfg=cfg,
                output_dir=tobii_processed_dir,
                report_dirs=report_dirs,
                save=True,
            )
            qc_rows.append(qc)
        except Exception as e:
            logger.error(f"FAILED: {filepath.name} — {e}")
            failed.append(filepath.name)

    if failed:
        logger.warning(f"{len(failed)} files failed: {failed}")

    if not qc_rows:
        logger.error("All participants failed — no QC summary to build.")
        return pd.DataFrame()

    summary = pd.DataFrame(qc_rows).set_index("participant")

    if report_dirs is not None:
        qc_dir = Path(report_dirs["qc"])
        qc_dir.mkdir(parents=True, exist_ok=True)
        summary_path = qc_dir / "tobii_qc_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"Global QC summary → {summary_path}")

    logger.info(
        f"Tobii preprocessing complete: {len(qc_rows)} succeeded, {len(failed)} failed."
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Tobii Preprocessing Pipeline")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--paths-config",          default="configs/paths.yaml")
    parser.add_argument(
        "--participant", default=None,
        help="Single participant filename. If omitted, processes all."
    )
    parser.add_argument(
        "--no-reports", action="store_true",
        help="Skip saving figures and QC reports."
    )
    args = parser.parse_args()

    paths    = load_config(args.paths_config)
    tobii_cfg = load_config(args.preprocessing_config)["tobii"]

    tobii_interim   = Path(paths["data"]["interim"]["tobii"])
    tobii_processed = Path(paths["data"]["processed"]["tobii"])

    report_dirs = None if args.no_reports else {
        "figures": paths["reports"]["preprocessing"]["figures"],
        "qc":      paths["reports"]["preprocessing"]["qc"],
    }

    if args.participant:
        filepath = tobii_interim / args.participant
        preprocess_participant(
            filepath, cfg=tobii_cfg,
            output_dir=tobii_processed,
            report_dirs=report_dirs, save=True
        )
    else:
        summary = preprocess_all(
            tobii_interim, tobii_processed,
            cfg=tobii_cfg, report_dirs=report_dirs
        )
        print("\n=== Tobii QC Summary ===")
        print(summary.to_string())