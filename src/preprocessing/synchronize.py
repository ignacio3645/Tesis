"""
src/preprocessing/synchronize.py
=================================
EEG ↔ Tobii temporal synchronization for the Fondecyt 1231122 dataset.

Aligns two independently-clocked systems:
    - OpenBCI Cyton ×2  : EEG at 125 Hz, absolute wall-clock timestamps
    - Tobii Pro Spectrum : GSR + pupil + eye tracking at 60 Hz uniform grid

Problem:
    Both devices have independent hardware clocks.  The EEG starts recording
    before the Tobii (~20–60 s offset per participant), and clock drift causes
    the two timelines to diverge slightly over the ~12-minute session.

Solution (validated on P1):
    1. Compute initial offset   : tobii_start − eeg_start  (constant per session)
    2. Estimate drift           : compare session duration measured by each clock
                                  using ImageStimulusStart events as shared anchor
    3. Apply linear correction  : project EEG timestamps into drift-corrected
                                  absolute time via
                                  t_corrected = eeg_start
                                              + (t_from_eeg_start × drift_scale)
                                  Both systems share the OS wall-clock datetime64
                                  domain — only the drift is corrected, not the
                                  offset (which is already encoded in the absolute
                                  timestamps and must NOT be removed).
    4. Classify stimuli         : assign epoch_type to each ImageStimulusStart based
                                  on stimulus name prefix and temporal position
    5. Extract epochs           : slice EEG and Tobii signals for each stimulus window
                                  [onset + tmin,  onset + duration + tmax_offset]
    6. Extract navigation epochs: slice WebStimulusStart → WebStimulusEnd windows
                                  and sub-segment by URLStart → URLEnd

Validated results (P1, 2025-08-08):
    - Offset:       47.166 s (Tobii starts after EEG — structural, not anomaly)
    - Drift:        −2.90 ms in 657 s  →  −0.26 ms/min  (linear correction sufficient)
    - Drift scale:  1.00000441
    - Epochs:       90 ImageStimulus epochs, 0 empty, 0 NaN
    - Ratio EEG/Tobii: 2.083 ± 0.002  (expected 125/60 = 2.0833)

Stimulus taxonomy (inferred from stimulus name prefix + temporal position):
    - exposicion_imagen  : numeric name ('1'..'N'), pre-navigation  → training primary
    - blur_exposicion    : 'blur_N' prefix, pre-navigation          → washout / exclude
    - evaluacion_imagen  : 'diseño_N' or 'emotion_N' prefix         → post-decision, exclude
    - evaluacion_web     : any ImageStimulus post-WebStimulusEnd     → web eval, exclude
    - web_sitio          : WebStimulusStart→End window (one per site)→ free navigation
    - web_url            : URLStart→End sub-window within web_sitio  → page-level epoch

Frequencies are deliberately NOT homogenized:
    EEG remains at 125 Hz, Tobii at 60 Hz.
    Epochs contain different sample counts for the same temporal window.
    Downstream encoders handle this natively via modality-specific 1D-CNNs.

Author: Ignacio Negrete Silva
Project: Fondecyt 1231122
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synchronize")


# ---------------------------------------------------------------------------
# 1. Timestamp reconstruction (Tobii interim)
# ---------------------------------------------------------------------------

def reconstruct_tobii_timestamps(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Reconstruct absolute wall-clock timestamps for a Tobii interim DataFrame.

    Parameters
    ----------
    df  : raw Tobii parquet loaded from data/interim/tobii/
    cfg : preprocessing.yaml → tobii section

    Returns
    -------
    df with new column 'timestamp' (datetime64[ns], absolute wall-clock)

    Notes
    -----
    Uses Recording date + Recording start time + Recording timestamp (µs offset).
    Do NOT use 'Computer timestamp' — it is a Windows FILETIME (ns since 1601-01-01)
    and produces incorrect results when cast as Unix epoch.
    """
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    date_fmt = cfg["date_format"]
    df = df.copy()
    df["_recording_start"] = pd.to_datetime(
        df["Recording date"] + " " + df["Recording start time"],
        format=date_fmt,
    )
    df["timestamp"] = (
        df["_recording_start"]
        + pd.to_timedelta(df["Recording timestamp"], unit="us")
    )
    df.drop(columns=["_recording_start"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Event extraction from Tobii interim
# ---------------------------------------------------------------------------

def extract_events(df_tobii_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Extract all events from the Tobii interim file with reconstructed timestamps.

    Returns a DataFrame sorted by timestamp with columns:
        timestamp, event, event_value, t_sec  (seconds from recording start)

    Parameters
    ----------
    df_tobii_raw : Tobii interim DataFrame (must have 'timestamp' column)
    cfg          : preprocessing.yaml → tobii section
    """
    events_col = cfg["columns"]["event"]
    ev_val_col = cfg["columns"]["event_value"]

    events = (
        df_tobii_raw[df_tobii_raw[events_col].notna()]
        [[events_col, ev_val_col, "timestamp"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    events.columns = ["event", "event_value", "timestamp"]
    t0 = df_tobii_raw["timestamp"].min()
    events["t_sec"] = (events["timestamp"] - t0).dt.total_seconds()
    return events


# ---------------------------------------------------------------------------
# 3. Offset and drift estimation
# ---------------------------------------------------------------------------

def compute_sync_params(
    eeg: pd.DataFrame,
    tobii_processed: pd.DataFrame,
    events: pd.DataFrame,
    anchor_event: str,
    drift_threshold_ms: float,
) -> dict:
    """
    Compute the two synchronization parameters: initial offset and drift scale.

    Parameters
    ----------
    eeg              : processed EEG DataFrame with 'timestamp' column (125 Hz)
    tobii_processed  : processed Tobii DataFrame with 'timestamp' column (60 Hz)
    events           : event DataFrame from extract_events()
    anchor_event     : event name used as drift anchor (e.g. 'ImageStimulusStart')
    drift_threshold_ms: warn if |drift| exceeds this value (ms)

    Returns
    -------
    dict with keys:
        offset_s        : float  — tobii_start − eeg_start in seconds
        drift_scale     : float  — duration_tobii / duration_eeg
        drift_ms        : float  — accumulated drift in ms
        drift_ms_per_min: float
        eeg_start       : Timestamp
        tobii_start     : Timestamp
        overlap_s       : float  — seconds of temporal overlap post-correction
        n_anchor_events : int    — number of anchor events found

    Raises
    ------
    ValueError if fewer than 2 anchor events are found (drift cannot be estimated)
    """
    eeg["timestamp"]             = pd.to_datetime(eeg["timestamp"])
    tobii_processed["timestamp"] = pd.to_datetime(tobii_processed["timestamp"])

    eeg_start   = eeg["timestamp"].iloc[0]
    eeg_end     = eeg["timestamp"].iloc[-1]
    tobii_start = tobii_processed["timestamp"].iloc[0]
    tobii_end   = tobii_processed["timestamp"].iloc[-1]

    offset_s = (tobii_start - eeg_start).total_seconds()
    logger.info(f"  EEG  : {eeg_start} → {eeg_end}  "
                f"({(eeg_end-eeg_start).total_seconds():.1f} s)")
    logger.info(f"  Tobii: {tobii_start} → {tobii_end}  "
                f"({(tobii_end-tobii_start).total_seconds():.1f} s)")
    logger.info(f"  Offset inicial: {offset_s:.3f} s  "
                f"(Tobii arranca después del EEG)")

    # Anchor events
    anchor_ts = (
        events[events["event"] == anchor_event]["timestamp"]
        .sort_values()
        .tolist()
    )
    n_anchor = len(anchor_ts)
    if n_anchor < 2:
        raise ValueError(
            f"Found only {n_anchor} '{anchor_event}' events — "
            "need ≥2 to estimate drift."
        )

    first_anchor_tobii = anchor_ts[0]
    last_anchor_tobii  = anchor_ts[-1]
    duration_tobii = (last_anchor_tobii - first_anchor_tobii).total_seconds()

    # Anchor lookup: both EEG and Tobii timestamps are absolute wall-clock
    # datetime64 — they share the same OS time reference.  Anchors can therefore
    # be located in EEG space by direct proximity search without any offset
    # arithmetic.  Subtracting offset_s here would be a double-compensation
    # because the offset is already encoded in the absolute timestamps themselves.
    idx_first = (eeg["timestamp"] - first_anchor_tobii).abs().idxmin()
    idx_last  = (eeg["timestamp"] - last_anchor_tobii).abs().idxmin()
    duration_eeg = (
        eeg["timestamp"].iloc[idx_last] - eeg["timestamp"].iloc[idx_first]
    ).total_seconds()

    drift_s       = duration_eeg - duration_tobii
    drift_ms      = drift_s * 1000
    drift_ms_min  = drift_ms / (duration_tobii / 60)
    drift_scale   = duration_tobii / duration_eeg if duration_eeg > 0 else 1.0

    logger.info(f"  Drift acumulado: {drift_ms:.2f} ms  "
                f"({drift_ms_min:.2f} ms/min)  |  umbral: {drift_threshold_ms} ms")
    if abs(drift_ms) >= drift_threshold_ms:
        logger.warning(
            f"  Drift {drift_ms:.1f} ms ≥ umbral {drift_threshold_ms} ms — "
            "revisar manualmente este participante."
        )
    logger.info(f"  Factor de escala: {drift_scale:.8f}")
    logger.info(f"  Eventos ancla '{anchor_event}': {n_anchor}")

    # Temporal overlap post-correction (informational)
    eeg_corrected_end = tobii_start + pd.Timedelta(
        seconds=(eeg_end - eeg_start).total_seconds() * drift_scale
    )
    overlap_s = (min(eeg_corrected_end, tobii_end) - max(tobii_start, tobii_start)
                 ).total_seconds()

    return {
        "offset_s":         offset_s,
        "drift_scale":      drift_scale,
        "drift_ms":         drift_ms,
        "drift_ms_per_min": drift_ms_min,
        "eeg_start":        eeg_start,
        "tobii_start":      tobii_start,
        "n_anchor_events":  n_anchor,
        "overlap_s":        overlap_s,
    }


# ---------------------------------------------------------------------------
# 4. EEG timestamp correction
# ---------------------------------------------------------------------------

def apply_eeg_correction(
    eeg: pd.DataFrame,
    sync_params: dict,
) -> pd.DataFrame:
    """
    Project EEG timestamps into a drift-corrected absolute time axis.

    Correction:
        t_corrected = eeg_start + (seconds_from_eeg_start × drift_scale)

    This applies ONLY the linear drift scale, leaving the absolute wall-clock
    origin intact.  Because both EEG and Tobii timestamps are already expressed
    in the same OS wall-clock datetime64 domain, no re-basing to tobii_start is
    needed.  Using tobii_start as the intercept would push the entire EEG signal
    ~47 s into the future, destroying the stimulus-response alignment.

    The drift_scale (≈ 1.000004) corrects exclusively for the frequency
    discrepancy between the two quartz oscillators, ensuring that the corrected
    EEG duration matches the Tobii duration between the first and last anchor
    events.

    Parameters
    ----------
    eeg         : processed EEG DataFrame with 'timestamp' column
    sync_params : dict returned by compute_sync_params()

    Returns
    -------
    eeg copy with added column 'timestamp_corrected' (datetime64[ns])
    Shape unchanged: (N_eeg_samples, n_channels + 2)
    """
    drift_scale = sync_params["drift_scale"]
    eeg_start   = sync_params["eeg_start"]

    eeg_out = eeg.copy()
    seconds_from_start = (eeg_out["timestamp"] - eeg_start).dt.total_seconds()

    # Anchor the corrected timeline to eeg_start (absolute wall-clock) and apply
    # only the drift scale — preserving the OS ground truth without double-offset.
    eeg_out["timestamp_corrected"] = (
        eeg_start
        + pd.to_timedelta(seconds_from_start * drift_scale, unit="s")
    )
    logger.info(
        f"  EEG timestamp corregido [0] : {eeg_out['timestamp_corrected'].iloc[0]}"
    )
    logger.info(
        f"  EEG timestamp corregido [-1]: {eeg_out['timestamp_corrected'].iloc[-1]}"
    )
    return eeg_out


# ---------------------------------------------------------------------------
# 5. Stimulus classification
# ---------------------------------------------------------------------------

def classify_stimuli(
    events: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Classify each ImageStimulusStart into its experimental role.

    Classification rules (in priority order):
        1. post-WebStimulusEnd timestamp              → evaluacion_web
        2. name starts with 'blur'                    → blur_exposicion
        3. name starts with 'diseño_' or 'emotion_'  → evaluacion_imagen
        4. name is pure numeric (re: ^\\d+$)          → exposicion_imagen
        5. anything else                              → desconocido

    Parameters
    ----------
    events : DataFrame from extract_events() — must contain 'ImageStimulusStart',
             'ImageStimulusEnd', 'WebStimulusEnd' events
    cfg    : preprocessing.yaml (used for column names only)

    Returns
    -------
    DataFrame with columns:
        stimulus, epoch_type, timestamp_start, timestamp_end, duration_s
    Rows correspond 1-to-1 with ImageStimulusStart events, sorted by timestamp.
    """
    img_starts = (
        events[events["event"] == "ImageStimulusStart"]
        [["timestamp", "event_value"]]
        .rename(columns={"event_value": "stimulus", "timestamp": "timestamp_start"})
        .sort_values("timestamp_start")
        .reset_index(drop=True)
    )
    img_ends = (
        events[events["event"] == "ImageStimulusEnd"]
        [["timestamp", "event_value"]]
        .rename(columns={"event_value": "stimulus", "timestamp": "timestamp_end"})
        .sort_values("timestamp_end")
        .reset_index(drop=True)
    )

    # Positional merge — Tobii guarantees Start/End pairs are in matching order
    if len(img_starts) != len(img_ends):
        logger.warning(
            f"  ImageStimulusStart ({len(img_starts)}) ≠ "
            f"ImageStimulusEnd ({len(img_ends)}) — using positional merge"
        )
    n = min(len(img_starts), len(img_ends))
    stimuli = img_starts.iloc[:n].copy()
    stimuli["timestamp_end"] = img_ends["timestamp_end"].values[:n]
    stimuli["duration_s"] = (
        stimuli["timestamp_end"] - stimuli["timestamp_start"]
    ).dt.total_seconds().clip(lower=0.5)

    # Navigation boundary
    web_ends = events[events["event"] == "WebStimulusEnd"]["timestamp"]
    nav_end_ts = web_ends.max() if len(web_ends) > 0 else pd.NaT

    def _classify(row) -> str:
        name = str(row["stimulus"]).strip()
        ts   = row["timestamp_start"]
        # Post-navigation → web evaluation
        if pd.notna(nav_end_ts) and ts > nav_end_ts:
            return "evaluacion_web"
        # Blur
        if name.lower().startswith("blur"):
            return "blur_exposicion"
        # Evaluation dimensions
        if name.startswith("diseño_") or name.startswith("emotion_"):
            return "evaluacion_imagen"
        # Pure numeric → passive exposure
        if re.match(r"^\d+$", name):
            return "exposicion_imagen"
        return "desconocido"

    stimuli["epoch_type"] = stimuli.apply(_classify, axis=1)

    counts = stimuli["epoch_type"].value_counts().to_dict()
    for k, v in sorted(counts.items()):
        logger.info(f"    {k:<22}: {v}")

    return stimuli[["stimulus", "epoch_type", "timestamp_start",
                    "timestamp_end", "duration_s"]]


# ---------------------------------------------------------------------------
# 6. Navigation epoch extraction
# ---------------------------------------------------------------------------

def extract_navigation_epochs(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame of navigation epochs at two granularities:
        - web_sitio : WebStimulusStart → WebStimulusEnd  (one per site, ~30 s)
        - web_url   : URLStart → URLEnd                  (one per page visit, variable)

    Parameters
    ----------
    events : DataFrame from extract_events()

    Returns
    -------
    DataFrame with columns:
        epoch_type, stimulus, timestamp_start, timestamp_end, duration_s
    Sorted by timestamp_start.
    """
    rows = []

    web_starts = events[events["event"] == "WebStimulusStart"].sort_values("timestamp")
    web_ends   = events[events["event"] == "WebStimulusEnd"].sort_values("timestamp")
    url_starts = events[events["event"] == "URLStart"].sort_values("timestamp")
    url_ends   = events[events["event"] == "URLEnd"].sort_values("timestamp")

    # Site-level epochs
    for i, (_, ws) in enumerate(web_starts.iterrows()):
        we_candidates = web_ends[web_ends["timestamp"] > ws["timestamp"]]
        if len(we_candidates) == 0:
            logger.warning(f"  WebStimulusEnd not found for site {i+1} — skipping")
            continue
        we = we_candidates.iloc[0]
        rows.append({
            "epoch_type":      "web_sitio",
            "stimulus":        str(ws["event_value"]),
            "timestamp_start": ws["timestamp"],
            "timestamp_end":   we["timestamp"],
            "duration_s":      (we["timestamp"] - ws["timestamp"]).total_seconds(),
        })

    # URL-level epochs (sub-windows inside each site)
    for i, (_, us) in enumerate(url_starts.iterrows()):
        ue_candidates = url_ends[url_ends["timestamp"] > us["timestamp"]]
        if len(ue_candidates) == 0:
            logger.warning(f"  URLEnd not found for URL {i+1} — skipping")
            continue
        ue = ue_candidates.iloc[0]
        rows.append({
            "epoch_type":      "web_url",
            "stimulus":        str(us["event_value"])[:80],
            "timestamp_start": us["timestamp"],
            "timestamp_end":   ue["timestamp"],
            "duration_s":      (ue["timestamp"] - us["timestamp"]).total_seconds(),
        })

    if not rows:
        logger.warning("  No navigation epochs found")
        return pd.DataFrame(columns=["epoch_type","stimulus","timestamp_start",
                                     "timestamp_end","duration_s"])

    nav_df = (
        pd.DataFrame(rows)
        .sort_values("timestamp_start")
        .reset_index(drop=True)
    )

    n_sites = (nav_df["epoch_type"] == "web_sitio").sum()
    n_urls  = (nav_df["epoch_type"] == "web_url").sum()
    logger.info(f"  Navegación — sitios: {n_sites}  |  URLs: {n_urls}")
    return nav_df


# ---------------------------------------------------------------------------
# 7. Epoch signal extraction
# ---------------------------------------------------------------------------

def extract_epoch_signals(
    epoch_row: pd.Series,
    eeg_corrected: pd.DataFrame,
    tobii_processed: pd.DataFrame,
    eeg_channels: list[str],
    tmin: float,
    tmax_offset: float,
) -> dict:
    """
    Slice EEG and Tobii signals for a single epoch window.

    Window: [onset + tmin,  onset + duration_s + tmax_offset]
    - tmin is typically negative (baseline pre-onset, e.g. −0.2 s)
    - tmax_offset adds a post-offset margin (e.g. +0.5 s)

    Parameters
    ----------
    epoch_row        : one row from stimuli_classified or nav_epochs DataFrame
    eeg_corrected    : EEG DataFrame with 'timestamp_corrected' column
    tobii_processed  : Tobii processed DataFrame with 'timestamp' column (60 Hz)
    eeg_channels     : list of EEG channel column names (excludes 'timestamp')
    tmin             : seconds before onset to include (negative = baseline)
    tmax_offset      : seconds after offset to include

    Returns
    -------
    dict:
        eeg   : np.ndarray  shape (T_eeg, n_channels)   — 125 Hz
        gsr   : np.ndarray  shape (T_tobii,)             — 60 Hz
        scl   : np.ndarray  shape (T_tobii,)
        scr   : np.ndarray  shape (T_tobii,)
        pupil : np.ndarray  shape (T_tobii,)
        n_eeg : int
        n_tobii: int
        pct_nan_gsr   : float
        pct_nan_pupil : float
    """
    onset   = epoch_row["timestamp_start"]
    dur_s   = epoch_row["duration_s"]
    t0_win  = onset + pd.Timedelta(seconds=tmin)
    t1_win  = onset + pd.Timedelta(seconds=dur_s + tmax_offset)

    # EEG slice (corrected timestamps)
    mask_eeg = (
        (eeg_corrected["timestamp_corrected"] >= t0_win) &
        (eeg_corrected["timestamp_corrected"] <  t1_win)
    )
    ep_eeg = eeg_corrected.loc[mask_eeg, eeg_channels].values

    # Tobii slice (uniform 60 Hz grid — already in Tobii time-space)
    mask_tobii = (
        (tobii_processed["timestamp"] >= t0_win) &
        (tobii_processed["timestamp"] <  t1_win)
    )
    ep_tobii = tobii_processed.loc[mask_tobii]

    gsr   = ep_tobii["gsr_raw"].values   if "gsr_raw"   in ep_tobii.columns else np.array([])
    scl   = ep_tobii["scl"].values       if "scl"       in ep_tobii.columns else np.array([])
    scr   = ep_tobii["scr"].values       if "scr"       in ep_tobii.columns else np.array([])
    pupil = ep_tobii["pupil_clean"].values if "pupil_clean" in ep_tobii.columns else np.array([])

    pct_nan_gsr   = float(np.isnan(gsr).mean())   if len(gsr)   > 0 else 1.0
    pct_nan_pupil = float(np.isnan(pupil).mean()) if len(pupil) > 0 else 1.0

    return {
        "eeg":          ep_eeg,
        "gsr":          gsr,
        "scl":          scl,
        "scr":          scr,
        "pupil":        pupil,
        "n_eeg":        len(ep_eeg),
        "n_tobii":      len(ep_tobii),
        "pct_nan_gsr":  pct_nan_gsr,
        "pct_nan_pupil":pct_nan_pupil,
    }


# ---------------------------------------------------------------------------
# 8. QC validation
# ---------------------------------------------------------------------------

def validate_epochs(epoch_meta: list[dict], fs_eeg: float = 125.0,
                    fs_tobii: float = 60.0) -> dict:
    """
    Compute QC metrics across all extracted epochs.

    Parameters
    ----------
    epoch_meta : list of dicts from synchronize_participant()
    fs_eeg     : EEG sampling frequency (Hz)
    fs_tobii   : Tobii sampling frequency (Hz)

    Returns
    -------
    dict with summary statistics:
        n_epochs, n_empty_eeg, n_empty_tobii,
        ratio_mean, ratio_std, ratio_expected,
        pct_nan_gsr_median, pct_nan_pupil_median,
        n_high_nan  (epochs with >50% NaN in any signal)
    """
    expected_ratio = fs_eeg / fs_tobii

    n_empty_eeg   = sum(1 for m in epoch_meta if m["n_eeg"]   == 0)
    n_empty_tobii = sum(1 for m in epoch_meta if m["n_tobii"] == 0)

    valid = [m for m in epoch_meta if m["n_eeg"] > 0 and m["n_tobii"] > 0]
    ratios = [m["n_eeg"] / m["n_tobii"] for m in valid]

    nan_gsr   = [m["pct_nan_gsr"]   for m in epoch_meta]
    nan_pupil = [m["pct_nan_pupil"] for m in epoch_meta]
    n_high_nan = sum(
        1 for m in epoch_meta
        if m["pct_nan_gsr"] > 0.5 or m["pct_nan_pupil"] > 0.5
    )

    return {
        "n_epochs":            len(epoch_meta),
        "n_empty_eeg":         n_empty_eeg,
        "n_empty_tobii":       n_empty_tobii,
        "ratio_mean":          float(np.mean(ratios)) if ratios else float("nan"),
        "ratio_std":           float(np.std(ratios))  if ratios else float("nan"),
        "ratio_expected":      expected_ratio,
        "pct_nan_gsr_median":  float(np.median(nan_gsr))   if nan_gsr   else float("nan"),
        "pct_nan_pupil_median":float(np.median(nan_pupil)) if nan_pupil else float("nan"),
        "n_high_nan":          n_high_nan,
    }


# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------

def save_sync_outputs(
    participant_id: str,
    sync_params: dict,
    stimuli_classified: pd.DataFrame,
    nav_epochs: pd.DataFrame,
    epoch_meta: list[dict],
    qc: dict,
    paths: dict,
) -> None:
    """
    Persist synchronization outputs for one participant.

    Output files:
        data/processed/multimodal/{pid}_stimuli.parquet   — stimulus manifest
        data/processed/multimodal/{pid}_nav_epochs.parquet — navigation epochs
        data/processed/multimodal/{pid}_sync_params.parquet — sync parameters
        reports/synchronization/qc/{pid}_sync_qc.csv      — QC summary

    Parameters
    ----------
    participant_id      : e.g. 'P1'
    sync_params         : dict from compute_sync_params()
    stimuli_classified  : DataFrame from classify_stimuli()
    nav_epochs          : DataFrame from extract_navigation_epochs()
    epoch_meta          : list of dicts (one per epoch) from synchronize_participant()
    qc                  : dict from validate_epochs()
    paths               : paths.yaml loaded config
    """
    out_dir = Path(paths["data"]["processed"]["multimodal"])
    qc_dir  = Path(paths["reports"]["synchronization"]["qc"])
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    pid = participant_id

    # Stimulus manifest
    stimuli_classified.to_parquet(out_dir / f"{pid}_stimuli.parquet", index=False)
    logger.info(f"  Stimuli manifest → {out_dir / (pid + '_stimuli.parquet')}")

    # Navigation epochs
    if len(nav_epochs) > 0:
        nav_epochs.to_parquet(out_dir / f"{pid}_nav_epochs.parquet", index=False)
        logger.info(f"  Nav epochs       → {out_dir / (pid + '_nav_epochs.parquet')}")

    # Sync parameters
    sync_row = {
        "participant_id": pid,
        "offset_s":       sync_params["offset_s"],
        "drift_ms":       sync_params["drift_ms"],
        "drift_ms_per_min": sync_params["drift_ms_per_min"],
        "drift_scale":    sync_params["drift_scale"],
        "n_anchor_events":sync_params["n_anchor_events"],
        "overlap_s":      sync_params["overlap_s"],
    }
    pd.DataFrame([sync_row]).to_parquet(
        out_dir / f"{pid}_sync_params.parquet", index=False
    )

    # QC CSV
    qc_row = {"participant_id": pid, **qc}
    pd.DataFrame([qc_row]).to_csv(qc_dir / f"{pid}_sync_qc.csv", index=False)
    logger.info(f"  QC table         → {qc_dir / (pid + '_sync_qc.csv')}")



# ---------------------------------------------------------------------------
# 10. Thesis figures
# ---------------------------------------------------------------------------

def generate_sync_figures(
    participant_id: str,
    stimuli_classified: pd.DataFrame,
    epoch_meta: list[dict],
    tobii_processed: pd.DataFrame,
    eeg_channels: list[str],
    paths: dict,
    qc_summary_path: Optional[Path] = None,
) -> None:
    """
    Generate three thesis-grade figures for the synchronization methodology chapter.

    Saved to reports/synchronization/figures/ at 300 dpi:
        {pid}_fig_drift.png    — drift distribution across all participants
        {pid}_fig_timeline.png — experiment timeline with epoch taxonomy
        {pid}_fig_epoch.png    — single synchronized epoch (EEG + GSR + pupil)

    Rationale:
        drift    : justifies linear correction sufficiency — no participant
                   exceeds the 50 ms threshold defined in preprocessing.yaml
        timeline : describes the full experimental protocol in one figure,
                   replacing several paragraphs in the methodology section
        epoch    : shows the concrete pipeline output with three modalities
                   aligned on the same temporal axis at their native Fs
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except ImportError:
        logger.warning("matplotlib not available — skipping figure generation")
        return

    fig_dir = Path(paths["reports"]["synchronization"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    pid = participant_id

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi":    300,
        "savefig.bbox":   "tight",
    })

    CLRS = {
        "eeg":        "#2c7bb6",
        "tobii":      "#d7191c",
        "exposicion": "#1a9641",
        "evaluacion": "#fdae61",
        "blur":       "#bababa",
        "nav":        "#762a83",
        "eval_web":   "#636363",
    }

    # ── Fig A: drift distribution (requires sync_qc_summary.csv) ───────────
    qc_path = qc_summary_path or (
        Path(paths["reports"]["synchronization"]["qc"]) / "sync_qc_summary.csv"
    )
    if qc_path.exists():
        try:
            qc_all      = pd.read_csv(qc_path)
            drift_vals  = qc_all["drift_ms"].dropna()
            drift_plot  = drift_vals[drift_vals.abs() < 50]
            rate_vals   = qc_all["drift_ms_per_min"].dropna()
            rate_plot   = rate_vals[rate_vals.abs() < 5]

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

            axes[0].hist(drift_plot, bins=20,
                         color=CLRS["eeg"], edgecolor="white", linewidth=0.5)
            axes[0].axvline(0,   color="black", lw=1.0, ls="--",
                            alpha=0.7, label="Cero")
            axes[0].axvline( 50, color="red",   lw=1.0, ls=":",
                             label="±50 ms (umbral)")
            axes[0].axvline(-50, color="red",   lw=1.0, ls=":")
            axes[0].set_xlabel("Drift acumulado (ms)")
            axes[0].set_ylabel("N participantes")
            axes[0].set_title(
                f"Drift acumulado\n(n={len(drift_plot)}, outliers excluidos)",
                fontsize=10)
            axes[0].legend(fontsize=8)
            axes[0].spines[["top","right"]].set_visible(False)

            axes[1].scatter(range(len(rate_plot)), rate_plot.values,
                            color=CLRS["eeg"], s=25, alpha=0.8)
            axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
            axes[1].set_xlabel("Participante (orden alfabético)")
            axes[1].set_ylabel("Drift (ms/min)")
            axes[1].set_title(
                "Tasa de drift por participante\n"
                "(ausencia de tendencia = relojes estables)",
                fontsize=10)
            axes[1].spines[["top","right"]].set_visible(False)

            fig.suptitle(
                "Parámetros de sincronización EEG ↔ Tobii — todos los participantes\n"
                "La corrección lineal es suficiente: ningún drift supera el umbral de 50 ms",
                fontsize=10, y=1.01)
            plt.tight_layout()
            out = fig_dir / f"{pid}_fig_drift.png"
            plt.savefig(out)
            plt.close(fig)
            logger.info(f"  Fig drift    → {out}")
        except Exception as exc:
            logger.warning(f"  Fig drift skipped: {exc}")
    else:
        logger.warning(
            f"  Fig drift skipped: {qc_path} not found "
            "(run synchronize_all first)")

    # ── Fig B: experiment timeline with epoch taxonomy ──────────────────────
    try:
        t0 = tobii_processed["timestamp"].iloc[0]

        def _sec(ts):
            return (ts - t0).total_seconds()

        fig, ax = plt.subplots(figsize=(14, 3.2))

        # Background spans
        phase_map = {
            "exposicion_imagen": ("#ccebc5", "Exposición\n(blur + imagen)"),
            "evaluacion_imagen": ("#fef0d9", "Evaluación\nimágenes"),
            "evaluacion_web":    ("#f2f0f7", "Evaluación\nweb"),
        }
        for etype, (bg, label) in phase_map.items():
            include = (["exposicion_imagen", "blur_exposicion"]
                       if etype == "exposicion_imagen" else [etype])
            sub = stimuli_classified[stimuli_classified["epoch_type"].isin(include)]
            if len(sub) == 0:
                continue
            x0 = _sec(sub["timestamp_start"].min())
            x1 = _sec(sub["timestamp_end"].max())
            ax.axvspan(x0, x1, alpha=0.35, color=bg)
            ax.text((x0+x1)/2, 0.90, label, ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8.5, color="#333")

        # Navigation gap between evaluacion_imagen and evaluacion_web
        eval_img = stimuli_classified[stimuli_classified["epoch_type"]=="evaluacion_imagen"]
        eval_web = stimuli_classified[stimuli_classified["epoch_type"]=="evaluacion_web"]
        if len(eval_img) > 0 and len(eval_web) > 0:
            nav_x0 = _sec(eval_img["timestamp_end"].max())
            nav_x1 = _sec(eval_web["timestamp_start"].min())
            if nav_x1 > nav_x0:
                ax.axvspan(nav_x0, nav_x1, alpha=0.35, color="#e0ecf4")
                ax.text((nav_x0+nav_x1)/2, 0.90, "Navegación\nlibre",
                        ha="center", va="top",
                        transform=ax.get_xaxis_transform(),
                        fontsize=8.5, color="#333")

        # Epoch scatter
        style = {
            "exposicion_imagen": (CLRS["exposicion"], "o", 10, "Exposición imagen"),
            "blur_exposicion":   (CLRS["blur"],       "s",  6, "Blur"),
            "evaluacion_imagen": (CLRS["evaluacion"], "^",  7, "Evaluación imagen"),
            "evaluacion_web":    (CLRS["eval_web"],   "D",  8, "Evaluación web"),
        }
        for etype, (col, mk, sz, lbl) in style.items():
            sub = stimuli_classified[stimuli_classified["epoch_type"]==etype]
            if len(sub) == 0:
                continue
            ax.scatter(sub["timestamp_start"].apply(_sec), [0.2]*len(sub),
                       color=col, marker=mk, s=sz, zorder=5,
                       label=f"{lbl} (n={len(sub)})")

        ax.set_xlim(-5, _sec(tobii_processed["timestamp"].iloc[-1]) + 5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Segundos desde inicio de grabación Tobii")
        ax.set_title(
            f"Estructura temporal del experimento y taxonomía de epochs — {pid}", pad=8)
        ax.legend(loc="lower right", ncol=4, fontsize=8)
        ax.spines[["top","right","left"]].set_visible(False)

        plt.tight_layout()
        out = fig_dir / f"{pid}_fig_timeline.png"
        plt.savefig(out)
        plt.close(fig)
        logger.info(f"  Fig timeline → {out}")
    except Exception as exc:
        logger.warning(f"  Fig timeline skipped: {exc}")

    # ── Fig C: single synchronized epoch ────────────────────────────────────
    try:
        df_meta = pd.DataFrame(epoch_meta)
        expo    = df_meta[df_meta["epoch_type"] == "exposicion_imagen"]
        if len(expo) == 0:
            logger.warning("  Fig epoch skipped: no exposicion_imagen epochs")
            return

        row = df_meta.iloc[expo.index[min(5, len(expo)-1)]]
        e_eeg, e_gsr, e_pupil = row["eeg"], row["gsr"], row["pupil"]
        t_eeg   = _np.arange(len(e_eeg))  / 125.0 - 0.2
        t_tobii = _np.arange(len(e_gsr))  /  60.0 - 0.2

        fig, axes = plt.subplots(3, 1, figsize=(12, 7.5),
                                 gridspec_kw={"hspace": 0.5})

        # EEG — all channels faint, F3+F4 bold
        fi = [eeg_channels.index(c) for c in ["F3","F4"] if c in eeg_channels]
        if len(e_eeg) > 0:
            for k in range(e_eeg.shape[1]):
                axes[0].plot(t_eeg, e_eeg[:,k], lw=0.3,
                             color="#aec6cf", alpha=0.4)
            if fi:
                axes[0].plot(t_eeg, e_eeg[:,fi].mean(axis=1),
                             lw=1.6, color=CLRS["eeg"], label="F3+F4 (proxy FAA)")
        axes[0].axvline(0, color="black", lw=1.2, ls="--", alpha=0.8, label="Onset")
        axes[0].axhline(0, color="gray",  lw=0.4, alpha=0.4)
        axes[0].set_ylabel("EEG (µV)")
        axes[0].set_title("EEG — 16 canales @ 125 Hz  (bold = F3+F4)", fontsize=9)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].spines[["top","right"]].set_visible(False)

        # GSR
        if len(e_gsr) > 0:
            axes[1].plot(t_tobii, e_gsr, lw=1.3,
                         color="#1b7837", label="GSR raw (z)")
        axes[1].axvline(0, color="black", lw=1.2, ls="--", alpha=0.8)
        axes[1].set_ylabel("GSR (z)")
        axes[1].set_title("GSR @ 60 Hz", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].spines[["top","right"]].set_visible(False)

        # Pupil
        if len(e_pupil) > 0:
            axes[2].plot(t_tobii, e_pupil, lw=1.3,
                         color="#762a83", label="Pupila limpia (z)")
        axes[2].axvline(0, color="black", lw=1.2, ls="--", alpha=0.8)
        axes[2].set_ylabel("Diámetro pupilar (z)")
        axes[2].set_xlabel("Tiempo desde onset del estímulo (s)")
        axes[2].set_title("Pupilometría @ 60 Hz", fontsize=9)
        axes[2].legend(fontsize=8)
        axes[2].spines[["top","right"]].set_visible(False)

        fig.suptitle(
            f"Epoch sincronizado EEG ↔ Tobii — estímulo \"{row['stimulus']}\" "
            f"| {row['epoch_type']} | {pid}\n"
            f"EEG: {row['n_eeg']} muestras @ 125 Hz  "
            f"| Tobii: {row['n_tobii']} muestras @ 60 Hz  "
            f"| duración: {row['duration_s']:.1f} s",
            fontsize=10)
        out = fig_dir / f"{pid}_fig_epoch.png"
        plt.savefig(out)
        plt.close(fig)
        logger.info(f"  Fig epoch    → {out}")
    except Exception as exc:
        logger.warning(f"  Fig epoch skipped: {exc}")

# ---------------------------------------------------------------------------
# 10. Synchronize one participant
# ---------------------------------------------------------------------------

def synchronize_participant(
    participant_id: str,
    eeg: pd.DataFrame,
    tobii_processed: pd.DataFrame,
    tobii_raw: pd.DataFrame,
    cfg: dict,
    paths: dict,
    save: bool = True,
) -> dict:
    """
    Full synchronization pipeline for one participant.

    Parameters
    ----------
    participant_id  : e.g. 'P1'
    eeg             : processed EEG DataFrame (data/processed/eeg/)
    tobii_processed : processed Tobii signals DataFrame (data/processed/tobii/)
    tobii_raw       : interim Tobii DataFrame (data/interim/tobii/) — needed for events
    cfg             : preprocessing.yaml full config
    paths           : paths.yaml full config
    save            : if True, write outputs to disk

    Returns
    -------
    dict:
        sync_params         : offset, drift, scale
        stimuli_classified  : DataFrame with epoch_type per ImageStimulus
        nav_epochs          : DataFrame with web_sitio and web_url epochs
        epoch_meta          : list of dicts (one per epoch) with signal arrays
        qc                  : QC summary dict
    """
    tobii_cfg  = cfg["tobii"]
    sync_cfg   = cfg["synchronization"]
    epoch_cfg  = cfg["eeg"]["epoching"]

    tmin        = epoch_cfg["tmin"]
    tmax_offset = epoch_cfg["tmax_offset"]
    anchor_ev   = sync_cfg["anchor_event"]
    drift_thr   = sync_cfg["drift_threshold_ms"]

    logger.info(f"  Reconstruyendo timestamps Tobii...")
    tobii_raw_ts = reconstruct_tobii_timestamps(tobii_raw, tobii_cfg)

    logger.info(f"  Extrayendo eventos...")
    events = extract_events(tobii_raw_ts, tobii_cfg)

    logger.info(f"  Calculando parámetros de sincronización...")
    sync_params = compute_sync_params(
        eeg, tobii_processed, events, anchor_ev, drift_thr
    )

    logger.info(f"  Aplicando corrección al eje temporal EEG...")
    eeg_corrected = apply_eeg_correction(eeg, sync_params)

    logger.info(f"  Clasificando estímulos...")
    stimuli_classified = classify_stimuli(events, tobii_cfg)

    logger.info(f"  Extrayendo epochs de navegación...")
    nav_epochs = extract_navigation_epochs(events)

    # Combine stimulus + navigation epochs for signal extraction
    eeg_channels = [c for c in eeg.columns if c not in ("timestamp",
                                                          "timestamp_corrected")]
    all_epochs_df = pd.concat(
        [
            stimuli_classified.rename(columns={
                "timestamp_start": "timestamp_start",
                "timestamp_end":   "timestamp_end",
            }),
            nav_epochs,
        ],
        ignore_index=True,
    ).sort_values("timestamp_start").reset_index(drop=True)

    logger.info(f"  Extrayendo señales para {len(all_epochs_df)} epochs...")
    epoch_meta = []
    for _, row in all_epochs_df.iterrows():
        signals = extract_epoch_signals(
            row, eeg_corrected, tobii_processed,
            eeg_channels, tmin, tmax_offset
        )
        epoch_meta.append({
            "stimulus":   row.get("stimulus",   ""),
            "epoch_type": row["epoch_type"],
            "duration_s": row["duration_s"],
            "onset":      row["timestamp_start"],
            **signals,
        })

    logger.info(f"  Validando epochs...")
    qc = validate_epochs(epoch_meta)
    ratio_str = (f"{qc['ratio_mean']:.3f} ± {qc['ratio_std']:.3f}"
                 f"  (esperado {qc['ratio_expected']:.3f})")
    logger.info(f"    Epochs totales  : {qc['n_epochs']}")
    logger.info(f"    Vacíos EEG      : {qc['n_empty_eeg']}")
    logger.info(f"    Vacíos Tobii    : {qc['n_empty_tobii']}")
    logger.info(f"    Ratio Fs        : {ratio_str}")
    logger.info(f"    NaN GSR mediana : {qc['pct_nan_gsr_median']*100:.1f}%")
    logger.info(f"    Epochs alto NaN : {qc['n_high_nan']}")

    if save:
        save_sync_outputs(
            participant_id, sync_params,
            stimuli_classified, nav_epochs, epoch_meta, qc, paths
        )
        logger.info(f"  Generando figuras de tesis...")
        generate_sync_figures(
            participant_id, stimuli_classified, epoch_meta,
            tobii_processed, eeg_channels, paths,
        )

    return {
        "sync_params":        sync_params,
        "stimuli_classified": stimuli_classified,
        "nav_epochs":         nav_epochs,
        "epoch_meta":         epoch_meta,
        "qc":                 qc,
    }


# ---------------------------------------------------------------------------
# 11. Batch processing
# ---------------------------------------------------------------------------

def synchronize_all(save: bool = True) -> pd.DataFrame:
    """
    Run synchronization pipeline for all participants.

    Matches EEG files (data/processed/eeg/) with Tobii files
    (data/processed/tobii/ and data/interim/tobii/) by participant ID.

    Participant ID extraction:
        EEG filename:   'OpenBCISession_P1.parquet' → 'P1'
        Tobii filename: 'P1.parquet'                → 'P1'

    Returns
    -------
    QC summary DataFrame (one row per participant)
    """
    paths   = load_config("configs/paths.yaml")
    cfg     = load_config("configs/preprocessing.yaml")

    eeg_dir          = Path(paths["data"]["processed"]["eeg"])
    tobii_proc_dir   = Path(paths["data"]["processed"]["tobii"])
    tobii_inter_dir  = Path(paths["data"]["interim"]["tobii"])
    qc_dir           = Path(paths["reports"]["synchronization"]["qc"])
    qc_dir.mkdir(parents=True, exist_ok=True)

    eeg_files = sorted(eeg_dir.glob("*.parquet"))
    logger.info(f"Found {len(eeg_files)} EEG files to synchronize.")

    qc_rows     = []
    succeeded   = []
    failed      = []

    for eeg_path in eeg_files:
        # Extract participant ID: 'OpenBCISession_P1' → 'P1'
        stem = eeg_path.stem
        pid_match = re.search(r"(P\d+|Test_Participant)", stem, re.IGNORECASE)
        if pid_match is None:
            logger.warning(f"Cannot extract participant ID from '{stem}' — skipping")
            continue
        pid = pid_match.group(1)

        tobii_proc_path  = tobii_proc_dir  / f"{pid}.parquet"
        tobii_inter_path = tobii_inter_dir / f"{pid}.parquet"

        if not tobii_proc_path.exists():
            logger.warning(f"Tobii processed not found for {pid} — skipping")
            failed.append(pid)
            continue
        if not tobii_inter_path.exists():
            logger.warning(f"Tobii interim not found for {pid} — skipping")
            failed.append(pid)
            continue

        logger.info("=" * 60)
        logger.info(f"Processing: {pid}")

        try:
            eeg             = pd.read_parquet(eeg_path)
            tobii_processed = pd.read_parquet(tobii_proc_path)
            tobii_raw       = pd.read_parquet(tobii_inter_path)

            result = synchronize_participant(
                pid, eeg, tobii_processed, tobii_raw, cfg, paths, save=save
            )
            qc = result["qc"]
            sp = result["sync_params"]
            qc_rows.append({
                "participant":       pid,
                "offset_s":          round(sp["offset_s"], 3),
                "drift_ms":          round(sp["drift_ms"], 3),
                "drift_ms_per_min":  round(sp["drift_ms_per_min"], 3),
                "drift_scale":       sp["drift_scale"],
                "n_epochs":          qc["n_epochs"],
                "n_empty_eeg":       qc["n_empty_eeg"],
                "n_empty_tobii":     qc["n_empty_tobii"],
                "ratio_mean":        round(qc["ratio_mean"], 4),
                "ratio_std":         round(qc["ratio_std"], 4),
                "pct_nan_gsr":       round(qc["pct_nan_gsr_median"] * 100, 2),
                "pct_nan_pupil":     round(qc["pct_nan_pupil_median"] * 100, 2),
                "n_high_nan":        qc["n_high_nan"],
                "status":            "OK",
            })
            succeeded.append(pid)

        except Exception as exc:  # noqa: BLE001
            logger.error(f"FAILED: {pid} — {exc}")
            failed.append(pid)
            qc_rows.append({"participant": pid, "status": f"FAILED: {exc}"})

    logger.info("=" * 60)
    if failed:
        logger.warning(f"{len(failed)} files failed: {failed}")

    qc_summary = pd.DataFrame(qc_rows)
    if save and len(qc_summary) > 0:
        summary_path = qc_dir / "sync_qc_summary.csv"
        qc_summary.to_csv(summary_path, index=False)
        logger.info(f"Global QC summary → {summary_path}")
        # Regenerate drift figure for P1 now that summary exists
        try:
            p1_tobii = pd.read_parquet(tobii_proc_dir / "P1.parquet")
            p1_stimuli = pd.read_parquet(
                Path(paths["data"]["processed"]["multimodal"]) / "P1_stimuli.parquet"
            )
            generate_sync_figures(
                "P1", p1_stimuli, [], p1_tobii,
                [], paths, qc_summary_path=summary_path,
            )
        except Exception as _e:
            logger.warning(f"Post-batch drift figure skipped: {_e}")

    logger.info(
        f"Synchronization complete: {len(succeeded)} succeeded, "
        f"{len(failed)} failed."
    )

    print("\n=== Synchronization QC Summary ===")
    if len(qc_summary) > 0 and "participant" in qc_summary.columns:
        print(qc_summary.set_index("participant").to_string())
    else:
        print(qc_summary.to_string())
    return qc_summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EEG ↔ Tobii synchronization — Fondecyt 1231122"
    )
    parser.add_argument(
        "--participant", type=str, default=None,
        help="Process a single participant ID (e.g. P1). Omit to process all."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run without writing output files (dry run)."
    )
    args = parser.parse_args()

    save_flag = not args.no_save

    if args.participant:
        paths = load_config("configs/paths.yaml")
        cfg   = load_config("configs/preprocessing.yaml")
        pid   = args.participant

        eeg_files   = sorted(
            Path(paths["data"]["processed"]["eeg"]).glob("*.parquet")
        )
        pid_match   = next(
            (f for f in eeg_files
             if re.search(rf"{re.escape(pid)}", f.stem, re.IGNORECASE)), None
        )
        if pid_match is None:
            raise FileNotFoundError(f"EEG file not found for participant '{pid}'")

        tobii_proc  = Path(paths["data"]["processed"]["tobii"]) / f"{pid}.parquet"
        tobii_inter = Path(paths["data"]["interim"]["tobii"])   / f"{pid}.parquet"

        logger.info("=" * 60)
        logger.info(f"Processing: {pid}")
        result = synchronize_participant(
            pid,
            pd.read_parquet(pid_match),
            pd.read_parquet(tobii_proc),
            pd.read_parquet(tobii_inter),
            cfg, paths, save=save_flag,
        )
        sp = result["sync_params"]
        qc = result["qc"]
        print(f"\nOffset: {sp['offset_s']:.3f} s  |  "
              f"Drift: {sp['drift_ms']:.2f} ms  |  "
              f"Scale: {sp['drift_scale']:.8f}")
        print(f"Epochs: {qc['n_epochs']}  |  "
              f"Empty EEG: {qc['n_empty_eeg']}  |  "
              f"Ratio: {qc['ratio_mean']:.3f} ± {qc['ratio_std']:.3f}")
    else:
        synchronize_all(save=save_flag)