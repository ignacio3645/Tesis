"""
main.py
=======
Pipeline orchestrator for the Fondecyt 1231122 neurophysiological dataset.

Usage
-----
    # Run a specific stage
    python main.py preprocessing_eeg
    python main.py preprocessing_tobii
    python main.py synchronization

    # Run all preprocessing stages in order
    python main.py preprocessing

    # Run a single participant through the full preprocessing pipeline
    python main.py preprocessing --participant P1

    # Dry run (no files written to disk)
    python main.py preprocessing --no-save

Pipeline stages
---------------
    preprocessing_eeg   : CAR → Notch → Bandpass → ICA (clean_eeg.py)
    preprocessing_tobii : Timestamps → Resample → GSR → Pupil → ET (clean_tobii.py)
    synchronization     : Offset + drift correction → epoch extraction (synchronize.py)
    preprocessing       : Runs all three stages above in order

Author: Ignacio Negrete Silva
Project: Fondecyt 1231122
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_preprocessing_eeg(participant: str | None, save: bool) -> None:
    """
    Stage 1: EEG preprocessing.
    CAR → Notch 50 Hz → Bandpass 1-40 Hz → Extended Infomax ICA + ICLabel.

    Input  : data/interim/eeg/   (one .parquet per participant)
    Output : data/processed/eeg/ (one .parquet per participant, 16 ch + timestamp)
    Reports: reports/preprocessing/qc/ica_summary.csv
    """
    from src.preprocessing.clean_eeg import (
        preprocess_participant,
        preprocess_all,
        build_ica_summary,
    )

    paths  = load_config("configs/paths.yaml")
    cfg    = load_config("configs/preprocessing.yaml")
    eeg_cfg = cfg["eeg"]

    interim_dir   = Path(paths["data"]["interim"]["eeg"])
    processed_dir = Path(paths["data"]["processed"]["eeg"])
    qc_dir        = Path(paths["reports"]["preprocessing"]["qc"])
    qc_dir.mkdir(parents=True, exist_ok=True)

    if participant:
        filepath = next(
            (f for f in interim_dir.glob("*.parquet") if participant in f.stem), None
        )
        if filepath is None:
            raise FileNotFoundError(
                f"No interim EEG file found for participant '{participant}' "
                f"in {interim_dir}"
            )
        logger.info(f"EEG preprocessing — single participant: {participant}")
        df_clean, ica_report = preprocess_participant(
            filepath, cfg=eeg_cfg, output_dir=processed_dir, save=save
        )
        logger.info(f"  Components excluded: {ica_report['excluded']}")
    else:
        logger.info("EEG preprocessing — all participants")
        reports = preprocess_all(interim_dir, processed_dir, cfg=eeg_cfg)
        summary = build_ica_summary(reports)
        if save and len(summary) > 0:
            summary_path = qc_dir / "ica_summary.csv"
            summary.to_csv(summary_path)
            logger.info(f"ICA summary → {summary_path}")
        print("\n=== ICA Quality Control Summary ===")
        print(summary.to_string())


def run_preprocessing_tobii(participant: str | None, save: bool) -> None:
    """
    Stage 2: Tobii preprocessing.
    Timestamps → Resample 60 Hz → GSR cleaning (SCL/SCR) → Pupil blink interpolation
    → Eye tracking fixation/saccade extraction.

    Input  : data/interim/tobii/   (one .parquet per participant)
    Output : data/processed/tobii/ (signals .parquet + fixations .parquet + saccades .parquet)
    Reports: reports/preprocessing/qc/tobii_qc_summary.csv
    """
    from src.preprocessing.clean_tobii import preprocess_participant, preprocess_all

    paths    = load_config("configs/paths.yaml")
    cfg      = load_config("configs/preprocessing.yaml")

    interim_dir   = Path(paths["data"]["interim"]["tobii"])
    processed_dir = Path(paths["data"]["processed"]["tobii"])
    report_dirs   = paths["reports"]["preprocessing"]

    if participant:
        filepath = interim_dir / f"{participant}.parquet"
        if not filepath.exists():
            raise FileNotFoundError(
                f"No interim Tobii file found for participant '{participant}': {filepath}"
            )
        logger.info(f"Tobii preprocessing — single participant: {participant}")
        preprocess_participant(
            filepath=filepath,
            cfg=cfg,
            output_dir=processed_dir,
            report_dirs=report_dirs,
            save=save,
        )
    else:
        logger.info("Tobii preprocessing — all participants")
        preprocess_all(
            tobii_interim_dir=interim_dir,
            tobii_processed_dir=processed_dir,
            cfg=cfg,
            report_dirs=report_dirs,
        )


def run_synchronization(participant: str | None, save: bool) -> None:
    """
    Stage 3: EEG ↔ Tobii synchronization + epoch extraction.
    Offset correction → drift scale → stimulus classification → epoch slicing.

    Input  : data/processed/eeg/   + data/processed/tobii/ + data/interim/tobii/
    Output : data/processed/multimodal/ (stimuli + nav_epochs + sync_params per participant)
    Reports: reports/synchronization/qc/sync_qc_summary.csv
    """
    from src.preprocessing.synchronize import synchronize_participant, synchronize_all

    if participant:
        import pandas as pd
        from src.preprocessing.synchronize import synchronize_participant

        paths = load_config("configs/paths.yaml")
        cfg   = load_config("configs/preprocessing.yaml")
        pid   = participant

        eeg_path    = next(
            Path(paths["data"]["processed"]["eeg"]).glob(f"*{pid}*.parquet"), None
        )
        tobii_proc  = Path(paths["data"]["processed"]["tobii"]) / f"{pid}.parquet"
        tobii_inter = Path(paths["data"]["interim"]["tobii"])   / f"{pid}.parquet"

        if eeg_path is None:
            raise FileNotFoundError(f"EEG processed file not found for '{pid}'")

        logger.info(f"Synchronization — single participant: {pid}")
        result = synchronize_participant(
            pid,
            pd.read_parquet(eeg_path),
            pd.read_parquet(tobii_proc),
            pd.read_parquet(tobii_inter),
            cfg, paths, save=save,
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
        logger.info("Synchronization — all participants")
        synchronize_all(save=save)


def run_full_preprocessing(participant: str | None, save: bool) -> None:
    """
    Run all three preprocessing stages in order:
        1. EEG preprocessing
        2. Tobii preprocessing
        3. Synchronization

    Each stage reads from the output of the previous stage.
    If a single participant is specified, runs all three for that participant only.
    """
    stages = [
        ("EEG preprocessing",   run_preprocessing_eeg),
        ("Tobii preprocessing",  run_preprocessing_tobii),
        ("Synchronization",      run_synchronization),
    ]

    total_start = time.time()

    for stage_name, stage_fn in stages:
        logger.info("=" * 60)
        logger.info(f"STAGE: {stage_name}")
        logger.info("=" * 60)
        t0 = time.time()
        try:
            stage_fn(participant=participant, save=save)
            elapsed = time.time() - t0
            logger.info(f"STAGE COMPLETE: {stage_name} — {elapsed:.1f} s")
        except Exception as exc:
            logger.error(f"STAGE FAILED: {stage_name} — {exc}")
            raise

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"FULL PREPROCESSING COMPLETE — total: {total_elapsed:.1f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGES = {
    "preprocessing_eeg":   run_preprocessing_eeg,
    "preprocessing_tobii": run_preprocessing_tobii,
    "synchronization":     run_synchronization,
    "preprocessing":       run_full_preprocessing,
}

STAGE_DESCRIPTIONS = {
    "preprocessing_eeg":   "EEG: CAR → Notch → Bandpass → ICA",
    "preprocessing_tobii": "Tobii: timestamps → resample → GSR → pupil → ET",
    "synchronization":     "Sync: offset + drift → epoch extraction",
    "preprocessing":       "Full pipeline: EEG + Tobii + Sync in order",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fondecyt 1231122 — Neurophysiological data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:<22} {v}" for k, v in STAGE_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "stage",
        choices=list(STAGES.keys()),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        metavar="ID",
        help="Process a single participant (e.g. P1). Omit to process all.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry run — process data without writing output files.",
    )

    args = parser.parse_args()
    save_flag = not args.no_save

    logger.info(f"Stage     : {args.stage}")
    logger.info(f"Participant: {args.participant or 'ALL'}")
    logger.info(f"Save      : {save_flag}")

    try:
        STAGES[args.stage](participant=args.participant, save=save_flag)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise