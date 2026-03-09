from pathlib import Path
import pandas as pd


def load_eeg_file(filepath: Path) -> pd.DataFrame:
    """
    Load an OpenBCI EEG text file.

    Parameters
    ----------
    filepath : Path
        Path to raw EEG file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, comment="%")

    return df


def convert_all_eeg(raw_dir: Path, interim_dir: Path):
    """
    Convert all EEG raw files to parquet.
    """

    interim_dir.mkdir(parents=True, exist_ok=True)

    for file in raw_dir.rglob("*.txt"):
        print(f"Processing EEG file: {file.name}")

        df = load_eeg_file(file)
        
        session_name = file.parent.name
        output_file = interim_dir / f"{session_name}.parquet"

        df.to_parquet(output_file, index=False)

        print(f"Saved: {output_file}")