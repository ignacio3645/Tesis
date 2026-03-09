from pathlib import Path
import pandas as pd


def load_tobii_file(filepath: Path) -> pd.DataFrame:
    """
    Load Tobii export TSV.
    """
    df = pd.read_csv(filepath, sep="\t")

    return df


def split_tobii_by_participant(raw_file: Path, interim_dir: Path):
    """
    Split Tobii dataset by participant and save parquet files.
    """

    df = load_tobii_file(raw_file)

    interim_dir.mkdir(parents=True, exist_ok=True)

    participants = df["Participant name"].unique()

    print(f"Found {len(participants)} participants")

    for p in participants:

        print(f"Processing participant: {p}")

        df_p = df[df["Participant name"] == p]

        safe_name = str(p).replace(" ", "_")

        output_file = interim_dir / f"{safe_name}.parquet"

        df_p.to_parquet(output_file, index=False)

        print(f"Saved: {output_file}")