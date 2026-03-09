from pathlib import Path

from src.ingestion.load_eeg import convert_all_eeg
from src.ingestion.load_tobii import split_tobii_by_participant
from src.utils.config_loader import load_config


def run():

    config = load_config("configs/paths.yaml")

    eeg_raw = Path(config["data"]["raw"]["eeg"])
    tobii_raw = Path(config["data"]["raw"]["tobii"])

    eeg_interim = Path(config["data"]["interim"]["eeg"])
    tobii_interim = Path(config["data"]["interim"]["tobii"])

    print("Starting ingestion pipeline")

    print("Converting EEG data")
    convert_all_eeg(eeg_raw, eeg_interim)

    print("Splitting Tobii data")
    split_tobii_by_participant(tobii_raw, tobii_interim)

    print("Ingestion completed")


if __name__ == "__main__":
    run()