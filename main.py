import argparse

from src.ingestion.run_ingestion import run as run_ingestion


def main():

    parser = argparse.ArgumentParser(
        description="Multimodal Neurophysiology Pipeline"
    )

    parser.add_argument(
        "stage",
        type=str,
        choices=[
            "ingestion",
            "preprocessing",
            "synchronization",
            "features",
            "train"
        ],
        help="Pipeline stage to run"
    )

    args = parser.parse_args()

    if args.stage == "ingestion":
        run_ingestion()

    elif args.stage == "preprocessing":
        print("Preprocessing not implemented yet")

    elif args.stage == "synchronization":
        print("Synchronization not implemented yet")

    elif args.stage == "features":
        print("Feature engineering not implemented yet")

    elif args.stage == "train":
        print("Training pipeline not implemented yet")


if __name__ == "__main__":
    main()