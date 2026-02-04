import kagglehub
from pathlib import Path
import shutil

print(f"HERE {Path(__file__)} is the path")

Project_root = Path(__file__).resolve().parents[1]
RAW_data_dir = Project_root / "data" / "raw"


def load_data():
    # returns raw data path object
    downloaded_path = Path(
        kagglehub.dataset_download("sonalshinde123/ai-driven-resume-screening-dataset")
    )

    for file in downloaded_path.iterdir():
        if file.is_file():
            path = RAW_data_dir / file.name
            shutil.copy(file, path)

    return RAW_data_dir
