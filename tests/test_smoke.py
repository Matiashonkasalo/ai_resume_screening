from pathlib import Path
import pandas as pd

from src.preprocess import preprocess_data
from src.train import train


def make_smoke_resume_df(n=12):

    return pd.DataFrame(
        {
            "years_experience": list(range(1, n + 1)),
            "skills_match_score": [10 + i * 5 for i in range(n)],
            "education_level": ["Bachelors", "Masters", "PhD", "High School"]
            * (n // 4),
            "project_count": list(range(1, n + 1)),
            "resume_length": [100 + i * 10 for i in range(n)],
            "github_activity": list(range(5, 5 + n)),
            "shortlisted": ["Yes", "No"] * (n // 2),
        }
    )


def test_training_e2e_smoke(tmp_path):

    # Create raw dataset
    df = make_smoke_resume_df()

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    raw_dir.mkdir()
    processed_dir.mkdir()
    artifacts_dir.mkdir()

    raw_file = raw_dir / "resume.csv"
    df.to_csv(raw_file, index=False)

    # Run preprocessing
    preprocess_data(raw_file, processed_dir)

    # Minimal config
    config = {
        "experiment": {"name": "smoke_test"},
        "model": {"selected": "random_forest"},
        "models": {"random_forest": {}},
        "tuning": {"enabled": False, "n_trials": 1},
        "data": {
            "processed_dir": str(processed_dir),
            "artifacts_dir": str(artifacts_dir),
        },
        "training": {"random_state": 42},
    }

    # Run training
    run_dir = train(config)
    run_dir = Path(run_dir)

    #  Assertions
    assert run_dir.exists()
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()

    latest_file = artifacts_dir / "random_forest" / "latest.json"
    assert latest_file.exists()
