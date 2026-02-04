import pandas as pd
from src.preprocess import preprocess_data
import pytest


def test_preprocess_creates_splits(tmp_path):

    df = pd.DataFrame({
    "years_experience": list(range(1, 13)),
    "skills_match_score": [10,20,30,40,50,60,70,80,90,40,50,60],
    "education_level": [
        "Bachelors","Masters","PhD","High School",
        "Bachelors","Masters","PhD","High School",
        "Bachelors","Masters","PhD","High School"
    ],
    "project_count": list(range(1, 13)),
    "resume_length": [100+i*10 for i in range(12)],
    "github_activity": list(range(5,17)),
    "shortlisted": ["Yes","No"] * 6
    })



    data_file = tmp_path / "input.csv"
    df.to_csv(data_file, index=False)

    output_dir = tmp_path / "processed"

    preprocess_data(data_file, output_dir)

    assert (output_dir / "X_train.csv").exists()
    assert (output_dir / "X_val.csv").exists()
    assert (output_dir / "X_test.csv").exists()

def test_target_mapping(tmp_path):

    df = pd.DataFrame({
    "years_experience": list(range(1, 13)),
    "skills_match_score": [10,20,30,40,50,60,70,80,90,40,50,60],
    "education_level": [
        "Bachelors","Masters","PhD","High School",
        "Bachelors","Masters","PhD","High School",
        "Bachelors","Masters","PhD","High School"
    ],
    "project_count": list(range(1, 13)),
    "resume_length": [100+i*10 for i in range(12)],
    "github_activity": list(range(5,17)),
    "shortlisted": ["Yes","No"] * 6
    })


    data_file = tmp_path / "input.csv"
    df.to_csv(data_file, index=False)

    output_dir = tmp_path / "processed"

    preprocess_data(data_file, output_dir)

    y_train = pd.read_csv(output_dir / "y_train.csv")

    assert set(y_train.iloc[:, 0].unique()).issubset({0, 1})



def test_invalid_education_fails(tmp_path):

    df = pd.DataFrame({
        "years_experience": [1],
        "skills_match_score": [0.5],
        "education_level": ["InvalidLevel"],
        "project_count": [1],
        "resume_length": [100],
        "github_activity": [5],
        "shortlisted": ["Yes"],
    })

    data_file = tmp_path / "input.csv"
    df.to_csv(data_file, index=False)

    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError):
        preprocess_data(data_file, output_dir)

@pytest.mark.parametrize(
    "column",
    ["years_experience", "project_count", "github_activity"]
)
def test_negative_numeric_fields_fail(tmp_path, column):

    df = pd.DataFrame({
        "years_experience": [1],
        "skills_match_score": [50],
        "education_level": ["Bachelors"],
        "project_count": [1],
        "resume_length": [100],
        "github_activity": [5],
        "shortlisted": ["Yes"],
    })

    df[column] = -1  # Inject negative value into tested column

    data_file = tmp_path / "input.csv"
    df.to_csv(data_file, index=False)

    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError):
        preprocess_data(data_file, output_dir)