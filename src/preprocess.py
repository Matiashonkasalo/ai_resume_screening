import pandas as pd
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split

Logger = logging.getLogger(__name__)


# Mappings

EDUCATION_MAPPING = {
    "High School": 0,
    "Bachelor": 1,
    "Bachelors": 1,
    "Master": 2,
    "Masters": 2,
    "PhD": 3,
}

TARGET_MAPPING = {
    "No": 0,
    "Yes": 1,
}


# Preprocessing


def preprocess_data(data_path, output_dir):
    """
    Preprocess validated dataset and produce train/val/test splits.
    """

    Logger.info(f"Starting preprocessing for {data_path}")

    df = pd.read_csv(data_path)

    # Target mapping

    y = df["shortlisted"].map(TARGET_MAPPING)

    if y.isnull().any():
        raise ValueError("Target contains unmapped values")

    X = df.drop(columns=["shortlisted"])

    # Feature transformations

    X["education_level"] = X["education_level"].map(EDUCATION_MAPPING)

    if X["education_level"].isnull().any():
        raise ValueError("Education level contains unmapped values")

    # Train / Val / Test split

    Logger.info("Splitting data into train / validation / test")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    # Class balance reporting

    def class_dist(y_series):
        return y_series.value_counts(normalize=True).to_dict()

    class_distribution = {
        "train": class_dist(y_train),
        "validation": class_dist(y_val),
        "test": class_dist(y_test),
    }

    Logger.info(f"Class distribution: {class_distribution}")

    # Save artifacts

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)

    X_val.to_csv(output_dir / "X_val.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)

    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    meta = {
        "processed_at": datetime.utcnow().isoformat(),
        "num_samples": len(X),
        "splits": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "features": list(X.columns),
        "target_mapping": TARGET_MAPPING,
        "education_mapping": EDUCATION_MAPPING,
        "class_distribution": class_distribution,
        "random_state": 42,
    }

    meta_path = output_dir / "preprocess_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    Logger.info("Preprocessing completed successfully")

    return {
        "X_train": str(output_dir / "X_train.csv"),
        "y_train": str(output_dir / "y_train.csv"),
        "X_val": str(output_dir / "X_val.csv"),
        "y_val": str(output_dir / "y_val.csv"),
        "X_test": str(output_dir / "X_test.csv"),
        "y_test": str(output_dir / "y_test.csv"),
        "meta": str(meta_path),
    }
