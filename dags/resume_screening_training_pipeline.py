import sys
sys.path.append("/opt/airflow")

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import get_current_context
from datetime import datetime
from pathlib import Path

from src.ingest import ingest_data
from src.validate_data import validate_dataset, write_validation_report
from src.preprocess import preprocess_data
from src.train import train
from src.config import load_config


with DAG(
    dag_id="resume_screening_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "training"],
    params={
        "model_name": "random_forest",
    },
) as dag:

    # Ingestion
    @task
    def ingest():
        return str(
            ingest_data(
                source="kaggle",
                dataset_id="sonalshinde123/ai-driven-resume-screening-dataset",
                raw_data_dir=Path("/opt/airflow/data/raw"),
            )
        )

    # Validation
    @task
    def validate(ingested_file):
        report = validate_dataset(Path(ingested_file))

        write_validation_report(
            report,
            output_dir=Path("/opt/airflow/data/validation"),
        )

        if report["status"] != "PASS":
            raise ValueError("Data validation failed")

        return ingested_file

    # Preprocessing
    @task
    def preprocess(ingested_file):
        return preprocess_data(
            data_path=Path(ingested_file),
            output_dir=Path("/opt/airflow/data/processed"),
        )


    # Training
    @task
    def train_model(_):
        context = get_current_context()
        conf = context["dag_run"].conf
        

        CONFIG = load_config("/opt/airflow/configs/train.yaml")

        if conf and "model_name" in conf:
            CONFIG["model"]["selected"] = conf["model_name"]

        return train(CONFIG)


    # DAG wiring
    ingested = ingest()
    validated = validate(ingested)
    processed = preprocess(validated)
    trained = train_model(processed)
