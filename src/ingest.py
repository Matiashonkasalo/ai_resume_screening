from pathlib import Path
from datetime import datetime
import json
import shutil
import logging
import pandas as pd
import kagglehub

Logger = logging.getLogger(__name__)

def ingest_from_kaggle(raw_data_dir, dataset_id) -> Path:

    """
    Function for ingesting data from kaggle into local raw data directory

    Args: 
        raw_data_dir: Path object to the local raw data directory
        dataset_id: Kaggle dataset identifier

    Returns:
        Path to the ingested CSV file
    """

    Logger.info("Ingestion from Kaggle started")
    Logger.info(f"Dataset ID: {dataset_id}")
    Logger.info(f"Ingested files are loaded to: {raw_data_dir}")

    downloaded_path = Path(kagglehub.dataset_download(dataset_id))

    csv_files = []
    for file in downloaded_path.iterdir():
        if not file.suffix=='.csv':
            raise RuntimeError("No CSV files found in Kaggle dataset")

        csv_files.append(file)

    # Explicit assumption: single CSV dataset
    source_file = csv_files[0]
    destination_file = raw_data_dir / source_file.name

    if not destination_file.exists():
        shutil.copy(source_file, destination_file) #copies the file to destination
        Logger.info(f"Copied file to {destination_file}")
    else:
        Logger.info(f"File already exists, skipping copy: {destination_file}")

    # Sanity check for the data 
        
    df = pd.read_csv(destination_file)

    if df.empty:
        raise RuntimeError("Ingested dataset is empty")

    Logger.info(f"Ingested dataset has {df.shape[0]} rows and {df.shape[1]} columns")

    # Writing ingestions metadata into a folder and storing it
    ingestion_meta = {
        "source": "kaggle",
        "dataset_id": dataset_id,
        "file_name": destination_file.name,
        "rows": len(df),
        "columns": list(df.columns),
        "ingested_at": datetime.utcnow().isoformat(),
    }

    meta_dir = raw_data_dir.parent / "ingestion"
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / "ingestion_meta.json"
    meta_path.write_text(json.dumps(ingestion_meta, indent=2))

    Logger.info(f"Ingestion metadata written to {meta_path}")
    Logger.info("Kaggle ingestion completed successfully")

    return destination_file



INGESTION_HANDLER = {
    "kaggle": ingest_from_kaggle,
}


def ingest_data(raw_data_dir, dataset_id, source = "kaggle") -> Path:
    
    """
    Data ingestion function

    Args: 
        source: str of the data source (in this project only used "kaggle")
        raw_data_dir: Path object to the folder where the data is stored locally.

    Returns: Path object pointing to the ingested data file.
    """

    raw_data_dir.mkdir(parents=True,exist_ok=True) # ensure the raw data folder exist

    try: 
        Handler = INGESTION_HANDLER[source] # gives the ingestion function 
    except KeyError:
        raise ValueError(f"Unsupported data source: {source}")
    

    return Handler(raw_data_dir,dataset_id)
   
