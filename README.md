# AI Resume Screening Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.0+-017CEE?logo=apache-airflow)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red)](https://github.com/astral-sh/ruff)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

End-to-end MLOps pipeline for automated resume screening using multiple machine learning models, Airflow orchestration, MLflow experiment tracking, and CI/CD validation.

---

## Project Overview

This project builds a fully reproducible machine learning training pipeline that predicts whether a candidate should be shortlisted based on resume features.

The system demonstrates production-grade MLOps practices including:
- Config-driven model training
- Multi-model experimentation
- Automated orchestration using Airflow
- MLflow experiment tracking
- Dockerized environment
- Continuous integration testing
- End-to-end smoke testing

---

## Problem Statement

Manual resume screening is time-consuming and subjective.  

This project simulates an automated pipeline capable of:
- Ingesting resume datasets
- Validating data quality
- Preprocessing features
- Training and tuning multiple models
- Logging experiments and artifacts
- Versioning trained models

---

## Dataset 

Link to dataset:
- https://www.kaggle.com/datasets/sonalshinde123/ai-driven-resume-screening-dataset

Data ingestion is automated through the Kaggle API within the Airflow pipeline.

### Features

| Feature | Type | Description |
|----------|----------|-------------|
| `years_experience` | Numeric | Total years of professional experience |
| `skills_match_score` | Numeric | Score representing how well candidate skills match job requirements |
| `education_level` | Categorical (encoded) | Highest education achieved (High School, Bachelor’s, Master’s, PhD) |
| `project_count` | Numeric | Number of completed professional or academic projects |
| `resume_length` | Numeric | Resume length measured in words |
| `github_activity` | Numeric | Proxy metric representing open-source activity or coding engagement |

---

### Target Label

| Label | Type | Description |
|----------|----------|-------------|
| `shortlisted` | Binary | Indicates whether candidate was shortlisted (Yes / No) |


## Architecture

```
┌─────────────────┐
│ Data Ingestion  │  ← Kaggle API
└────────┬────────┘
         ↓
┌─────────────────┐
│ Data Validation │  ← Schema checks, quality gates
└────────┬────────┘
         ↓
┌─────────────────┐
│ Preprocessing   │  ← Feature engineering, scaling
└────────┬────────┘
         ↓
┌─────────────────┐
│ Model Training  │  ← Multi-model, hyperparameter tuning
└────────┬────────┘
         ↓
┌─────────────────┐
│   Evaluation    │  ← Metrics, visualizations
└────────┬────────┘
         ↓
┌─────────────────┐
│ MLflow Logging  │  ← Experiment tracking, artifacts
└────────┬────────┘
         ↓
┌─────────────────┐
│Model Versioning │  ← Artifact storage, registry
└─────────────────┘
```

---

## Pipeline Orchestration

The training workflow is orchestrated using **Apache Airflow**.

The DAG performs:
1. Dataset ingestion from Kaggle  
2. Data validation checks  
3. Feature preprocessing and stratified splitting  
4. Model training with configurable model selection  
5. Evaluation and MLflow experiment logging  
6. Artifact storage and version tracking  

---

## Supported Models

- Random Forest
- AdaBoost
- XGBoost

Model selection is runtime configurable through Airflow parameters.

---

## Experiment Tracking

Experiments are tracked using **MLflow**, logging:
- Model parameters
- Evaluation metrics
- Confusion matrices
- ROC curves
- Classification reports
- Model artifacts

---

## Project Structure

- `dags/` → Airflow pipelines
- `src/` → Core pipeline logic
- `tests/` → Unit + smoke tests
- `configs/` → Training configuration
- `artifacts/` → Model outputs and metrics
- `data/` → Raw and processed datasets

---

## Quickstart

Clone repository:

git clone https://github.com/Matiashonkasalo/ai_resume_screening.git
cd ai_resume_screening

Start pipeline environment:

docker-compose up --build

Airflow UI will be available at:
```
http://localhost:8080
```

**Trigger DAG:** `resume_screening_training_pipeline`

---

## Testing & CI

The project includes:
- Unit tests
- Configuration validation tests
- Preprocessing tests
- Model factory tests
- End-to-end smoke tests

Continuous integration automatically validates:
- Code style using Ruff and Black
- Test execution using Pytest
- Dependency reproducibility using uv

---

## Artifacts

Each training run produces:
- Model binaries
- Evaluation metrics
- Visualization reports
- Latest model pointers

---


## Tech Stack

- Python
- Scikit-learn
- XGBoost
- Apache Airflow
- MLflow
- Docker
- Pytest
- Ruff
- Black
- GitHub Actions
- uv dependency manager




