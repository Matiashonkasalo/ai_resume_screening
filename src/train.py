from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

from src.evaluate import plot_confusion_matrix, generate_classification_report
from src.tune import tune_model
from src.models import build_model
from src.evaluate import plot_roc_curve
import time


Logger = logging.getLogger(__name__)


def train(config):
    """
    Train, evaluate, and register a model using Optuna + MLflow.
    """

    
    # Extract values from config
    model_name = config["model"]["selected"]
    model_cfg = config["models"][model_name]

    class_weight = model_cfg.get("class_weight")

    default_params = model_cfg.copy()
    default_params.pop("class_weight", None)

    tuning_enabled = config["tuning"]["enabled"]
    n_trials = config["tuning"]["n_trials"]

    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = Path(config["data"]["artifacts_dir"])

    Logger.info(f"Starting training for model: {model_name}")
    
    # Load data splits
   
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()

    X_val = pd.read_csv(processed_dir / "X_val.csv")
    y_val = pd.read_csv(processed_dir / "y_val.csv").squeeze()

    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()


    # MLflow run
    start_time = time.time()

    experiment_name = config["experiment"]["name"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    run_name = f"{experiment_name}_{model_name}_{timestamp}"


    with mlflow.start_run(run_name=run_name):

        mlflow.log_dict(config, "config.yaml")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("class_weight", class_weight)

        best_params = {}
        best_cv_score = None

        if tuning_enabled:
            Logger.info("Starting hyperparameter tuning")
            mlflow.log_param("n_trials", n_trials)

            best_params, best_cv_score = tune_model(
                X_train,
                y_train,
                model_name=model_name,
                n_trials=n_trials,
            )
            
            model_params = {**default_params, **best_params}
            mlflow.log_metric("cv_f1", best_cv_score)
            mlflow.log_params(best_params)
        else:
            Logger.info("Hyperparameter tuning disabled")
            model_params = default_params


        model = build_model(
            model_name,
            model_params,
            class_weight=class_weight
        )

        model.fit(X_train, y_train)

        # Validation
        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds)
        val_acc = accuracy_score(y_val, val_preds)

        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_accuracy", val_acc)

        # Test
        test_preds = model.predict(X_test)
        test_f1 = f1_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, test_preds)

        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_accuracy", test_acc)

        # Evaluation artifacts
        conf_matrix_fig = plot_confusion_matrix(y_test, test_preds)
        class_report = generate_classification_report(y_test, test_preds)
        roc_fig, roc_auc = plot_roc_curve(model, X_test, y_test)

        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_figure(conf_matrix_fig, "confusion_matrix.png")
        mlflow.log_text(class_report, "classification_report.txt")
        mlflow.log_figure(roc_fig, "roc_curve.png")

        # Save artifacts locally
        run_dir = artifacts_dir / model_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        conf_matrix_fig.savefig(run_dir / "confusion_matrix.png")
        (run_dir / "classification_report.txt").write_text(class_report)
        roc_fig.savefig(run_dir / "roc_curve.png")

        plt.close(conf_matrix_fig)
        plt.close(roc_fig)

        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, "model")

        metrics = {
            "cv_f1": best_cv_score,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
            "roc_auc": roc_auc,
        }

        params = {
            "model_name": model_name,
            "best_params": best_params,
            "class_weight": class_weight,
        }

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (run_dir / "params.json").write_text(json.dumps(params, indent=2))

        # Update latest pointer
        latest_path = artifacts_dir / model_name / "latest.json"
        latest_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "test_f1": test_f1,
                    "test_accuracy": test_acc,
                },
                indent=2,
            )
        )

        Logger.info(f"Model registered at {run_dir}")

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)


    return str(run_dir)
