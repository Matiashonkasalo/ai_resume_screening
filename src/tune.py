import optuna
from sklearn.model_selection import cross_val_score
from src.models import build_model


def get_search_space(trial, model_name: str):
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "random_state": 42,
            "n_jobs": -1,
        }

    if model_name == "adaboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "random_state": 42,
        }

    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
        }


def tune_model(X_train, y_train, model_name: str, n_trials: int = 30):
    def objective(trial):
        params = get_search_space(trial, model_name)
        pipeline = build_model(model_name, params)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring="f1",
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value
