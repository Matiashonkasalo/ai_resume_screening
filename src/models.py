from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier



def build_model(model_name, params, class_weight=None):

    params = params.copy() 

    if model_name == "random_forest":

        if class_weight is not None:
            params["class_weight"] = class_weight

        model = RandomForestClassifier(**params)

    elif model_name == "adaboost":

        model = AdaBoostClassifier(**params)

    elif model_name == "xgboost":

        # XGBoost imbalance handling uses scale_pos_weight
        if class_weight == "balanced":
            params.setdefault("scale_pos_weight", 1)

        model = XGBClassifier(
            eval_metric="logloss",
            **params
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])


