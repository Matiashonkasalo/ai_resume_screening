from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging

Logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = {
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
    "github_activity",
    "shortlisted",
}

EDUCATION_LEVELS = {"High School", "Bachelor", "Master", "PhD"}
SHORTLISTED_VALUES = {"Yes", "No"}


def pass_check(report, name, details=None):
    report["checks"][name] = {
        "passed": True,
        "details": details,
    }


def fail_check(report, name, details=None):
    report["checks"][name] = {
        "passed": False,
        "details": details,
    }
    report["status"] = "FAIL"



# Main validation function

def validate_dataset(data_path):
    Logger.info(f"Starting validation for {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)

    report = {
        "file": str(data_path),
        "validated_at": datetime.utcnow().isoformat(),
        "row_count": len(df),
        "column_count": df.shape[1],
        "checks": {},
        "status": "PASS",
    }

   
    # Basic sanity
   
    if df.empty:
        raise ValueError("Dataset is empty")

    pass_check(report, "dataset_not_empty", {"rows": len(df)})

   
    # Schema check

    missing_cols = EXPECTED_SCHEMA - set(df.columns)
    extra_cols = set(df.columns) - EXPECTED_SCHEMA

    if missing_cols:
        fail_check(
            report,
            "schema",
            {
                "missing_columns": list(missing_cols),
                "extra_columns": list(extra_cols),
            },
        )
    else:
        pass_check(
            report,
            "schema",
            {
                "extra_columns": list(extra_cols),
            },
        )

    # Null values 
   
    null_counts = df.isnull().sum().to_dict()

    if any(v > 0 for v in null_counts.values()):
        pass_check(
            report,
            "nulls",
            {
                "null_counts": null_counts,
                "warning": True,
            },
        )
        Logger.warning(f"Null values found: {null_counts}")
    else:
        pass_check(report, "nulls")

   
    # Numeric sanity 
    
    critical_numeric = {
        "years_experience_negative": int((df["years_experience"] < 0).sum()),
        "project_count_negative": int((df["project_count"] < 0).sum()),
        "github_activity_negative": int((df["github_activity"] < 0).sum()),
    }

    if any(v > 0 for v in critical_numeric.values()):
        fail_check(report, "numeric_critical", critical_numeric)
    else:
        pass_check(report, "numeric_critical")

    # Numeric sanity 
  
    non_critical_numeric = {
        "skills_score_out_of_range": int(
            ((df["skills_match_score"] < 0) | (df["skills_match_score"] > 1)).sum()
        ),
        "resume_length_non_positive": int((df["resume_length"] <= 0).sum()),
    }

    pass_check(
        report,
        "numeric_non_critical",
        {
            "issues": non_critical_numeric,
            "warning": True,
        },
    )

    if any(v > 0 for v in non_critical_numeric.values()):
        Logger.warning(f"Non-critical numeric issues found: {non_critical_numeric}")

   
    # Categorical sanity (WARNING only)
   
    invalid_education = set(df["education_level"].unique()) - EDUCATION_LEVELS
    invalid_shortlisted = set(df["shortlisted"].unique()) - SHORTLISTED_VALUES

    pass_check(
        report,
        "categorical_values",
        {
            "invalid_education_level": list(invalid_education),
            "invalid_shortlisted": list(invalid_shortlisted),
            "warning": True,
        },
    )

    if invalid_education or invalid_shortlisted:
        Logger.warning(
            f"Unexpected categorical values found. "
            f"Education: {invalid_education}, Shortlisted: {invalid_shortlisted}"
        )

    # Final summary
   
    Logger.info("Validation summary:")
    Logger.info(json.dumps(report, indent=2))

    Logger.info(f"Validation finished with status: {report['status']}")

    return report



# Write validation report

def write_validation_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    return output_path
