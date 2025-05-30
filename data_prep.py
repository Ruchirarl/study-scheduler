import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import zipfile
from io import TextIOWrapper


def prepare_student_dataset():
    """
    Loads raw CSVs from data.zip, engineers features, and returns:
      - merged: pandas DataFrame ready for modeling
      - features: list of column names to use as model inputs
      - subject_feature_templates: dict mapping course code to mean feature vector
    """
    # Helper to read a CSV from the zip archive
    def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
        with zf.open(name) as f:
            return pd.read_csv(TextIOWrapper(f, "utf-8"))

    # Open the zip containing data
    with zipfile.ZipFile("data.zip") as z:
        info = _read_csv_from_zip(z, "studentInfo.csv")
        assessment = _read_csv_from_zip(z, "studentAssessment.csv")
        assessments_meta = _read_csv_from_zip(z, "assessments.csv")
        student_vle = _read_csv_from_zip(z, "studentVle.csv")

    # 1) Clean & merge assessment scores with metadata
    assessment["score"] = pd.to_numeric(assessment["score"], errors="coerce")
    assessment = assessment.merge(
        assessments_meta[
            ["id_assessment", "code_module", "code_presentation", "date", "weight", "assessment_type"]
        ],
        on="id_assessment",
        how="left"
    )

    # 2) Weighted average helper
    def safe_weighted_avg(x):
        w = assessment.loc[x.index, "weight"]
        if w.isnull().all() or w.sum() == 0:
            return x.mean()
        return np.average(x.fillna(0), weights=w.fillna(0))

    # 3) Aggregate per student-module
    agg_scores = (
        assessment
        .groupby(["id_student", "code_module"])
        .agg(
            avg_score=("score", "mean"),
            std_score=("score", "std"),
            count=("score", "count"),
            last_score=("score", lambda x: x.iloc[-1]),
            weighted_score=("score", safe_weighted_avg)
        )
        .reset_index()
    )

    # 4) Score trend
    assessment_sorted = assessment.sort_values(["id_student", "date"]);
    score_trend = (
        assessment_sorted
        .groupby("id_student")["score"]
        .apply(lambda x: np.polyfit(range(len(x)), x.fillna(0), 1)[0] if len(x) > 1 else 0)
        .reset_index(name="score_trend")
    )

    # 5) VLE features per student
    vle_features = (
        student_vle
        .groupby("id_student")
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            avg_clicks_per_day=("sum_click", lambda x: x.sum() / max(1, len(x))),
            click_variability=("sum_click", "std")
        )
        .reset_index()
    )

    # 6) First 14 and last 7 days
    first_14 = (
        student_vle[student_vle["date"] <= 14]
        .groupby("id_student")["sum_click"].sum()
        .reset_index(name="clicks_first_14_days")
    )
    last_7 = (
        student_vle[student_vle["date"] >= student_vle["date"].max() - 7]
        .groupby("id_student")["sum_click"].sum()
        .reset_index(name="clicks_last_7_days")
    )

    # 7) Encode demographics
    info_filtered = info[["id_student", "code_module", "final_result", "age_band", "highest_education", "imd_band"]].copy()
    encoder = OrdinalEncoder()
    info_filtered[["age_band", "highest_education", "imd_band"]] = encoder.fit_transform(
        info_filtered[["age_band", "highest_education", "imd_band"]].astype(str)
    )

    # 8) Merge all
    merged = agg_scores.merge(info_filtered, on=["id_student", "code_module"], how="left")
    merged = merged.merge(vle_features, on="id_student", how="left")
    merged = merged.merge(score_trend, on="id_student", how="left")
    merged = merged.merge(first_14, on="id_student", how="left")
    merged = merged.merge(last_7, on="id_student", how="left")

    # 9) Filter
    merged = merged[merged["final_result"] != "Withdrawn"]
    merged = merged[merged["count"] >= 2]
    merged = merged[merged["total_clicks"] > 0]

    # 10) Map target
    label_map = {"Fail": 0, "Pass": 1, "Distinction": 2}
    merged["target_class"] = merged["final_result"].map(label_map)

    # 11) Extra features
    merged["score_click_interaction"] = merged["avg_score"] * merged["total_clicks"]
    merged["click_std_ratio"] = merged["click_variability"] / (merged["avg_clicks_per_day"] + 1e-3)

    # 12) Feature list
    features = [
        "avg_score","std_score","count","last_score","score_trend","weighted_score",
        "total_clicks","active_days","avg_clicks_per_day","click_variability",
        "clicks_first_14_days","clicks_last_7_days",
        "age_band","highest_education","imd_band",
        "score_click_interaction","click_std_ratio"
    ]

    # 13) Subject templates
    subject_feature_templates = {}
    for subject in merged["code_module"].unique():
        df_sub = merged[merged["code_module"] == subject]
        subject_feature_templates[subject] = df_sub[features].mean().to_dict()

    return merged, features, subject_feature_templates
