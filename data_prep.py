import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def prepare_student_dataset():
    """
    Loads raw CSVs, engineers features, and returns:
      - merged:   a pandas DataFrame ready for modeling
      - features: list of column names to use as model inputs
      - subject_feature_templates: 
            dict mapping each course code to the per-course–average feature vector
    """
    # 1) Load data
    info = pd.read_csv("studentInfo.csv")
    assessment = pd.read_csv("studentAssessment.csv")
    assessments_meta = pd.read_csv("assessments.csv")
    student_vle = pd.read_csv("studentVle.csv")

    # 2) Clean & merge assessment scores with metadata
    assessment["score"] = pd.to_numeric(assessment["score"], errors="coerce")
    assessment = assessment.merge(
        assessments_meta[
            ["id_assessment", "code_module", "code_presentation", "date", "weight", "assessment_type"]
        ],
        on="id_assessment",
        how="left"
    )

    # 3) Helper for weighted average
    def safe_weighted_avg(x):
        w = assessment.loc[x.index, "weight"]
        if w.isnull().all() or w.sum() == 0:
            return x.mean()
        return np.average(x.fillna(0), weights=w.fillna(0))

    # 4) Aggregate per student-module
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

    # 5) Compute score trend (slope) over time for each student
    assessment_sorted = assessment.sort_values(["id_student", "date"])
    score_trend = (
        assessment_sorted
        .groupby("id_student")["score"]
        .apply(lambda x: np.polyfit(range(len(x)), x.fillna(0), 1)[0] if len(x) > 1 else 0)
        .reset_index(name="score_trend")
    )

    # 6) VLE (click) features per student
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

    # 7) First 14 days and last 7 days clicks
    first_14 = (
        student_vle[student_vle["date"] <= 14]
        .groupby("id_student")["sum_click"]
        .sum()
        .reset_index(name="clicks_first_14_days")
    )
    last_7 = (
        student_vle[student_vle["date"] >= student_vle["date"].max() - 7]
        .groupby("id_student")["sum_click"]
        .sum()
        .reset_index(name="clicks_last_7_days")
    )

    # 8) Encode demographics
    info_filtered = info[
        ["id_student", "code_module", "final_result", "age_band", "highest_education", "imd_band"]
    ].copy()
    encoder = OrdinalEncoder()
    info_filtered[["age_band", "highest_education", "imd_band"]] = encoder.fit_transform(
        info_filtered[["age_band", "highest_education", "imd_band"]].astype(str)
    )

    # 9) Merge everything together
    merged = agg_scores.merge(info_filtered, on=["id_student", "code_module"], how="left")
    merged = merged.merge(vle_features, on="id_student", how="left")
    merged = merged.merge(score_trend, on="id_student", how="left")
    merged = merged.merge(first_14, on="id_student", how="left")
    merged = merged.merge(last_7, on="id_student", how="left")

    # 10) Filter out Withdrawn & too-few data
    merged = merged[merged["final_result"] != "Withdrawn"]
    merged = merged[merged["count"] >= 2]
    merged = merged[merged["total_clicks"] > 0]

    # 11) Map final results to numeric target
    label_map = {"Fail": 0, "Pass": 1, "Distinction": 2}
    merged["target_class"] = merged["final_result"].map(label_map)

    # 12) Additional interaction features
    merged["score_click_interaction"] = merged["avg_score"] * merged["total_clicks"]
    merged["click_std_ratio"] = merged["click_variability"] / (merged["avg_clicks_per_day"] + 1e-3)

    # 13) Define the feature list for modeling
    features = [
        "avg_score", "std_score", "count", "last_score", "score_trend", "weighted_score",
        "total_clicks", "active_days", "avg_clicks_per_day", "click_variability",
        "clicks_first_14_days", "clicks_last_7_days",
        "age_band", "highest_education", "imd_band",
        "score_click_interaction", "click_std_ratio"
    ]

    # 14) Build per-subject “template” = mean feature vector
    subject_feature_templates = {}
    for subject in merged["code_module"].unique():
        df_sub = merged[merged["code_module"] == subject]
        subject_feature_templates[subject] = df_sub[features].mean().to_dict()

    return merged, features, subject_feature_templates
