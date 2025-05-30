import os, sys
# Determine if running on Streamlit Cloud
USE_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

# If using PySpark, ensure it uses this Python
if not USE_CLOUD:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import streamlit as st
from datetime import date
from data_prep import prepare_student_dataset

# Import both Spark and pandas-based training/inference
from model_agent import (
    train_model_pyspark,
    predict_subject_scores_pyspark,
    train_model_pandas,
    predict_subject_scores_pandas,
    SchedulerAgent
)

st.title("📚 Personalized Study Schedule Builder")

@st.cache_data(show_spinner=False)
def load_resources():
    merged_df, features, subject_templates = prepare_student_dataset()
    if USE_CLOUD:
        # Fall back to pandas+sklearn pipeline on Cloud
        model, preprocessors = train_model_pandas(merged_df, features)
        predict_fn = lambda fd: predict_subject_scores_pandas(model, preprocessors, fd)
    else:
        # Use PySpark locally
        model, spark_pipeline = train_model_pyspark(merged_df, features)
        predict_fn = lambda fd: predict_subject_scores_pyspark(model, spark_pipeline, fd)
    return subject_templates, predict_fn

subject_templates, predict_fn = load_resources()

st.sidebar.header("Inputs")

# 1) Select subjects
subjects = st.sidebar.multiselect(
    "Select your courses:",
    options=list(subject_templates.keys())
)

# 2) Exam dates
exam_dates = {}
for subj in subjects:
    exam_dates[subj] = st.sidebar.date_input(
        f"Exam date for {subj}",
        value=date.today(),
        key=subj
    ).strftime("%Y-%m-%d")

# 3) Daily study hours
daily_hours = st.sidebar.number_input(
    "Hours you can study each day:",
    min_value=1, max_value=24, value=4
)

# 4) Generate schedule
if st.sidebar.button("Generate Schedule"):
    if not subjects:
        st.warning("Please select at least one subject.")
    else:
        feature_dict = {s: subject_templates[s] for s in subjects}
        preds = predict_fn(feature_dict)
        schedule_df = SchedulerAgent(preds, exam_dates, daily_hours).run()
        st.subheader("🗓️ Your Study Schedule")
        st.dataframe(schedule_df)
