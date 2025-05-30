import os, sys
import streamlit as st
from datetime import date
from data_prep import prepare_student_dataset
from model_agent import (
    train_model_pyspark,
    predict_subject_scores_pyspark,
    train_model_pandas,
    predict_subject_scores_pandas,
    SchedulerAgent
)

st.title("📚 Personalized Study Schedule Builder")
st.sidebar.header("Inputs")

# Helper to load and train the model on each run (no caching)
def load_resources():
    """
    Attempt Spark-based training; on failure use pandas/Sklearn.
    Returns subject_templates dict and prediction function.
    """
    merged_df, features, subject_templates = prepare_student_dataset()
    # Try Spark
    try:
        # Ensure Spark uses this Python interpreter
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        model, spark_pipeline = train_model_pyspark(merged_df, features)
        predict_fn = lambda fd: predict_subject_scores_pyspark(model, spark_pipeline, fd)
    except Exception as e:
        st.warning(f"⚠️ Spark failed, using pandas pipeline. Error: {e}")
        model, preprocessors = train_model_pandas(merged_df, features)
        predict_fn = lambda fd: predict_subject_scores_pandas(model, preprocessors, fd)
    return subject_templates, predict_fn

# Load resources (no caching to avoid pickling issues)
subject_templates, predict_fn = load_resources()

# Sidebar inputs
subjects = st.sidebar.multiselect(
    "Select your courses:",
    options=list(subject_templates.keys())
)

exam_dates = {}
for subj in subjects:
    exam_dates[subj] = st.sidebar.date_input(
        f"Exam date for {subj}",
        value=date.today(),
        key=subj
    ).strftime("%Y-%m-%d")

daily_hours = st.sidebar.number_input(
    "Hours you can study each day:",
    min_value=1, max_value=24, value=4
)

# Generate and display schedule
if st.sidebar.button("Generate Schedule"):
    if not subjects:
        st.warning("Please select at least one subject.")
    else:
        # Build feature dict and predict
        feature_dict = {s: subject_templates[s] for s in subjects}
        preds = predict_fn(feature_dict)
        # Build schedule DataFrame
        schedule_df = SchedulerAgent(preds, exam_dates, daily_hours).run()
        st.subheader("🗓️ Your Study Schedule")
        st.dataframe(schedule_df)