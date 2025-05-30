import os, sys
# Ensure Spark uses this Python interpreter (for PySpark workers)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import streamlit as st
from datetime import date
from data_prep import prepare_student_dataset
from model_agent import train_model, SchedulerAgent, predict_subject_scores_pyspark

# Load data & model once and cache
@st.cache_data(show_spinner=False)
def load_resources():
    merged_df, features, subject_templates = prepare_student_dataset()
    model, pipeline_model = train_model(merged_df, features)
    return subject_templates, model, pipeline_model

subject_templates, model, pipeline_model = load_resources()

st.title("📚 Personalized Study Schedule Builder")

st.sidebar.header("Inputs")

# 1) Pick subjects
subjects = st.sidebar.multiselect(
    "Select your courses:",
    options=list(subject_templates.keys())
)

# 2) Exam dates for each subject
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

# 4) When clicked, run scheduler
if st.sidebar.button("Generate Schedule"):
    if not subjects:
        st.warning("Please select at least one subject to schedule for.")
    else:
        feature_dict = {s: subject_templates[s] for s in subjects}
        preds = predict_subject_scores_pyspark(model, pipeline_model, feature_dict)
        schedule_df = SchedulerAgent(preds, exam_dates, daily_hours).run()
        st.subheader("🗓️ Your Study Schedule")
        st.dataframe(schedule_df)
