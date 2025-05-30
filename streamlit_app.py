import streamlit as st
from datetime import date
from data_prep import prepare_student_dataset
from model_agent import train_model, SchedulerAgent, predict_subject_scores_pytorch

# Cache loading of data and model to speed up app start
@st.cache_data(show_spinner=False)
def load_resources():
    merged_df, features, subject_templates = prepare_student_dataset()
    model, pipeline_model = train_model(merged_df, features)
    return subject_templates, model, pipeline_model

# Load once
subject_templates, model, pipeline_model = load_resources()

# App UI
st.title("📚 Personalized Study Schedule Builder")

st.sidebar.header("Inputs")
# 1) Select courses
subjects = st.sidebar.multiselect(
    "Select your courses:",
    options=list(subject_templates.keys())
)

# 2) Exam dates for each selected course
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

# 4) Generate button
if st.sidebar.button("Generate Schedule"):
    if not subjects:
        st.warning("Please select at least one subject to schedule for.")
    else:
        # Prepare feature dict
        feature_dict = {s: subject_templates[s] for s in subjects}
        # Predict performance
        preds = predict_subject_scores_pytorch(model, pipeline_model, feature_dict)
        # Build schedule
        schedule_df = SchedulerAgent(preds, exam_dates, daily_hours).run()
        # Display
        st.subheader("🗓️ Your Study Schedule")
        st.dataframe(schedule_df)
