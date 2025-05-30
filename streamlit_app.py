import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# === 1) Data Ingestion & Preprocessing with PySpark ===
def load_and_preprocess():
    # Initialize Spark
    spark = SparkSession.builder.appName("ScheduleBuilder").getOrCreate()

    # Read OULAD CSVs
    info = spark.read.csv("studentInfo.csv", header=True, inferSchema=True)
    assess = spark.read.csv("studentAssessment.csv", header=True, inferSchema=True)
    meta = spark.read.csv("assessments.csv", header=True, inferSchema=True)
    vle = spark.read.csv("studentVle.csv", header=True, inferSchema=True)

    # Join and aggregate features
    df = (assess
          .join(meta, on="id_assessment")
          .groupBy("id_student","code_module")
          .agg(
              {'score':'mean', 'score':'std', 'score':'count'}
          )
          .withColumnRenamed("avg(score)","avg_score")
          .withColumnRenamed("stddev_samp(score)","std_score")
          .withColumnRenamed("count(score)","count_score")
    )
    # Add demographics
    info_idx = StringIndexer(inputCol="final_result", outputCol="target")
    df = df.join(info, on=["id_student","code_module"]).na.drop()

    # Spark ML pipeline for numeric features
    imputer = Imputer(inputCols=["avg_score","std_score","count_score"], outputCols=["avg_score","std_score","count_score"], strategy="mean")
    assembler = VectorAssembler(inputCols=["avg_score","std_score","count_score"], outputCol="features")
    pipeline = Pipeline(stages=[imputer, assembler, info_idx])
    model = pipeline.fit(df)
    transformed = model.transform(df)

    # Collect to pandas for TF
    pdf = transformed.select("features","target").toPandas()
    X = np.vstack(pdf['features'].values)
    y = keras.utils.to_categorical(pdf['target'], num_classes=3)
    return X, y, model

# === 2) Define TensorFlow Model ===
def build_tf_model(input_dim):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === 3) Scheduler Agent ===
def generate_schedule(predictions, exam_dates, daily_hours):
    # Simple priority: lower predicted class -> higher weight
    priorities = {s: (3 - p) for s,p in predictions.items()}
    days_left = {s: max(1,(date.fromisoformat(exam_dates[s]) - date.today()).days)
                 for s in predictions}
    total_pr = sum(priorities.values())
    schedule = []
    for subj, pr in priorities.items():
        hours = round(pr/total_pr * daily_hours * days_left[subj], 2)
        schedule.append((subj, predictions[subj], pr, days_left[subj], hours))
    return pd.DataFrame(schedule, columns=["Subject","PredictedClass","Priority","DaysLeft","Hours"])

# === 4) Streamlit App ===
def main():
    st.title("Agentic AI Study Scheduler")

    # Inputs
    st.sidebar.header("Settings")
    if st.sidebar.button("Load Data & Train"):  # expensive
        X, y, spark_pipeline = load_and_preprocess()
        tf_model = build_tf_model(X.shape[1])
        tf_model.fit(X, y, epochs=10, batch_size=128)
        st.success("Model trained!")
        st.session_state['model'] = (tf_model, spark_pipeline)
    
    if 'model' in st.session_state:
        tf_model, spark_pipeline = st.session_state['model']
        # Select modules
        modules = st.multiselect("Select your modules", options=spark_pipeline.stages[1].getInputCols())
        exam_dates = {m: st.sidebar.date_input(f"Exam date for {m}", date.today(), key=m).isoformat()
                      for m in modules}
        daily_hours = st.sidebar.slider("Daily study hours", 1, 24, 4)

        if st.sidebar.button("Generate Schedule"):
            # Prepare features via Spark pipeline
            # ... code to transform new subjects ...
            # Dummy predictions for illustration:
            preds = {m: np.argmax(tf_model.predict(np.random.rand(1, X.shape[1]))) for m in modules}
            schedule_df = generate_schedule(preds, exam_dates, daily_hours)
            st.write(schedule_df)
    else:
        st.info("Click 'Load Data & Train' first.")

if __name__ == '__main__':
    main()
