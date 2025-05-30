import os, sys
# Tell PySpark to use this Python interpreter
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from datetime import datetime
import builtins

# === PyTorch MLP model ===
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# === Training helper using PySpark ===
def train_model(merged_df: pd.DataFrame, features: list, label_col: str = 'target_class', epochs: int = 100, lr: float = 0.001):
    """
    Builds a Spark preprocessing pipeline, trains the PyTorch MLP, and returns (model, pipeline).
    Requires Java JDK installed and on PATH.
    """
    spark = SparkSession.builder.appName("train_model").getOrCreate()
    spark_df = spark.createDataFrame(merged_df.dropna(subset=[label_col]))

    # Spark ML pipeline: impute, assemble, scale
    imputer = Imputer(inputCols=features, outputCols=features)
    assembler = VectorAssembler(inputCols=features, outputCol="features_vec")
    scaler = StandardScaler(inputCol="features_vec", outputCol="features")
    pipeline = Pipeline(stages=[imputer, assembler, scaler])
    fitted_pipeline = pipeline.fit(spark_df)
    final_df = fitted_pipeline.transform(spark_df).select("features", label_col)

    # Convert to torch tensors
    X = np.array(final_df.rdd.map(lambda r: r[0].toArray()).collect())
    y = np.array(final_df.rdd.map(lambda r: int(r[1])).collect())
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Model setup
    model = MLP(input_size=X_tensor.shape[1], num_classes=len(np.unique(y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    return model, fitted_pipeline

# === Inference helper ===
def predict_subject_scores_pyspark(model, pipeline_model, subject_feature_dict: dict):
    """
    Predict classes using the trained model and Spark pipeline.
    """
    spark = SparkSession.builder.getOrCreate()
    predictions = {}
    for subj_code, features in subject_feature_dict.items():
        pdf = pd.DataFrame([features])
        sdf = spark.createDataFrame(pdf)
        transformed = pipeline_model.transform(sdf).select("features")
        X_np = np.array(transformed.rdd.map(lambda r: r[0].toArray()).collect())
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_tensor)
            pred = torch.argmax(logits, axis=1).item()
        predictions[subj_code] = pred
    return predictions

# === SchedulerAgent ===
class SchedulerAgent:
    def __init__(self, predicted_scores: dict, exam_dates: dict, total_daily_hours: int):
        self.predicted_scores = {subj: int(score) for subj, score in predicted_scores.items()}
        self.exam_dates = exam_dates
        self.total_daily_hours = total_daily_hours

    def _calculate_priority_scores(self):
        max_score = max(self.predicted_scores.values(), default=2)
        return {subj: (max_score - score + 1) for subj, score in self.predicted_scores.items()}

    def _calculate_days_until_exam(self):
        today = datetime.today().date()
        return {subj: max(1, (datetime.strptime(date, "%Y-%m-%d").date() - today).days)
                for subj, date in self.exam_dates.items()}

    def _compute_total_hours(self, priority_scores, days_left):
        total_priority = builtins.sum(priority_scores.values())
        if total_priority == 0:
            return {subj: 0 for subj in priority_scores}
        return {subj: round((priority_scores[subj] / total_priority) * self.total_daily_hours * days_left[subj], 2)
                for subj in priority_scores}

    def run(self):
        priority = self._calculate_priority_scores()
        days = self._calculate_days_until_exam()
        hours = self._compute_total_hours(priority, days)

        return pd.DataFrame({
            "Subject": list(self.predicted_scores.keys()),
            "Predicted Class": [self.predicted_scores[s] for s in self.predicted_scores],
            "Priority Score": [priority[s] for s in self.predicted_scores],
            "Days Until Exam": [days[s] for s in self.predicted_scores],
            "Total Hours Assigned": [hours[s] for s in self.predicted_scores]
        }).sort_values(by="Priority Score", ascending=False).reset_index(drop=True)
