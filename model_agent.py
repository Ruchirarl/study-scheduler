import os, sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml import Pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler as SklearnScaler
from datetime import datetime
import builtins

# === Ensure PySpark uses this Python interpreter ===
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# === PyTorch MLP Model ===
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

# === Spark-based Training ===
def train_model_pyspark(merged_df: pd.DataFrame, features: list, label_col: str = 'target_class', epochs: int = 100, lr: float = 0.001):
    """
    Train using PySpark pipeline + PyTorch MLP. Returns (model, spark_pipeline).
    """
    spark = SparkSession.builder.appName("schedule_builder").getOrCreate()
    sdf = spark.createDataFrame(merged_df.dropna(subset=[label_col]))

    # Spark ML stages
    imputer = Imputer(inputCols=features, outputCols=features)
    assembler = VectorAssembler(inputCols=features, outputCol="features_vec")
    scaler = SparkScaler(inputCol="features_vec", outputCol="features")
    pipe = Pipeline(stages=[imputer, assembler, scaler])
    fitted_pipe = pipe.fit(sdf)
    transformed = fitted_pipe.transform(sdf).select("features", label_col)

    # Collect for PyTorch
    arr = np.array(transformed.rdd.map(lambda r: r[0].toArray()).collect())
    labels = np.array(transformed.rdd.map(lambda r: int(r[1])).collect())
    X = torch.tensor(arr, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model = MLP(X.shape[1], len(np.unique(labels)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
    return model, fitted_pipe

# === Spark-based Prediction ===
def predict_subject_scores_pyspark(model, pipeline_model, feature_dict: dict):
    spark = SparkSession.builder.getOrCreate()
    preds = {}
    for subj, feats in feature_dict.items():
        pdf = pd.DataFrame([feats])
        sdf = spark.createDataFrame(pdf)
        transformed = pipeline_model.transform(sdf).select("features")
        arr = np.array(transformed.rdd.map(lambda r: r[0].toArray()).collect())
        X = torch.tensor(arr, dtype=torch.float32)
        with torch.no_grad():
            p = torch.argmax(model(X), dim=1).item()
        preds[subj] = p
    return preds

# === Pandas+sklearn-based Training ===
def train_model_pandas(merged_df: pd.DataFrame, features: list, label_col: str = 'target_class', epochs: int = 100, lr: float = 0.001):
    X_np = merged_df[features].values
    y_np = merged_df[label_col].values.astype(int)
    imp = SimpleImputer(strategy='mean')
    scl = SklearnScaler()
    X_imp = imp.fit_transform(X_np)
    X_scaled = scl.fit_transform(X_imp)
    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    model = MLP(X.shape[1], len(np.unique(y_np)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
    return model, (imp, scl)

# === Pandas+sklearn-based Prediction ===
def predict_subject_scores_pandas(model, preprocessors, feature_dict: dict):
    imp, scl = preprocessors
    preds = {}
    for subj, feats in feature_dict.items():
        arr = np.array([list(feats.values())])
        X_imp = imp.transform(arr)
        X_scaled = scl.transform(X_imp)
        X = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            p = torch.argmax(model(X), dim=1).item()
        preds[subj] = p
    return preds

# === SchedulerAgent ===
class SchedulerAgent:
    def __init__(self, predicted_scores: dict, exam_dates: dict, total_daily_hours: int):
        self.scores = {k: int(v) for k, v in predicted_scores.items()}
        self.dates = exam_dates
        self.hours = total_daily_hours
    def _priority(self):
        m = max(self.scores.values())
        return {s: (m - v + 1) for s, v in self.scores.items()}
    def _days_left(self):
        today = datetime.today().date()
        return {s: max(1, (datetime.strptime(d, "%Y-%m-%d").date() - today).days)
                for s, d in self.dates.items()}
    def _alloc(self, pr, dl):
        total = builtins.sum(pr.values())
        if total == 0: return {s:0 for s in pr}
        return {s: round(pr[s]/total * self.hours * dl[s],2) for s in pr}
    def run(self):
        pr = self._priority()
        dl = self._days_left()
        ah = self._alloc(pr, dl)
        return pd.DataFrame({
            "Subject": list(self.scores.keys()),
            "Predicted Class": list(self.scores.values()),
            "Priority Score": [pr[s] for s in self.scores],
            "Days Until Exam": [dl[s] for s in self.scores],
            "Total Hours Assigned": [ah[s] for s in self.scores]
        }).sort_values(by="Priority Score", ascending=False).reset_index(drop=True)
