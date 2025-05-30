# Streamlit App with PySpark ML for Prediction
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from io import TextIOWrapper
from datetime import date
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- 1) Load & Preprocess with PySpark ---
def load_and_train_spark():
    # Unzip CSVs into pandas
    with zipfile.ZipFile('data.zip') as zf:
        def _read(fn):
            with zf.open(fn) as f:
                return pd.read_csv(TextIOWrapper(f, 'utf-8'))
        info   = _read('studentInfo.csv')
        assess = _read('studentAssessment.csv')
        meta   = _read('assessments.csv')
        vle    = _read('studentVle.csv')

    # Merge and feature engineering in pandas
    assess = assess.merge(meta[['id_assessment','code_module','weight']], on='id_assessment')
    assess['score'] = pd.to_numeric(assess['score'], errors='coerce')
    agg = assess.groupby(['id_student','code_module']).agg(
        avg_score=('score','mean'),
        std_score=('score','std'),
        count_score=('score','count')
    ).reset_index()
    vle_feats = vle.groupby('id_student').agg(
        total_clicks=('sum_click','sum'),
        active_days=('date','nunique')
    ).reset_index()
    info = info[['id_student','code_module','final_result']]
    info = info[info['final_result']!='Withdrawn'].copy()

    merged = (agg
              .merge(info, on=['id_student','code_module'])
              .merge(vle_feats, on='id_student', how='left')
             ).dropna()
    merged['target_class'] = merged['final_result'].map({'Fail':0,'Pass':1,'Distinction':2})

    # Convert to Spark
    spark = SparkSession.builder.appName('SchedulerApp').getOrCreate()
    sdf = spark.createDataFrame(merged)

    # Build Spark ML pipeline
    features = ['avg_score','std_score','count_score','total_clicks','active_days']
    imputer = Imputer(inputCols=features, outputCols=features)
    assembler = VectorAssembler(inputCols=features, outputCol='features_vec')
    scaler = StandardScaler(inputCol='features_vec', outputCol='features')
    indexer = StringIndexer(inputCol='target_class', outputCol='label')
    rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=50)
    pipeline = Pipeline(stages=[imputer, assembler, scaler, indexer, rf])

    # Train/test split
    train, test = sdf.randomSplit([0.8,0.2], seed=42)
    model = pipeline.fit(train)
    pred = model.transform(test)

    # Collect predictions
    pdf = pred.select('label','prediction').toPandas()
    acc = accuracy_score(pdf['label'], pdf['prediction'])
    report = classification_report(pdf['label'], pdf['prediction'],
                                   target_names=['Fail','Pass','Distinction'], output_dict=True)
    return model, features, acc, report, spark, merged

# --- 2) Schedule Generation ---
def generate_schedule(preds, exam_dates, daily_hours):
    prios = {s:(3-p) for s,p in preds.items()}
    days = {s:max(1,(date.fromisoformat(exam_dates[s])-date.today()).days)
            for s in preds}
    total = sum(prios.values())
    rows=[]
    for subj, p in prios.items():
        hrs = round(p/total * daily_hours * days[subj],2)
        rows.append({'Subject':subj,'Priority':p,'DaysLeft':days[subj],'Hours':hrs})
    return pd.DataFrame(rows)

# --- 3) Streamlit UI ---
def main():
    st.title('📚 Agentic AI Study Scheduler')

    if 'loaded' not in st.session_state:
        st.subheader('Training with PySpark ML')
        model, features, acc, report, spark, merged = load_and_train_spark()
        st.write(f"**Test Accuracy:** {acc:.2%}")
        st.json(report)
        st.session_state.update({
            'model': model,
            'features': features,
            'spark': spark,
            'data': merged
        })
        st.session_state['loaded']=True

    if st.session_state.get('loaded'):
        st.subheader('Build Your Schedule')
        modules = st.multiselect('Select modules',
                                  st.session_state['data']['code_module'].unique().tolist())
        exam_dates = {m:st.date_input(f'Exam date for {m}', date.today(), key=m).isoformat()
                      for m in modules}
        hours = st.slider('Hours per day',1,24,4)
        if st.button('Generate Schedule'):
            model = st.session_state['model']
            features = st.session_state['features']
            spark = st.session_state['spark']
            df = st.session_state['data']
            # Predict on average features per module
            templates = {m:df[df['code_module']==m][features].mean().to_dict() for m in modules}
            pdf_new = pd.DataFrame.from_dict(templates, orient='index')
            sdf_new = spark.createDataFrame(pdf_new.reset_index().rename(columns={'index':'code_module'}))
            transformed = model.transform(sdf_new)
            preds = {row['code_module']:int(row['prediction']) for row in transformed.collect()}
            sched = generate_schedule(preds, exam_dates, hours)
            st.dataframe(sched)

if __name__=='__main__':
    main()
