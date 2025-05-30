# Standalone Streamlit App: no external modules needed
# It trains and evaluates a TensorFlow model, then provides an interactive scheduling UI

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import zipfile
from io import TextIOWrapper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
import tensorflow as tf

# --- 1) Load & Preprocess Data ---
def load_data(zip_path="data.zip"):
    """Read CSVs from ZIP, engineer features, and return DataFrame + feature list"""
    with zipfile.ZipFile(zip_path) as zf:
        def _read(fn):
            with zf.open(fn) as f:
                return pd.read_csv(TextIOWrapper(f, 'utf-8'))
        info    = _read("studentInfo.csv")
        assess  = _read("studentAssessment.csv")
        meta    = _read("assessments.csv")
        vle     = _read("studentVle.csv")

    # Merge assessment & metadata
    assess = pd.merge(assess, meta[['id_assessment','code_module','date','weight']],
                      on='id_assessment', how='left')
    assess['score'] = pd.to_numeric(assess['score'], errors='coerce')

    # Aggregate per student-module
    agg = assess.groupby(['id_student','code_module']).agg(
        avg_score=('score','mean'),
        std_score=('score','std'),
        count_score=('score','count'),
        last_score=('score', lambda x: x.iloc[-1]),
        weighted_score=('score', lambda x: np.average(x.fillna(0),
                                                       weights=assess.loc[x.index,'weight'].fillna(0)))
    ).reset_index()

    # VLE click features
    vle_feats = vle.groupby('id_student').agg(
        total_clicks=('sum_click','sum'),
        active_days=('date','nunique'),
        avg_clicks_per_day=('sum_click', lambda x: x.sum()/max(1,len(x))),
        click_std=('sum_click','std')
    ).reset_index()

    # Demographics encoding
    info = info[['id_student','code_module','final_result','age_band','highest_education','imd_band']]
    info['final_result'] = info['final_result'].replace('Withdrawn', np.nan)
    info = info.dropna(subset=['final_result'])
    enc = OrdinalEncoder()
    info[['age_band','highest_education','imd_band']] = enc.fit_transform(
        info[['age_band','highest_education','imd_band']].astype(str)
    )

    # Merge all
    df = (agg
          .merge(info, on=['id_student','code_module'], how='inner')
          .merge(vle_feats, on='id_student', how='left')
         )
    df = df.dropna()
    df['target_class'] = df['final_result'].map({'Fail':0,'Pass':1,'Distinction':2})

    # Interaction features
    df['score_click'] = df['avg_score'] * df['total_clicks']
    df['click_ratio'] = df['click_std']/(df['avg_clicks_per_day']+1e-3)

    # Feature list
    features = [
        'avg_score','std_score','count_score','last_score','weighted_score',
        'total_clicks','active_days','avg_clicks_per_day','click_std',
        'age_band','highest_education','imd_band','score_click','click_ratio'
    ]
    return df, features

# --- 2) Train & Evaluate TensorFlow Model ---
def train_and_evaluate(df, features, epochs=20):
    """Split data, train MLP, and return model, scaler, test accuracy, and report"""
    X = df[features].values
    y = df['target_class'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    # One-hot encode labels
    y_train_o = keras.utils.to_categorical(y_train,3)
    y_test_o  = keras.utils.to_categorical(y_test,3)

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train_o, epochs=epochs, batch_size=128, verbose=0)

    preds = np.argmax(model.predict(X_test_s), axis=1)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds,
                                   target_names=['Fail','Pass','Distinction'],
                                   output_dict=True)
    return model, scaler, acc, report

# --- 3) Schedule Generation ---
def generate_schedule(preds_dict, exam_dates, daily_hours):
    priorities = {s: (3 - p) for s,p in preds_dict.items()}
    days_left = {s: max(1, (date.fromisoformat(exam_dates[s]) - date.today()).days)
                 for s in preds_dict}
    total_p = sum(priorities.values())
    schedule = []
    for subj, pr in priorities.items():
        hrs = round(pr/total_p * daily_hours * days_left[subj],2)
        schedule.append({'Subject':subj,'PredictedClass':preds_dict[subj],
                         'Priority':pr,'DaysLeft':days_left[subj],'Hours':hrs})
    return pd.DataFrame(schedule)

# --- 4) Streamlit App ---
def main():
    st.title('📚 AI-Powered Study Scheduler')

    # 4.1 Data & Model
    df, features = load_data()
    st.subheader('Data Preview')
    st.dataframe(df.head())

    if 'trained' not in st.session_state:
        st.subheader('Model Training & Evaluation')
        model, scaler, acc, report = train_and_evaluate(df, features)
        st.write(f'**Test Accuracy:** {acc:.2%}')
        st.json(report)
        st.session_state['model_info'] = (model, scaler)
        st.session_state['trained'] = True

    # 4.2 Interactive Schedule
    if st.session_state.get('trained'):
        st.subheader('Build Your Schedule')
        modules = st.multiselect('Select courses', df['code_module'].unique().tolist())
        exam_dates = {m: st.date_input(f'Exam date for {m}', date.today(), key=m).isoformat()
                      for m in modules}
        daily_hours = st.slider('Hours per day', 1,24,4)
        if st.button('Generate Schedule'):
            model, scaler = st.session_state['model_info']
            # Predict for selected modules
            templates = {m: df[df['code_module']==m][features].mean().to_dict() for m in modules}
            X_new = np.array([list(templates[m].values()) for m in modules])
            X_new_s = scaler.transform(X_new)
            preds = np.argmax(model.predict(X_new_s), axis=1)
            preds_dict = dict(zip(modules, preds))
            sched_df = generate_schedule(preds_dict, exam_dates, daily_hours)
            st.subheader('Recommended Schedule')
            st.dataframe(sched_df)

if __name__=='__main__':
    main()
