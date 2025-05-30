import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, date
import zipfile
from io import TextIOWrapper

# --- Data Preparation ---
def load_data():
    """Load and preprocess data from data.zip."""
    # Read CSVs from ZIP
    with zipfile.ZipFile("data.zip") as z:
        def _read(name):
            with z.open(name) as f:
                return pd.read_csv(TextIOWrapper(f, "utf-8"))
        info = _read("studentInfo.csv")
        assessment = _read("studentAssessment.csv")
        assessments_meta = _read("assessments.csv")
        vle = _read("studentVle.csv")

    # Merge assessment and metadata
    assessment['score'] = pd.to_numeric(assessment['score'], errors='coerce')
    assessment = assessment.merge(
        assessments_meta[['id_assessment','code_module','date','weight']],
        on='id_assessment', how='left')

    # Aggregate scores per student-module
    agg = assessment.groupby(['id_student','code_module']).agg(
        avg_score=('score','mean'),
        std_score=('score','std'),
        count=('score','count'),
        last_score=('score', lambda x: x.iloc[-1]),
        weighted_score=('score', lambda x: np.average(x.fillna(0),weights=assessment.loc[x.index,'weight'].fillna(0)))
    ).reset_index()

    # VLE features
    vle_feats = vle.groupby('id_student').agg(
        total_clicks=('sum_click','sum'),
        active_days=('date','nunique'),
        avg_clicks_per_day=('sum_click', lambda x: x.sum()/max(1,len(x))),
        click_std=('sum_click','std')
    ).reset_index()

    # Encode demographics
    info_f = info[['id_student','code_module','final_result','age_band','highest_education','imd_band']].copy()
    enc = OrdinalEncoder()
    info_f[['age_band','highest_education','imd_band']] = enc.fit_transform(info_f[['age_band','highest_education','imd_band']].astype(str))

    # Merge all
    df = agg.merge(info_f, on=['id_student','code_module'], how='left')
    df = df.merge(vle_feats, on='id_student', how='left')
    df = df[df['final_result']!='Withdrawn']
    df = df[df['count']>=2]
    df = df[df['total_clicks']>0]
    df['target_class'] = df['final_result'].map({'Fail':0,'Pass':1,'Distinction':2})

    # Interaction features
    df['score_click'] = df['avg_score'] * df['total_clicks']
    df['click_ratio'] = df['click_std']/(df['avg_clicks_per_day']+1e-3)

    # Feature list
    features = [
        'avg_score','std_score','count','last_score','weighted_score',
        'total_clicks','active_days','avg_clicks_per_day','click_std','score_click','click_ratio',
        'age_band','highest_education','imd_band'
    ]

    # Templates per module
    templates = {mod: df[df['code_module']==mod][features].mean().to_dict()
                 for mod in df['code_module'].unique()}
    return df, features, templates

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,num_classes)
        )
    def forward(self,x): return self.net(x)

@st.cache(allow_output_mutation=True)
def train_model(df, features, epochs=100, lr=0.001):
    """Train PyTorch MLP with pandas+sklearn preprocessing"""
    X = df[features].values
    y = df['target_class'].values.astype(int)
    imp = SimpleImputer(strategy='mean')
    scl = StandardScaler()
    X_imp = imp.fit_transform(X)
    X_scaled = scl.fit_transform(X_imp)
    Xt = torch.tensor(X_scaled, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model = MLP(Xt.shape[1], len(np.unique(y)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward(); opt.step()
    return model, imp, scl

# --- SchedulerAgent ---
class SchedulerAgent:
    def __init__(self, preds, exams, daily_hours):
        self.preds, self.exams, self.hours = preds, exams, daily_hours
    def run(self):
        pr = {s: (max(self.preds.values())-v+1) for s,v in self.preds.items()}
        dl = {s: max(1,(datetime.strptime(d,'%Y-%m-%d').date()-date.today()).days)
              for s,d in self.exams.items()}
        total = sum(pr.values())
        alloc = {s: round(pr[s]/total*self.hours*dl[s],2) for s in pr} if total else {s:0 for s in pr}
        return pd.DataFrame({'Subject':list(self.preds),'Predicted Class':list(self.preds.values()),
                             'Priority':list(pr.values()),'Days Left':list(dl.values()),
                             'Hours Assigned':list(alloc.values())})

# --- App UI ---
def main():
    df, features, templates = load_data()
    model, imp, scl = train_model(df, features)
    st.title('Study Schedule Builder')
    subj = st.multiselect('Choose courses', list(templates.keys()))
    exams = {}
    for s in subj:
        exams[s] = st.date_input(f'Exam date for {s}', date.today(), key=s).strftime('%Y-%m-%d')
    hours = st.slider('Hours per day', 1,24,4)
    if st.button('Generate Schedule'):
        feats = {s:templates[s] for s in subj}
        arr = np.array([list(feats[s].values()) for s in subj])
        arr_imp = imp.transform(arr); arr_scaled = scl.transform(arr_imp)
        preds = {s:int(torch.argmax(model(torch.tensor(arr_scaled,dtype=torch.float32))[i]))
                 for i,s in enumerate(subj)}
        sched = SchedulerAgent(preds, exams, hours).run()
        st.write(sched)

if __name__=='__main__':
    main()
