import nbformat as nbf

nb = nbf.v4.new_notebook()

# -------------------------------------------------------------------------
# CELL 1: CONFIGURATION (DÜZELTİLDİ)
# -------------------------------------------------------------------------
# BURADAKİ DEĞİŞİKLİK: Bucket ismi artık Project ID'ye göre otomatik oluşuyor.
# Böylece Two-Tower modelinin kaydedildiği yerle birebir eşleşiyor.
text_1 = """# @title ⚙️ Ranking Model Configuration
# @markdown Enter your project details below.

import os
from datetime import timedelta

# @markdown ### ☁️ Cloud Project Settings
# @markdown Enter your Project ID. Bucket name will be auto-generated to match the Setup Script.
PROJECT_ID = "your-project-id-here" # @param {type:"string"}
REGION = "us-central1" # @param {type:"string"}

# BUCKET NAME AUTOMATION (Fix for Path Issues)
# Setup scriptinde oluşturulan standart ismi kullanıyoruz
BUCKET_NAME = f"hm-workshop-{PROJECT_ID}"

# @markdown ### 🧪 Experiment Settings
TOP_K_RETRIEVAL = 60 # @param {type:"integer"}
NUM_TRAIN_WEEKS = 6 # @param {type:"slider", min:1, max:10}
LEARNING_RATE = 0.005 # @param {type:"number"}
NUM_LEAVES = 255 # @param {type:"integer"}
NUM_ROUNDS = 5000 # @param {type:"integer"}

# Environment & Paths
os.environ["GCLOUD_PROJECT"] = PROJECT_ID
BASE_PATH = f'gs://{BUCKET_NAME}'
ARTIFACTS_PATH = os.path.join(BASE_PATH, 'models/ranking_model') 

# Raw Data Paths
ARTICLES_PATH = os.path.join(BASE_PATH, 'articles.csv')
CUSTOMERS_PATH = os.path.join(BASE_PATH, 'customers.csv')
TRANSACTIONS_PATH = os.path.join(BASE_PATH, 'transactions.csv')

# Two-Tower Model Path (Otomatik Olarak Doğru Yeri Gösterecek)
RETRIEVAL_MODEL_PATH = os.path.join(BASE_PATH, 'models/two-tower-model')

print(f"✅ Configuration set for Project: {PROJECT_ID}")
print(f"📂 Target Bucket: {BASE_PATH}")
print(f"🔍 Retrieval Model Path: {RETRIEVAL_MODEL_PATH}")
print("⚠️ Please run the next cell to install dependencies.")
"""
cell_1 = nbf.v4.new_code_cell(text_1)
cell_1.metadata = {"cellView": "form", "id": "config_cell"}

# -------------------------------------------------------------------------
# CELL 2: INSTALLATION
# -------------------------------------------------------------------------
text_2 = """# @title 📥 Step 1: Install Dependencies & Fix Protobuf
!pip uninstall -y protobuf > /dev/null
!pip install protobuf==3.20.3 > /dev/null
!pip install -q tensorflow-recommenders lightgbm pandas numpy gcsfs
print("✅ Installation Complete.")
"""
cell_2 = nbf.v4.new_code_cell(text_2)
cell_2.metadata = {"cellView": "form", "id": "install_cell"}

# -------------------------------------------------------------------------
# CELL 3: IMPORTS
# -------------------------------------------------------------------------
text_3 = """# @title 📚 Step 2: Import Libraries
import gc
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import tensorflow_recommenders as tfrs

def apk(actual, predicted, k=10):
    if len(predicted) > k: predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual: return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

print(f"TensorFlow Version: {tf.__version__}")
print("✅ Libraries Imported.")
"""
cell_3 = nbf.v4.new_code_cell(text_3)
cell_3.metadata = {"cellView": "form", "id": "imports_cell"}

# -------------------------------------------------------------------------
# CELL 4: STATIC DATA
# -------------------------------------------------------------------------
text_4 = """# @title 💾 Step 3: Load Static Data
def load_static_data():
    print(">>> Loading Static Data...")
    cols = ['article_id', 'product_code', 'product_type_name', 'product_group_name',
            'graphical_appearance_no', 'colour_group_code', 'section_no', 'garment_group_no']
    articles = pd.read_csv(ARTICLES_PATH, dtype={'article_id': str}, usecols=cols)
    for c in cols:
        if c != 'article_id': articles[c] = pd.factorize(articles[c].astype(str), sort=True)[0]

    articles['article_id_int'], _ = pd.factorize(articles['article_id'], sort=True)
    article_map = dict(zip(articles['article_id'], articles['article_id_int']))
    articles['article_id'] = articles['article_id_int']
    del articles['article_id_int']

    cust_cols = ['customer_id', 'FN', 'Active', 'age', 'club_member_status']
    customers = pd.read_csv(CUSTOMERS_PATH, usecols=cust_cols, dtype={'customer_id': str})
    customers['FN'] = customers['FN'].fillna(0)
    customers['Active'] = customers['Active'].fillna(0)
    customers['age'] = customers['age'].fillna(customers['age'].mean())
    customers['club_member_status'] = pd.factorize(customers['club_member_status'].fillna('Unknown'), sort=True)[0]

    customers['customer_id_int'], _ = pd.factorize(customers['customer_id'], sort=True)
    customer_map = dict(zip(customers['customer_id'], customers['customer_id_int']))
    customers['customer_id'] = customers['customer_id_int']
    del customers['customer_id_int']

    return articles, customers, article_map, customer_map

articles_df, customers_df, article_map, customer_map = load_static_data()
print("✅ Static data loaded.")
"""
cell_4 = nbf.v4.new_code_cell(text_4)
cell_4.metadata = {"cellView": "form", "id": "load_static_cell"}

# -------------------------------------------------------------------------
# CELL 5: DATA ENGINE
# -------------------------------------------------------------------------
text_5 = """# @title 🏭 Step 4: Data Generation Engine
def generate_weekly_data(target_start_date, df_trans, tf_model, is_training=True):
    history_cutoff = target_start_date
    target_end_date = target_start_date + timedelta(days=7)
    df_history = df_trans[df_trans['t_dat'] < history_cutoff]

    if is_training:
        df_target = df_trans[(df_trans['t_dat'] >= target_start_date) & (df_trans['t_dat'] < target_end_date)]
        target_users = df_target['customer_id'].unique()
    else:
        df_target = df_trans[df_trans['t_dat'] >= target_start_date]
        target_users = df_target['customer_id'].unique()

    if len(target_users) == 0: return None

    last_week_start = history_cutoff - timedelta(days=7)
    df_last_week = df_history[df_history['t_dat'] > last_week_start]
    item_trend_score = df_last_week.groupby('article_id').size().reset_index(name='trend_score')

    hist_age = df_history[['article_id', 'customer_id']].merge(customers_df[['customer_id', 'age']], on='customer_id')
    item_avg_age = hist_age.groupby('article_id')['age'].mean().reset_index(name='item_avg_age')

    top_items = item_trend_score.sort_values('trend_score', ascending=False).head(12)['article_id'].tolist()
    
    repurchase_start = history_cutoff - timedelta(days=28)
    df_rep = df_history[(df_history['t_dat'] > repurchase_start) & (df_history['customer_id'].isin(target_users))]
    user_history = df_rep.groupby('customer_id')['article_id'].apply(lambda x: list(set(x))).to_dict()

    inv_cust_map = {v: k for k, v in customer_map.items()}
    tf_cands_dict = {}
    tf_scores_dict = {}
    
    BATCH = 1000
    tgt_list = list(target_users)

    for i in range(0, len(tgt_list), BATCH):
        batch_uids = tgt_list[i:i+BATCH]
        batch_strs = [inv_cust_map[u] for u in batch_uids]
        inp = {
            "customer_id": tf.constant(batch_strs),
            "age_bin": tf.constant(["25"]*len(batch_strs)),
            "month_of_year": tf.constant(["9"]*len(batch_strs)),
            "week_of_month": tf.constant(["2"]*len(batch_strs))
        }
        res = tf_model(inp)
        cands = res['candidates'].numpy().astype(str)
        scores = res['scores'].numpy()

        for idx, u in enumerate(batch_uids):
            c_list = []
            s_map = {}
            for j in range(min(TOP_K_RETRIEVAL, len(cands[idx]))):
                art_str = cands[idx][j]
                if art_str in article_map:
                    art_int = article_map[art_str]
                    c_list.append(art_int)
                    s_map[art_int] = float(scores[idx][j])
            tf_cands_dict[u] = c_list
            tf_scores_dict[u] = s_map

    data = []
    for u in target_users:
        candidates = set()
        candidates.update(top_items)
        if u in user_history: candidates.update(user_history[u])
        if u in tf_cands_dict: candidates.update(tf_cands_dict[u])

        for aid in candidates:
            t_score = tf_scores_dict.get(u, {}).get(aid, 0.0)
            data.append([u, aid, t_score])

    df_week = pd.DataFrame(data, columns=['customer_id', 'article_id', 'tf_score'])

    if is_training:
        df_target['purchased'] = 1
        truth = df_target[['customer_id', 'article_id', 'purchased']].drop_duplicates()
        df_week = df_week.merge(truth, on=['customer_id', 'article_id'], how='left')
        df_week['label'] = df_week['purchased'].fillna(0).astype('int8')
        del df_week['purchased']

        pos = df_week[df_week['label'] == 1]
        neg = df_week[df_week['label'] == 0]
        if len(neg) > 1_500_000:
            neg = neg.sample(n=1_500_000, random_state=42)
        df_week = pd.concat([pos, neg])

    df_week = df_week.merge(item_trend_score, on='article_id', how='left').fillna({'trend_score': 0})
    df_week = df_week.merge(customers_df[['customer_id', 'age']], on='customer_id', how='left')
    df_week = df_week.merge(item_avg_age, on='article_id', how='left')
    df_week['item_avg_age'] = df_week['item_avg_age'].fillna(30)
    df_week['age_diff'] = np.abs(df_week['age'] - df_week['item_avg_age'])
    df_week = df_week.merge(articles_df, on='article_id', how='left')
    cust_cols_static = [c for c in customers_df.columns if c not in ['age', 'customer_id']]
    df_week = df_week.merge(customers_df[['customer_id'] + cust_cols_static], on='customer_id', how='left')

    return df_week

print("✅ Data Generation Engine ready.")
"""
cell_5 = nbf.v4.new_code_cell(text_5)
cell_5.metadata = {"cellView": "form", "id": "data_engine_cell"}

# -------------------------------------------------------------------------
# CELL 6: TRAINING (İndirme Kısmı)
# -------------------------------------------------------------------------
text_6 = """# @title 🏋️ Step 5: Train LightGBM Model
print(">>> Loading Transactions...")
df_trans = pd.read_csv(TRANSACTIONS_PATH, dtype={'article_id': str, 'customer_id': str}, parse_dates=['t_dat'])
df_trans['article_id'] = df_trans['article_id'].map(article_map).fillna(-1).astype('int32')
df_trans['customer_id'] = df_trans['customer_id'].map(customer_map).fillna(-1).astype('int32')
df_trans = df_trans[(df_trans['article_id'] != -1) & (df_trans['customer_id'] != -1)]

print(">>> Loading Retrieval Model...")
# Cell 1'de otomatik oluşturulan RETRIEVAL_MODEL_PATH kullanılıyor
if not os.path.exists("two-tower-model"):
    print(f"Downloading from: {RETRIEVAL_MODEL_PATH}")
    os.system(f"gsutil -m cp -r {RETRIEVAL_MODEL_PATH} two-tower-model") 

if os.path.exists("two-tower-model/two-tower-model"):
    tf_model = tf.saved_model.load("two-tower-model/two-tower-model")
else:
    tf_model = tf.saved_model.load("two-tower-model")

print(f"Generating training data for {NUM_TRAIN_WEEKS} weeks...")
VAL_WEEK_START = pd.to_datetime('2020-09-16')

big_train_df = pd.DataFrame()
for w in range(1, NUM_TRAIN_WEEKS + 1):
    target_start = VAL_WEEK_START - timedelta(weeks=w)
    print(f"   Processing Week: {target_start.date()}")
    week_df = generate_weekly_data(target_start, df_trans, tf_model, is_training=True)
    if week_df is not None:
        big_train_df = pd.concat([big_train_df, week_df])
        del week_df
        gc.collect()

print("Preparing LightGBM Dataset...")
big_train_df = big_train_df.sort_values(by=['customer_id'], kind='mergesort')
drop_cols = ['customer_id', 'article_id', 'label']
X = big_train_df.drop(columns=drop_cols)
y = big_train_df['label']
group = big_train_df.groupby('customer_id', sort=False).size().to_numpy()

cat_cols = ['product_code', 'product_type_name', 'product_group_name',
            'graphical_appearance_no', 'colour_group_code', 'section_no',
            'garment_group_no', 'club_member_status']

train_set = lgb.Dataset(X, y, group=group, categorical_feature=cat_cols, free_raw_data=False)

params = {
    'objective': 'lambdarank',
    'metric': 'map',
    'eval_at': [12],
    'learning_rate': LEARNING_RATE,
    'num_leaves': NUM_LEAVES,
    'max_depth': -1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'force_col_wise': True,
    'verbose': -1
}

print(">>> Starting Training...")
model = lgb.train(params, train_set, num_boost_round=NUM_ROUNDS)
model.save_model("model.model")

print(">>> Evaluating...")
eval_df = generate_weekly_data(VAL_WEEK_START, df_trans, tf_model, is_training=False)
model_features = model.feature_name()
for c in model_features:
    if c not in eval_df.columns: eval_df[c] = 0

X_eval = eval_df[model_features]
eval_df['score'] = model.predict(X_eval)

top_recs = eval_df.sort_values(['customer_id', 'score'], ascending=[True, False]).groupby('customer_id').head(12)
preds_map = top_recs.groupby('customer_id')['article_id'].apply(list).to_dict()

val_target_df = df_trans[df_trans['t_dat'] >= VAL_WEEK_START]
ground_truth = val_target_df.groupby('customer_id')['article_id'].apply(list).to_dict()
val_users = list(ground_truth.keys())

map_scores = [apk(ground_truth[uid], preds_map[uid], k=12) if uid in preds_map else 0.0 for uid in val_users]
final_map = np.mean(map_scores)
print(f"🎉 FINAL MAP@12 SCORE: {final_map:.5f}")

print(f"Uploading model to {ARTIFACTS_PATH}...")
os.system(f"gsutil cp model.model {ARTIFACTS_PATH}/model.model")
print("✅ Training and Evaluation complete.")
"""
cell_6 = nbf.v4.new_code_cell(text_6)
cell_6.metadata = {"cellView": "form", "id": "train_eval_cell"}

# -------------------------------------------------------------------------
# CELL 7: PREPARE SERVING ARTIFACTS
# -------------------------------------------------------------------------
text_7 = """# @title 📦 Step 6: Prepare & Upload Serving Artifacts
def generate_image_url(article_id):
    article_id_str = str(article_id)
    if len(article_id_str) == 9:
        article_id_str = "0" + article_id_str
    folder = article_id_str[:2]
    return f"https://repo.hops.works/dev/jdowling/h-and-m/images/{folder}/{article_id_str}.jpg"

print("1. Processing Articles for Serving...")
df_articles = pd.read_csv(ARTICLES_PATH, dtype={'article_id': str})
df_articles['image_url'] = df_articles['article_id'].apply(generate_image_url)

if 'trend_score' not in df_articles.columns: df_articles['trend_score'] = 0.5 
if 'item_avg_age' not in df_articles.columns: df_articles['item_avg_age'] = 30.0

for col in df_articles.select_dtypes(include=['object']).columns:
    if col not in ['article_id', 'prod_name', 'image_url', 'detail_desc']:
        df_articles[col] = df_articles[col].astype('category')
        
df_articles.to_parquet('app_articles.parquet', index=False)
print("   -> app_articles.parquet created.")

print("2. Processing Customers for Serving...")
df_customers = pd.read_csv(CUSTOMERS_PATH, dtype={'customer_id': str})
df_customers['age'] = df_customers['age'].fillna(df_customers['age'].mean())
df_customers['club_member_status'] = df_customers['club_member_status'].fillna('Unknown')
df_customers.to_parquet('app_customers.parquet', index=False)
print("   -> app_customers.parquet created.")

print("3. Processing User History (Last 28 Days)...")
df_tr = pd.read_csv(TRANSACTIONS_PATH, usecols=['t_dat', 'customer_id', 'article_id'],
                 dtype={'article_id': str, 'customer_id': str},
                 parse_dates=['t_dat'])
VAL_START = pd.to_datetime('2020-09-16')
hist_start = VAL_START - timedelta(days=28)
df_hist = df_tr[(df_tr['t_dat'] >= hist_start) & (df_tr['t_dat'] < VAL_START)]
user_history = df_hist.groupby('customer_id')['article_id'].apply(lambda x: list(set(x))).reset_index()
user_history.columns = ['customer_id', 'article_ids']
user_history.to_parquet('app_user_history.parquet', index=False)
print("   -> app_user_history.parquet created.")

print("4. Processing Validation Truth (For Analysis)...")
val_df = df_tr[df_tr['t_dat'] >= VAL_START]
val_df[['customer_id', 'article_id']].to_parquet('val_truth.parquet', index=False)
print("   -> val_truth.parquet created.")

print("5. Generating Stats (Trending Items)...")
last_week_start = VAL_START - timedelta(days=7)
df_trend = df_tr[(df_tr['t_dat'] >= last_week_start) & (df_tr['t_dat'] < VAL_START)]
item_stats = df_trend.groupby('article_id').size().reset_index(name='trend_score')
item_stats['item_avg_age'] = 30.0 
item_stats.to_parquet('app_stats.parquet', index=False)
print("   -> app_stats.parquet created.")

print(f"Uploading all serving artifacts to {ARTIFACTS_PATH}...")
os.system(f"gsutil cp app_*.parquet {ARTIFACTS_PATH}/")
os.system(f"gsutil cp val_truth.parquet {ARTIFACTS_PATH}/")
print("✅ Serving artifacts are ready in GCS.")
"""
cell_7 = nbf.v4.new_code_cell(text_7)
cell_7.metadata = {"cellView": "form", "id": "prep_serving_cell"}

# -------------------------------------------------------------------------
# CELL 8: DEPLOYMENT (CLOUD RUN)
# -------------------------------------------------------------------------
text_8 = """# @title 🚀 Step 7: Deploy Hybrid System to Cloud Run
import os

os.makedirs("deploy_app", exist_ok=True)

# 2. Write FastAPI App (main.py)
app_code = f\"\"\"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import contextlib
import traceback
import gc

# --- CONFIGURATION (Dynamically injected) ---
BUCKET_NAME = '{BUCKET_NAME}' 
GCS_BASE = f'gs://{{BUCKET_NAME}}'
ARTIFACTS_PATH = f'{{GCS_BASE}}/models/ranking_model'
TF_PATH = f'{{GCS_BASE}}/models/two-tower-model'

PARQUET_FILES = [
    'app_articles.parquet',
    'app_customers.parquet',
    'app_stats.parquet', 
    'app_user_history.parquet'
]

models = {{}}
data = {{}}
TOP_K_TREND = 12 

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> [INIT] Starting up...")
    try:
        print(">>> [DOWNLOAD] Downloading artifacts...")
        for p_file in PARQUET_FILES:
            src = f"{{ARTIFACTS_PATH}}/{{p_file}}"
            os.system(f"gsutil cp {{src}} .")

        if not os.path.exists("model.model"):
            os.system(f"gsutil cp {{ARTIFACTS_PATH}}/model.model model.model")

        if not os.path.exists("two-tower-model"):
            os.system(f"gsutil -m cp -r {{TF_PATH}} two-tower-model")

        print(">>> [LOAD] Loading Parquet files...")
        if os.path.exists('app_articles.parquet'):
            data['articles'] = pd.read_parquet('app_articles.parquet')
            data['customers'] = pd.read_parquet('app_customers.parquet')
            
            if os.path.exists('app_user_history.parquet'):
                h_df = pd.read_parquet('app_user_history.parquet')
                data['user_history_map'] = dict(zip(h_df['customer_id'], h_df['article_ids']))
            else:
                data['user_history_map'] = {{}}

            if os.path.exists('app_stats.parquet'):
                stats = pd.read_parquet('app_stats.parquet')
                data['stats'] = stats
                data['top_trend_items'] = stats.sort_values('trend_score', ascending=False).head(TOP_K_TREND)['article_id'].tolist()
            else:
                data['stats'] = pd.DataFrame()
                data['top_trend_items'] = []

            print(">>> [MAP] Creating ID mappings...")
            data['articles']['article_id_idx'], _ = pd.factorize(data['articles']['article_id'], sort=True)
            data['article_map'] = dict(zip(data['articles']['article_id'], data['articles']['article_id_idx']))
            
            if not data['stats'].empty:
                data['stats']['article_id_idx'] = data['stats']['article_id'].map(data['article_map'])
                data['stats'] = data['stats'].dropna().rename(columns={{'trend_score': 'stat_trend', 'item_avg_age': 'stat_age'}})
                data['stats']['article_id_idx'] = data['stats']['article_id_idx'].astype(int)

            print(">>> [DATA] Ready.")
        else:
            print("!!! [CRITICAL] Parquet files missing.")

        print(">>> [MODELS] Loading models...")
        if os.path.exists("model.model"):
            models['lgb'] = lgb.Booster(model_file="model.model")
        
        if os.path.exists("two-tower-model"):
            models['tf'] = tf.saved_model.load("two-tower-model")
            
        print(">>> [READY] Service is ready.")

    except Exception as e:
        print(f"!!! [INIT ERROR] {{e}}")
        traceback.print_exc()

    yield
    models.clear()
    data.clear()
    gc.collect()

app = FastAPI(lifespan=lifespan)

class RecRequest(BaseModel):
    user_id: str
    history: list[str] = []

@app.post("/predict")
async def predict(req: RecRequest):
    try:
        user_id = req.user_id
        candidates = set()
        
        # 1. CANDIDATES
        if 'top_trend_items' in data: candidates.update(data['top_trend_items'])
        
        try:
            if 'tf' in models:
                inp = {{
                    "customer_id": tf.constant([user_id]),
                    "age_bin": tf.constant(["25"]), 
                    "month_of_year": tf.constant(["9"]), 
                    "week_of_month": tf.constant(["2"])
                }}
                tf_res = models['tf'](inp)['candidates'].numpy()[0].astype(str)
                candidates.update(tf_res)
        except: pass

        if 'user_history_map' in data and user_id in data['user_history_map']:
            past_items = data['user_history_map'][user_id]
            if isinstance(past_items, (np.ndarray, list)): candidates.update(past_items)
            else: candidates.add(str(past_items))
            
        if req.history: candidates.update(req.history)

        if not candidates: return {{"recommendations": []}}

        # 2. DATAFRAME CONSTRUCTION
        valid_cands, valid_idxs = [], []
        article_map = data.get('article_map', {{}})
        
        for c in candidates:
            c_str = str(c).strip().zfill(10)
            if c_str in article_map:
                valid_cands.append(c_str)
                valid_idxs.append(article_map[c_str])

        if not valid_cands: return {{"recommendations": []}}

        cand_df = pd.DataFrame({{'article_id': valid_cands, 'article_id_idx': valid_idxs}})
        
        cand_df = cand_df.merge(data['articles'], on='article_id', how='left')
        if 'stats' in data and not data['stats'].empty:
            cand_df = cand_df.merge(data['stats'], on='article_id_idx', how='left')
        
        cand_df['trend_score'] = cand_df.get('stat_trend', 0.0).fillna(0)
        cand_df['item_avg_age'] = cand_df.get('stat_age', 30.0).fillna(30.0)
        
        cust_df = data['customers']
        if user_id in cust_df['customer_id'].values:
            u_row = cust_df[cust_df['customer_id'] == user_id].iloc[0]
            cand_df['age'] = u_row['age']
            cand_df['club_member_status'] = u_row['club_member_status']
        else:
            cand_df['age'] = 30.0
            cand_df['club_member_status'] = 0

        cand_df['age_diff'] = np.abs(cand_df['age'] - cand_df['item_avg_age'])

        # 3. PREDICT
        if 'lgb' in models:
            feats = models['lgb'].feature_name()
            for f in feats:
                if f not in cand_df.columns: cand_df[f] = 0
                if cand_df[f].dtype == 'object' or str(cand_df[f].dtype) == 'category':
                    cand_df[f] = pd.factorize(cand_df[f])[0]
            cand_df['score'] = models['lgb'].predict(cand_df[feats])
        else:
            cand_df['score'] = cand_df['trend_score']

        top_recs = cand_df.sort_values('score', ascending=False).head(12)
        results = []
        for _, row in top_recs.iterrows():
            results.append({{
                "article_id": row['article_id'],
                "prod_name": str(row.get('prod_name', 'Unknown')),
                "image_url": str(row.get('image_url', '')),
                "score": float(row['score'])
            }})
            
        return {{"recommendations": results}}

    except Exception as e:
        traceback.print_exc()
        return {{ "error": str(e) }}, 500
\"\"\"

with open("deploy_app/main.py", "w") as f:
    f.write(app_code)

# 3. Write Dockerfile
dockerfile_code = \"\"\"
FROM python:3.9-slim
RUN apt-get update && apt-get install -y curl gnupg libgomp1
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \\
    apt-get update -y && apt-get install google-cloud-sdk -y
WORKDIR /app
RUN pip install flask gunicorn tensorflow tensorflow-recommenders scann lightgbm pandas pyarrow fastapi uvicorn gcsfs
COPY main.py .
CMD exec uvicorn main:app --host 0.0.0.0 --port 8080
\"\"\"

with open("deploy_app/Dockerfile", "w") as f:
    f.write(dockerfile_code)

print("✅ Deployment files generated.")

# 4. Build and Deploy
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/hm-recommender-app"
SERVICE_NAME = "hm-recommender-service"

print(f"🔨 Building Container: {IMAGE_NAME}")
!gcloud builds submit --tag $IMAGE_NAME deploy_app

print(f"🚀 Deploying to Cloud Run: {SERVICE_NAME}")
!gcloud run deploy $SERVICE_NAME \\
  --image $IMAGE_NAME \\
  --platform managed \\
  --region $REGION \\
  --allow-unauthenticated \\
  --memory 4Gi

print("✅ Deployment Complete! Check the URL.")
"""
cell_8 = nbf.v4.new_code_cell(text_8)
cell_8.metadata = {"cellView": "form", "id": "deploy_cell"}

# Add all cells
nb.cells.extend([cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7, cell_8])

with open('hm_ranking_lightgbm_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("🎉 'hm_ranking_lightgbm_training.ipynb' updated with dynamic paths!")