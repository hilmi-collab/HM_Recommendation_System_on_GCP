import json
import os

def create_notebook():
    # Notebook Metadata
    notebook = {
        "cells": [],
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_cell(source_code, cell_type="code"):
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_code.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None
        }
        if cell_type == "markdown":
            del cell["outputs"]
            del cell["execution_count"]
        notebook["cells"].append(cell)

    # --- CELL 1: Configuration ---
    cell_1 = """# -*- coding: utf-8 -*-
# @title ⚙️ Ranking Model Configuration
import os
from datetime import timedelta

# @markdown ### ☁️ Project Settings
PROJECT_ID = "ace-team-hilmi-service" # @param {type:"string"}
REGION = "us-central1" # @param {type:"string"}

# 1. PUBLIC DATA BUCKET (Read-Only)
DATA_BUCKET_NAME = "hm-recommendation-workshop"
DATA_GCS_PATH = f"gs://{DATA_BUCKET_NAME}"

# 2. PRIVATE WORK BUCKET (Write)
WORK_BUCKET_NAME = f"hm-workshop-{PROJECT_ID}"
WORK_GCS_PATH = f"gs://{WORK_BUCKET_NAME}"

# ARTIFACTS PATH
ARTIFACTS_PATH = os.path.join(WORK_GCS_PATH, 'models/ranking_model')

# INPUT DATA PATHS
ARTICLES_PATH = os.path.join(DATA_GCS_PATH, 'articles.csv')
CUSTOMERS_PATH = os.path.join(DATA_GCS_PATH, 'customers.csv')
TRANSACTIONS_PATH = os.path.join(DATA_GCS_PATH, 'transactions.csv')

# TWO-TOWER MODEL PATH
RETRIEVAL_MODEL_PATH = os.path.join(WORK_GCS_PATH, 'models/two-tower-model')

# @markdown ### 🧪 Experiment Settings
TOP_K_RETRIEVAL = 60 # @param {type:"integer"}
NUM_TRAIN_WEEKS = 6 # @param {type:"slider", min:1, max:6}
LEARNING_RATE = 0.05 # @param {type:"number"}
NUM_LEAVES = 63 # @param {type:"integer"}
NUM_ROUNDS = 1000 # @param {type:"integer"}

os.environ["GCLOUD_PROJECT"] = PROJECT_ID

print(f"✅ Config Set:")
print(f"   📥 Raw Data: {DATA_GCS_PATH}")
print(f"   🔍 Retrieval Model: {RETRIEVAL_MODEL_PATH}")
print(f"   💾 Ranking Model Output: {ARTIFACTS_PATH}")
print("⚠️ Please run the next cell to install dependencies.")"""
    add_cell(cell_1)

    # --- CELL 2: Install Dependencies ---
    cell_2 = """# @title 📥 Step 1: Install Dependencies
# @markdown Installing libraries using the standard method (similar to Retrieval notebook).

# 1. Install TensorFlow Recommenders without dependencies first
!pip install -q tensorflow-recommenders --no-deps

# 2. Install ScaNN (TensorFlow compatible) and other libs
!pip install -q "scann[tf]" tensorflow-recommenders lightgbm pandas numpy gcsfs

print("✅ Installation Complete.")
print("⚠️ Please RESTART RUNTIME (Runtime > Restart Runtime) before running the next cell.")"""
    add_cell(cell_2)

    # --- CELL 3: Import Libraries ---
    cell_3 = """# @title 📚 Step 2: Import Libraries
import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import subprocess

# Import order matters for registering ops
import tensorflow as tf
import tensorflow_recommenders as tfrs

# --- IMPORTANT: SCANN IMPORT ---
# Import ScaNN after TF to register C++ operations.
import scann
# -------------------------------

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
print("✅ Libraries Imported.")"""
    add_cell(cell_3)

    # --- CELL 4: Load Static Data ---
    cell_4 = """# @title 💾 Step 3: Load Static Data
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
print("✅ Static data loaded.")"""
    add_cell(cell_4)

    # --- CELL 5: Data Generation Engine ---
    cell_5 = """# @markdown Fixed: Accessing model output as tuple (scores, candidates) instead of dict.

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

        # --- FIX HERE ---
        # When SavedModel is loaded, ScaNN output is usually a tuple (scores, candidates).
        # res[0] -> Scores
        # res[1] -> Candidates (IDs)
        if isinstance(res, dict):
            cands = res['candidates'].numpy().astype(str)
            scores = res['scores'].numpy()
        else:
            # Tuple behavior (SavedModel)
            scores = res[0].numpy()
            cands = res[1].numpy().astype(str)
        # ------------------------

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

print("✅ Data Generation Engine ready.")"""
    add_cell(cell_5)

    # --- CELL 6: Train Model ---
    cell_6 = """# @title 🏋️ Step 5: Train LightGBM Model
print(">>> Loading Transactions...")
df_trans = pd.read_csv(TRANSACTIONS_PATH, dtype={'article_id': str, 'customer_id': str}, parse_dates=['t_dat'])
df_trans['article_id'] = df_trans['article_id'].map(article_map).fillna(-1).astype('int32')
df_trans['customer_id'] = df_trans['customer_id'].map(customer_map).fillna(-1).astype('int32')
df_trans = df_trans[(df_trans['article_id'] != -1) & (df_trans['customer_id'] != -1)]

print(">>> Loading Retrieval Model...")

# 1. Helper Function: Find saved_model.pb recursively
def find_model_path(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "saved_model.pb" in files:
            return root
    return None

# 2. Download Model
if os.path.exists("two-tower-model"):
    os.system("rm -rf two-tower-model")
os.makedirs("two-tower-model", exist_ok=True)

# Debug Check
if os.system(f"gsutil -q stat {RETRIEVAL_MODEL_PATH}/saved_model.pb") != 0:
    print(f"❌ ERROR: File not found in GCS: {RETRIEVAL_MODEL_PATH}/saved_model.pb")
    raise FileNotFoundError("Retrieval model failed to download.")

print(f"   Downloading from: {RETRIEVAL_MODEL_PATH}")
# Wildcard download
os.system(f"gsutil -m cp -r '{RETRIEVAL_MODEL_PATH}/*' two-tower-model/")

# 3. Locate & Load Model
model_dir = find_model_path("two-tower-model")

if model_dir:
    print(f"✅ Found SavedModel at: {model_dir}")
    import scann
    tf_model = tf.saved_model.load(model_dir)
else:
    print("❌ saved_model.pb not found after download.")
    raise FileNotFoundError("saved_model.pb not found.")

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
    'verbose': 100
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
print("✅ Training and Evaluation complete.")"""
    add_cell(cell_6)

    # --- CELL 7: Demo Report ---
    cell_7 = """# @markdown This cell finds and reports the most successful predictions (hits) of the trained model on the validation set.

print(">>> Loading Product Names for Human-Readable Report...")
# Extracting names from the original file
raw_articles = pd.read_csv(ARTICLES_PATH, usecols=['article_id', 'prod_name', 'product_type_name'], dtype={'article_id': str})
# We need to convert IDs to int because model output is int
raw_articles['article_id_int'] = raw_articles['article_id'].map(article_map).fillna(-1).astype(int)
raw_articles = raw_articles[raw_articles['article_id_int'] != -1]

# Map: Int ID -> "Product Name (Type)"
name_map = dict(zip(raw_articles['article_id_int'], raw_articles['prod_name'] + ' (' + raw_articles['product_type_name'] + ')'))
# Map: Int ID -> String ID
inv_article_map = dict(zip(raw_articles['article_id_int'], raw_articles['article_id']))

# Customer Map Inverse
inv_cust_map = {v: k for k, v in customer_map.items()}

print(">>> Analyzing Hits...")
success_list = []

# preds_map and ground_truth are available in memory from the previous cell!
for uid, items in preds_map.items():
    if uid in ground_truth:
        actual = set(ground_truth[uid])
        hits = list(set(items) & actual)

        if len(hits) > 0:
            # Convert to human readable format
            u_str = inv_cust_map.get(uid, "Unknown")

            # Hit details
            hit_details = []
            hit_ids = []
            for h in hits:
                h_str = inv_article_map.get(h, str(h))
                h_name = name_map.get(h, "Unknown Product")
                hit_details.append(f"{h_name}")
                hit_ids.append(h_str)

            success_list.append({
                'customer_id': u_str,
                'hit_count': len(hits),
                'hit_names': ", ".join(hit_details),
                'hit_ids': ", ".join(hit_ids)
            })

df_success = pd.DataFrame(success_list).sort_values('hit_count', ascending=False)

print("\\n" + "="*60)
print(f"DEMO REPORT: At least 1 hit for a total of {len(df_success)} users!")
print("="*60)

if not df_success.empty:
    print("\\n🏆 TOP 10 PREDICTION EXAMPLES:")
    for i, row in df_success.head(10).iterrows():
        print(f"👤 Customer: {row['customer_id']}")
        print(f"🏆 Hit Count: {row['hit_count']}")
        print(f"✅ Known Products: {row['hit_names']} (ID: {row['hit_ids']})")
        print("-" * 40)

    # Save CSV
    df_success.to_csv('demo_success_results.csv', index=False)
    print(f"📄 Detailed report saved: demo_success_results.csv")
else:
    print("😔 Unfortunately, there are no hits in this validation set.")"""
    add_cell(cell_7)

    # --- CELL 8: Prepare Artifacts ---
    cell_8 = """# @title 📦 Step 6: Prepare & Upload Serving Artifacts (FULL & FIXED)
import pickle
import gc
import pandas as pd
import numpy as np
import os

# We fix everything here to prevent ID Mismatch.
print(">>> 1. Saving Mapping Objects (Dictionary)...")

# article_map: "0108775015" (str) -> 5 (int)
with open('article_map.pkl', 'wb') as f:
    pickle.dump(article_map, f)

# customer_map: "000058a12d..." (str) -> 2 (int)
with open('customer_map.pkl', 'wb') as f:
    pickle.dump(customer_map, f)

# Inverse Mapping (Int -> Str) - Will be needed shortly
inv_article_map = {v: k for k, v in article_map.items()}
inv_customer_map = {v: k for k, v in customer_map.items()}

print(">>> 2. Processing Articles for Serving (Feature Store)...")
# articles_df is already in memory with Integer IDs (from Step 3).
# Let's add the original String ID as a column just in case.
serving_articles = articles_df.copy()
serving_articles['article_id_str'] = serving_articles['article_id'].map(inv_article_map)

# Optimize feature data types (Category)
for col in serving_articles.columns:
    if serving_articles[col].dtype == 'object' and col != 'article_id_str':
        serving_articles[col] = serving_articles[col].astype('category')

serving_articles.to_parquet('app_articles_features.parquet', index=False)
print("   -> app_articles_features.parquet saved.")

print(">>> 3. Processing Customers for Serving...")
# customers_df is also in memory with Integer IDs.
serving_customers = customers_df.copy()
serving_customers['customer_id_str'] = serving_customers['customer_id'].map(inv_customer_map)
serving_customers.to_parquet('app_customers_features.parquet', index=False)
print("   -> app_customers_features.parquet saved.")

print(">>> 4. Processing User History (Integer IDs)...")
# df_trans was already converted to Integer IDs in Step 5.
# It is best to store history as Integers since the model expects Integer IDs.
VAL_START = pd.to_datetime('2020-09-16')
hist_start = VAL_START - timedelta(days=28)

# Data for the last 28 days
df_hist = df_trans[(df_trans['t_dat'] >= hist_start) & (df_trans['t_dat'] < VAL_START)]

# List of items purchased for each user (Integer List)
user_history = df_hist.groupby('customer_id')['article_id'].apply(list).reset_index()
user_history.columns = ['customer_id', 'article_ids'] # customer_id: INT, article_ids: LIST[INT]

user_history.to_parquet('app_user_history_int.parquet', index=False)
print("   -> app_user_history_int.parquet saved (Integer IDs).")

print(">>> 5. Generating Stats / Trending Items (Integer IDs)...")
# Most popular items of the last week (for Cold Start and Candidate recommendation)
last_week_start = VAL_START - timedelta(days=7)
df_trend = df_trans[(df_trans['t_dat'] >= last_week_start) & (df_trans['t_dat'] < VAL_START)]

# Count and save
item_stats = df_trend.groupby('article_id').size().reset_index(name='trend_score')
item_stats['item_avg_age'] = 30.0 # Simplified average age (calculate real if needed)

# article_id in item_stats is already Integer.
item_stats.to_parquet('app_stats_int.parquet', index=False)
print("   -> app_stats_int.parquet saved (Integer IDs).")

print(">>> 6. Processing Validation Truth (STRING IDs for Frontend)...")
# We need String IDs to show "What did they actually buy?" on the Streamlit side.
# Because we don't want to load pickle maps on the frontend to keep file size small.
val_df = df_trans[df_trans['t_dat'] >= VAL_START].copy()

# Converting Integer IDs back to String
val_df['article_id'] = val_df['article_id'].map(inv_article_map)
val_df['customer_id'] = val_df['customer_id'].map(inv_customer_map)

# Clean if there are NaNs (Unmapped)
val_df = val_df.dropna(subset=['article_id', 'customer_id'])

# Save only necessary columns
val_df[['customer_id', 'article_id']].to_parquet('val_truth.parquet', index=False)
print("   -> val_truth.parquet saved (String IDs).")

print(f"\\n🚀 Uploading artifacts to {ARTIFACTS_PATH}...")
upload_files = [
    'article_map.pkl',
    'customer_map.pkl',
    'app_articles_features.parquet',
    'app_customers_features.parquet',
    'app_user_history_int.parquet',
    'app_stats_int.parquet',
    'val_truth.parquet',
    'model.model' # Let's secure the LightGBM model as well
]

for f in upload_files:
    if os.path.exists(f):
        os.system(f"gsutil cp {f} {ARTIFACTS_PATH}/")
        print(f"   -> Uploaded: {f}")
    else:
        print(f"   ⚠️ Warning: File not found {f}")

print("\\n✅ All Serving Artifacts are Ready & Consistent!")"""
    add_cell(cell_8)

    # --- CELL 9: Backend App ---
    # Not: This is a string within a string, handled carefully.
    cell_9 = """# @title 📦 Step 7: Create Backend Application (FIXED)

import os

os.makedirs("deploy_app", exist_ok=True)

# Writing main.py with updated file names and import fixes
app_code = f\"\"\"
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import pickle
import gc
import contextlib
import traceback

# --- CONFIG ---
BUCKET_NAME = '{WORK_BUCKET_NAME}'
GCS_BASE = f'gs://{{BUCKET_NAME}}'
ARTIFACTS_PATH = f'{{GCS_BASE}}/models/ranking_model'
TF_PATH = f'{{GCS_BASE}}/models/two-tower-model'

models = {{}}
data = {{}}
TOP_K_TREND = 12

def generate_image_url(article_id_str):
    # Ensure ID is string and 10 digits (for the leading zero)
    aid_str = str(article_id_str).strip().zfill(10)
    folder = aid_str[:3]
    return f"https://repo.hops.works/dev/jdowling/h-and-m/images/{{folder}}/{{aid_str}}.jpg"

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> [INIT] Loading Artifacts...")
    try:
        # 1. Download Files
        files = [
            'article_map.pkl',
            'customer_map.pkl',
            'app_articles_features.parquet',
            'app_customers_features.parquet',
            'app_stats_int.parquet',
            'app_user_history_int.parquet',
            'model.model'
        ]

        for f in files:
            if not os.path.exists(f):
                print(f"Downloading {{f}}...")
                os.system(f"gsutil cp {{ARTIFACTS_PATH}}/{{f}} .")

        if not os.path.exists("two-tower-model"):
            os.makedirs("two-tower-model", exist_ok=True)
            os.system(f"gsutil -m cp -r {{TF_PATH}}/* two-tower-model/")

        # 2. Load MAPs (Critical Step)
        print(">>> Loading Maps...")
        with open('article_map.pkl', 'rb') as f:
            article_map = pickle.load(f) # String -> Int

        with open('customer_map.pkl', 'rb') as f:
            customer_map = pickle.load(f) # String -> Int

        # Inverse Map (Int -> String)
        inv_article_map = {{v: k for k, v in article_map.items()}}

        # 3. Load Feature Tables (Already with Integer IDs)
        print(">>> Loading Features...")
        articles_feat = pd.read_parquet('app_articles_features.parquet')
        articles_feat = articles_feat.set_index('article_id')

        customers_feat = pd.read_parquet('app_customers_features.parquet')
        customers_feat = customers_feat.set_index('customer_id')

        stats = pd.read_parquet('app_stats_int.parquet')
        top_trend_ids = stats.sort_values('trend_score', ascending=False).head(TOP_K_TREND)['article_id'].tolist()

        # User History (List of Integers)
        user_history_map = {{}}
        if os.path.exists('app_user_history_int.parquet'):
            hist_df = pd.read_parquet('app_user_history_int.parquet')
            # customer_id (int) -> article_ids (list of int)
            user_history_map = dict(zip(hist_df['customer_id'], hist_df['article_ids']))

        # Register to Global Data
        data['article_map'] = article_map
        data['inv_article_map'] = inv_article_map
        data['customer_map'] = customer_map
        data['articles_feat'] = articles_feat
        data['customers_feat'] = customers_feat
        data['stats'] = stats.set_index('article_id')
        data['top_trend_ids'] = top_trend_ids
        data['user_history_map'] = user_history_map

        # 4. Load Models
        print(">>> Loading Models...")
        models['lgb'] = lgb.Booster(model_file='model.model')

        try:
            # TF ScaNN import sometimes requires a trick
            import scann
            models['tf'] = tf.saved_model.load("two-tower-model")
            print("   -> TF/ScaNN Loaded.")
        except Exception as e:
            print(f"   -> TF Load Warning: {{e}}")

        print(">>> [READY] Service loaded correctly.")

    except Exception as e:
        print(f"!!! Server Start Error: {{e}}")
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
        uid_str = str(req.user_id)

        # 1. Customer ID Conversion (String -> Int)
        u_idx = data['customer_map'].get(uid_str, -1)

        # 2. Candidate Generation - ALL INTEGERS
        candidate_ids_int = set()

        # A) Trends
        candidate_ids_int.update(data['top_trend_ids'])

        # B) Past Purchases (User History)
        if u_idx != -1 and u_idx in data['user_history_map']:
            # Take last 12 to prevent candidate explosion
            past_items = data['user_history_map'][u_idx][-12:]
            candidate_ids_int.update(past_items)

        # C) Two-Tower (TF ScaNN)
        if 'tf' in models:
            try:
                inp = {{
                    "customer_id": tf.constant([uid_str]),
                    "age_bin": tf.constant(["25"]),
                    "month_of_year": tf.constant(["9"]),
                    "week_of_month": tf.constant(["2"])
                }}
                res = models['tf'](inp)

                # ScaNN outputs are String IDs.
                if isinstance(res, dict): cands_str = res['candidates'].numpy()[0].astype(str)
                else: cands_str = res[1].numpy()[0].astype(str)

                # Convert String candidates to Integer
                for c in cands_str:
                    if c in data['article_map']:
                        candidate_ids_int.add(data['article_map'][c])
            except: pass

        # D) Frontend Cart (History)
        for h in req.history:
            if h in data['article_map']:
                candidate_ids_int.add(data['article_map'][h])

        if not candidate_ids_int: return {{"recommendations": []}}

        # 3. Create Feature Table
        cand_list = list(candidate_ids_int)
        df_cand = pd.DataFrame({{'article_id': cand_list}}) # article_id = INT

        # Join Article Features
        df_cand = df_cand.join(data['articles_feat'], on='article_id', rsuffix='_feat')

        # Add Customer Features
        if u_idx != -1 and u_idx in data['customers_feat'].index:
            cust_row = data['customers_feat'].loc[u_idx]
            for col in data['customers_feat'].columns:
                if col != 'customer_id_str':
                    df_cand[col] = cust_row[col]
        else:
            # Default values (Cold User)
            df_cand['age'] = 30.0
            df_cand['club_member_status'] = 0

        # Add Stats (Trend Score)
        if not data['stats'].empty:
            df_cand = df_cand.join(data['stats'][['trend_score', 'item_avg_age']], on='article_id', rsuffix='_stat')

        df_cand['trend_score'] = df_cand.get('trend_score', 0).fillna(0)
        item_age = df_cand.get('item_avg_age', 30.0).fillna(30.0)

        # Age Diff
        df_cand['age_diff'] = np.abs(df_cand['age'] - item_age)

        # 4. LightGBM Prediction
        if 'lgb' in models:
            feats = models['lgb'].feature_name()
            # Ensure feature order
            for f in feats:
                if f not in df_cand.columns: df_cand[f] = 0

            df_cand['score'] = models['lgb'].predict(df_cand[feats])
        else:
            df_cand['score'] = df_cand['trend_score'] # Fallback

        # 5. Ranking and Result
        top_recs = df_cand.sort_values('score', ascending=False).head(12)

        results = []
        for idx, row in top_recs.iterrows():
            aid_int = int(row['article_id'])
            # Int -> String ID
            aid_str = data['inv_article_map'].get(aid_int, "Unknown")

            results.append({{
                "article_id": aid_str,
                "score": float(row['score']),
                "image_url": generate_image_url(aid_str),
                "prod_name": str(row.get('prod_name', 'Product'))
            }})

        return {{"recommendations": results}}

    except Exception as e:
        print(f"!!! Predict Error: {{e}}")
        traceback.print_exc()
        return {{"error": str(e), "recommendations": []}}
\"\"\"

with open("deploy_app/main.py", "w") as f:
    f.write(app_code)

print("✅ Backend code (main.py) updated with 'contextlib' fix.")"""
    add_cell(cell_9)

    # --- CELL 10: Dockerfile ---
    cell_10 = """# 3. Write Dockerfile
dockerfile_code = \"\"\"
FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim

RUN apt-get update && apt-get install -y python3-pip python3-dev libgomp1 && \\\\
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# TF 2.17 compatible
RUN pip3 install --no-cache-dir --break-system-packages \\\\
    flask gunicorn fastapi uvicorn \\\\
    "scann[tf]" \\\\
    tensorflow-recommenders \\\\
    lightgbm pandas pyarrow gcsfs

COPY main.py .

CMD exec uvicorn main:app --host 0.0.0.0 --port 8080
\"\"\"

with open("deploy_app/Dockerfile", "w") as f:
    f.write(dockerfile_code)

print("✅ Deployment files generated.")"""
    add_cell(cell_10)

    # --- CELL 11: Build & Deploy ---
    cell_11 = """# 4. Build and Deploy
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/hm-recommender-app"
SERVICE_NAME = "hm-recommender-service"

# Build
print(f"🔨 Building Container: {IMAGE_NAME}")
!gcloud builds submit --tag $IMAGE_NAME deploy_app

print(f"🚀 Deploying to Cloud Run: {SERVICE_NAME}")

# TIMEOUT: 3600
# MEMORY: 16Gi
# CPU: 4
!gcloud run deploy $SERVICE_NAME --image $IMAGE_NAME --platform managed --region $REGION --allow-unauthenticated --memory 16Gi --cpu 4 --timeout 3600 --cpu-boost

print("✅ Deployment Complete! Check the URL.")"""
    add_cell(cell_11)

    # Save to file
    with open('hm_ranking_lightgbm_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print("Successfully created 'hm_ranking_lightgbm_training.ipynb'.")

if __name__ == "__main__":
    create_notebook()