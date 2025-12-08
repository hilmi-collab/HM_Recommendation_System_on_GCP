import nbformat as nbf

# Initialize a new notebook
nb = nbf.v4.new_notebook()

# -------------------------------------------------------------------------
# CELL 1: CONFIGURATION (FORM)
# -------------------------------------------------------------------------
text_1 = """# @title ⚙️ Workshop Configuration & Setup
# @markdown Please enter your project details and training parameters below.
# @markdown ---

import os

# @markdown ### ☁️ Cloud Project Settings
PROJECT_ID = "your-project-id-here" # @param {type:"string"}
BUCKET_NAME = "hm-recommendation-workshop" # @param {type:"string"}
REGION = "us-central1" # @param {type:"string"}

# @markdown ### 🚀 Model Hyperparameters
EMBEDDING_DIM = 64 # @param {type:"integer"}
LEARNING_RATE = 0.1 # @param {type:"number"}
EPOCHS = 5 # @param {type:"slider", min:1, max:10, step:1}

# @markdown ### 📦 Data Paths (Relative to Bucket)
# @markdown Do not change these unless your bucket structure is different.
ARTICLES_FILE = "articles.csv" # @param {type:"string"}
CUSTOMERS_FILE = "customers.csv" # @param {type:"string"}
TRANSACTIONS_FILE = "transactions.csv" # @param {type:"string"}

# Setup Environment Variables
os.environ["GCLOUD_PROJECT"] = PROJECT_ID
# Important: Use legacy Keras behavior for TF 2.x compatibility with ScaNN
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

GCS_BASE_PATH = f"gs://{BUCKET_NAME}"

print(f"✅ Configuration set for Project: {PROJECT_ID}")
print(f"📂 Data Source: {GCS_BASE_PATH}")
print(f"weights will be saved to: {GCS_BASE_PATH}/models/two-tower-model")
"""
cell_1 = nbf.v4.new_code_cell(text_1)
# Adding metadata to hide the cell code and show the form
cell_1.metadata = {"cellView": "form", "id": "config_cell"}

# -------------------------------------------------------------------------
# CELL 2: INSTALLATION
# -------------------------------------------------------------------------
text_2 = """# @title 📥 Step 1: Install Libraries
# @markdown Installing TensorFlow Recommenders, ScaNN, and Datasets.
# @markdown This may take 1-2 minutes.

import sys

# Install TF-compatible version of ScaNN and other dependencies
!pip install -q tensorflow-recommenders --no-deps
!pip install -q --upgrade tensorflow-datasets
!pip install -q "scann[tf]" tensorflow-recommenders tensorflow-datasets

print("✅ Installation Complete.")
"""
cell_2 = nbf.v4.new_code_cell(text_2)
cell_2.metadata = {"cellView": "form", "id": "install_cell"}

# -------------------------------------------------------------------------
# CELL 3: IMPORTS & DATA LOADING
# -------------------------------------------------------------------------
text_3 = """# @title 💾 Step 2: Load Data from Cloud Storage
# @markdown Reading CSV files directly from your GCS Bucket using the paths defined in configuration.

import os
import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text, List
from tensorflow.keras.layers import StringLookup, Embedding, Dense

# GPU Check
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🚀 GPU Active: {gpus[0].name}")
else:
    print("⚠️ Running on CPU (Training might be slower)")

# Paths
ARTICLES_PATH = os.path.join(GCS_BASE_PATH, ARTICLES_FILE)
CUSTOMERS_PATH = os.path.join(GCS_BASE_PATH, CUSTOMERS_FILE)
TRANSACTIONS_PATH = os.path.join(GCS_BASE_PATH, TRANSACTIONS_FILE)

# --- 1. Load Articles ---
print(f"Loading Articles from: {ARTICLES_PATH}")
ARTICLE_FEATURES = ['article_id', 'product_type_name', 'product_group_name', 'colour_group_name', 'department_name']
articles_df = pd.read_csv(ARTICLES_PATH, usecols=ARTICLE_FEATURES, dtype={'article_id': str})
for col in ARTICLE_FEATURES:
    if col != 'article_id':
        articles_df[col] = articles_df[col].fillna('Unknown')

# --- 2. Load Customers ---
print(f"Loading Customers from: {CUSTOMERS_PATH}")
customers_df = pd.read_csv(CUSTOMERS_PATH, usecols=['customer_id', 'age'])
age_bins = [0, 19, 25, 29, 35, 39, 45, 49, 59, 69, 100]
age_labels = ['0-19', '20-25', '26-29', '30-35', '36-39', '40-45', '46-49', '50-59', '60-69', '70+']
customers_df['age_bin'] = pd.cut(customers_df['age'], bins=age_bins, labels=age_labels, right=False)
customers_df['age_bin'] = customers_df['age_bin'].cat.add_categories('Unknown').fillna('Unknown').astype(str)

# --- 3. Load Transactions ---
print(f"Loading Transactions from: {TRANSACTIONS_PATH}")
transactions_df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=['t_dat'], dtype={'article_id': str})

# Filter last year data
val_start_date = pd.to_datetime('2020-09-09')
train_start_date = val_start_date - pd.DateOffset(years=1)
train_df = transactions_df[
    (transactions_df['t_dat'] < val_start_date) & 
    (transactions_df['t_dat'] >= train_start_date)
].copy()

# Feature Engineering
train_df['month_of_year'] = train_df['t_dat'].dt.month.astype(str)
train_df['week_of_month'] = ((train_df['t_dat'].dt.day - 1) // 7 + 1).astype(str)
interactions_df = train_df[['customer_id', 'article_id', 'month_of_year', 'week_of_month']]

print(f"✅ Training dataset ready: {len(interactions_df)} rows.")

# Clean up RAM
del transactions_df, train_df
"""
cell_3 = nbf.v4.new_code_cell(text_3)
cell_3.metadata = {"cellView": "form", "id": "data_load_cell"}

# -------------------------------------------------------------------------
# CELL 4: PREPROCESSING & PIPELINE
# -------------------------------------------------------------------------
text_4 = """# @title 🔧 Step 3: Preprocessing & Lookup Tables
# @markdown Creating string lookup tables for features (Age, Product Group, Department, etc.)

customer_ids = customers_df['customer_id'].unique()
article_ids = articles_df['article_id'].unique()
age_groups = customers_df['age_bin'].unique()
months = [str(i) for i in range(1, 13)]
weeks = [str(i) for i in range(1, 6)]

print("Creating Lookup Tables...")
cust_age_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(customers_df['customer_id'], customers_df['age_bin']),
    default_value='Unknown'
)

article_tables = {}
feature_cols = ['product_type_name', 'product_group_name', 'colour_group_name', 'department_name']
for col in feature_cols:
    article_tables[col] = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(articles_df['article_id'], articles_df[col]),
        default_value='Unknown'
    )

# Create Datasets
articles_ds = tf.data.Dataset.from_tensor_slices(dict(articles_df))
interactions_ds = tf.data.Dataset.from_tensor_slices(dict(interactions_df))

def add_features(features):
    features['age_bin'] = cust_age_table.lookup(features['customer_id'])
    for col in feature_cols:
        features[col] = article_tables[col].lookup(features['article_id'])
    return features

interactions_ds = interactions_ds.map(add_features, num_parallel_calls=tf.data.AUTOTUNE)
print("✅ Lookup tables and pipelines created.")
"""
cell_4 = nbf.v4.new_code_cell(text_4)
cell_4.metadata = {"cellView": "form", "id": "prep_cell"}

# -------------------------------------------------------------------------
# CELL 5: MODEL DEFINITION
# -------------------------------------------------------------------------
text_5 = """# @title 🧠 Step 4: Define Two-Tower Model
# @markdown Defining User Tower and Item Tower architectures using TensorFlow Recommenders.

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.customer_id_lookup = StringLookup(vocabulary=customer_ids, mask_token=None)
        self.customer_id_emb = Embedding(len(customer_ids) + 1, EMBEDDING_DIM)
        
        self.age_bin_lookup = StringLookup(vocabulary=age_groups, mask_token=None)
        self.age_bin_emb = Embedding(len(age_groups) + 1, EMBEDDING_DIM // 4)
        
        self.month_lookup = StringLookup(vocabulary=months, mask_token=None)
        self.month_emb = Embedding(len(months) + 1, EMBEDDING_DIM // 4)
        
        self.week_lookup = StringLookup(vocabulary=weeks, mask_token=None)
        self.week_emb = Embedding(len(weeks) + 1, EMBEDDING_DIM // 4)
        self.projection = Dense(EMBEDDING_DIM)

    def call(self, inputs):
        x = tf.concat([
            self.customer_id_emb(self.customer_id_lookup(inputs['customer_id'])),
            self.age_bin_emb(self.age_bin_lookup(inputs['age_bin'])),
            self.month_emb(self.month_lookup(inputs['month_of_year'])),
            self.week_emb(self.week_lookup(inputs['week_of_month'])),
        ], axis=1)
        return self.projection(x)

class ItemModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.article_id_lookup = StringLookup(vocabulary=article_ids, mask_token=None)
        self.article_id_emb = Embedding(len(article_ids) + 1, EMBEDDING_DIM)
        
        self.lookups = {}
        self.embeddings = {}
        for col in feature_cols:
            vocab = articles_df[col].unique()
            self.lookups[col] = StringLookup(vocabulary=vocab, mask_token=None)
            self.embeddings[col] = Embedding(len(vocab) + 1, EMBEDDING_DIM // 4)
        self.projection = Dense(EMBEDDING_DIM)

    def call(self, inputs):
        embs = [self.article_id_emb(self.article_id_lookup(inputs['article_id']))]
        for col in feature_cols:
            embs.append(self.embeddings[col](self.lookups[col](inputs[col])))
        x = tf.concat(embs, axis=1)
        return self.projection(x)

class HMRModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_model = UserModel()
        self.item_model = ItemModel()
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features)
        item_embeddings = self.item_model(features)
        return self.task(user_embeddings, item_embeddings)

print("✅ Model architecture defined.")
"""
cell_5 = nbf.v4.new_code_cell(text_5)
cell_5.metadata = {"cellView": "form", "id": "model_def_cell"}

# -------------------------------------------------------------------------
# CELL 6: TRAINING
# -------------------------------------------------------------------------
text_6 = """# @title 🏋️ Step 5: Train the Model
# @markdown Training the Two-Tower model with Adagrad optimizer.
# @markdown *Note: This uses the Epochs and Learning Rate defined in Step 1.*

# Cache data in RAM
cached_train = interactions_ds.shuffle(100_000).batch(16384).cache().prefetch(tf.data.AUTOTUNE)

model = HMRModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE))

print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(cached_train, epochs=EPOCHS)
print("✅ Training finished.")
"""
cell_6 = nbf.v4.new_code_cell(text_6)
cell_6.metadata = {"cellView": "form", "id": "train_cell"}

# -------------------------------------------------------------------------
# CELL 7: SCANN INDEXING
# -------------------------------------------------------------------------
text_7 = """# @title 🔍 Step 6: Build ScaNN Index
# @markdown Building Approximate Nearest Neighbor index for fast retrieval.

print("Building ScaNN index...")
scann_index = tfrs.layers.factorized_top_k.ScaNN(
    model.user_model,
    num_reordering_candidates=500,
    num_leaves=1000,
    num_leaves_to_search=30,
    k=50
)

# Index all items
candidate_dataset = articles_ds.batch(2048).map(lambda x: (x["article_id"], model.item_model(x)))
scann_index.index_from_dataset(candidate_dataset)

# Build check
sample_query = {
    "customer_id": tf.constant([customer_ids[0]]),
    "age_bin": tf.constant([age_groups[0]]),
    "month_of_year": tf.constant(["9"]),
    "week_of_month": tf.constant(["2"])
}
_ = scann_index(sample_query)
print("✅ ScaNN index built successfully.")
"""
cell_7 = nbf.v4.new_code_cell(text_7)
cell_7.metadata = {"cellView": "form", "id": "scann_cell"}

# -------------------------------------------------------------------------
# CELL 8: SAVE TO GCS
# -------------------------------------------------------------------------
text_8 = """# @title 💾 Step 7: Save Model to GCS
# @markdown Saving the serving-ready model to Google Cloud Storage.

class ServingModel(tf.keras.Model):
    def __init__(self, index_layer):
        super().__init__()
        self.index_layer = index_layer

    @tf.function(input_signature=[{
        "customer_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
        "age_bin": tf.TensorSpec(shape=(None,), dtype=tf.string),
        "month_of_year": tf.TensorSpec(shape=(None,), dtype=tf.string),
        "week_of_month": tf.TensorSpec(shape=(None,), dtype=tf.string)
    }])
    def call(self, features):
        return self.index_layer(features)

serving_model = ServingModel(scann_index)
_ = serving_model(sample_query) # Final build check

MODEL_SAVE_PATH = os.path.join(GCS_BASE_PATH, 'models/two-tower-model')
print(f"Saving model to: {MODEL_SAVE_PATH}")
tf.saved_model.save(serving_model, MODEL_SAVE_PATH)
print("✅ Model saved successfully.")
"""
cell_8 = nbf.v4.new_code_cell(text_8)
cell_8.metadata = {"cellView": "form", "id": "save_cell"}

# -------------------------------------------------------------------------
# CELL 9: DEPLOYMENT (CLOUD RUN)
# -------------------------------------------------------------------------
text_9 = """# @title 🚀 Step 8: Deploy to Cloud Run (Serverless)
# @markdown We will create a Flask app, package it with Docker, and deploy it to Cloud Run.

import os

# 1. Create Deployment Directory
os.makedirs("deploy_app", exist_ok=True)

# 2. Write Flask App
app_code = f\"\"\"
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
from flask import Flask, request, jsonify

app = Flask(__name__)

LOCAL_MODEL_PATH = "/app/model"
MODEL_GCS_PATH = "{MODEL_SAVE_PATH}"

# Download Model at startup
print("Downloading model from GCS...")
os.system(f"gsutil -m cp -r {{MODEL_GCS_PATH}}/* {{LOCAL_MODEL_PATH}}")

print("Loading model...")
model = tf.saved_model.load(LOCAL_MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        inputs = {{
            "customer_id": tf.constant([data['customer_id']]),
            "age_bin": tf.constant([data['age_bin']]),
            "month_of_year": tf.constant([data['month_of_year']]),
            "week_of_month": tf.constant([data['week_of_month']])
        }}
        scores, ids = model(inputs)
        recommendations = [id.decode('utf-8') for id in ids.numpy()[0]]
        return jsonify({{"recommendations": recommendations}})
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
\"\"\"

with open("deploy_app/main.py", "w") as f:
    f.write(app_code)

# 3. Write Dockerfile
dockerfile_code = \"\"\"
FROM python:3.9-slim
RUN apt-get update && apt-get install -y curl gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \\
    apt-get update -y && apt-get install google-cloud-sdk -y
WORKDIR /app
RUN mkdir -p /app/model
RUN pip install flask gunicorn tensorflow tensorflow-recommenders scann
COPY main.py .
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app
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
  --memory 2Gi

print("✅ Deployment Complete! Click the URL above to test your API.")
"""
cell_9 = nbf.v4.new_code_cell(text_9)
cell_9.metadata = {"cellView": "form", "id": "deploy_cell"}


# Add cells to notebook
nb.cells.extend([cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7, cell_8, cell_9])

# Save the file
with open('hm_two_tower_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("🎉 'hm_two_tower_training.ipynb' has been successfully created!")