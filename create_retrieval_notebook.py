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
# @title ⚙️ Workshop Configuration
# @markdown Please enter your Project ID.

import os

# @markdown ### ☁️ Project Settings
PROJECT_ID = "YOUR_PROJECT_ID" # @param {type:"string"}
REGION = "us-central1" # @param {type:"string"}

# 1. PUBLIC DATA BUCKET (Read-Only)
# Raw data (CSV) will be read from here.
DATA_BUCKET_NAME = "hm-recommendation-workshop"
DATA_GCS_PATH = f"gs://{DATA_BUCKET_NAME}"

# 2. PRIVATE WORK BUCKET (Write)
# Trained models will be saved here.
WORK_BUCKET_NAME = f"hm-workshop-{PROJECT_ID}"
WORK_GCS_PATH = f"gs://{WORK_BUCKET_NAME}"

# @markdown ### 🚀 Model Hyperparameters
EMBEDDING_DIM = 64 # @param {type:"integer"}
LEARNING_RATE = 0.1 # @param {type:"number"}
EPOCHS = 5 # @param {type:"slider", min:1, max:10, step:1}

# Setup Environment
os.environ["GCLOUD_PROJECT"] = PROJECT_ID
os.environ["TF_USE_LEGACY_KERAS"] = "1"

print(f"✅ Config Set:")
print(f"   📥 Reading Data from: {DATA_GCS_PATH}")
print(f"   💾 Saving Models to:  {WORK_GCS_PATH}/models/two-tower-model")"""
    add_cell(cell_1)

    # --- CELL 2: Install Libraries ---
    cell_2 = """# @title 📥 Step 1: Install Libraries
!pip install -q tensorflow-recommenders --no-deps
!pip install -q --upgrade tensorflow-datasets
!pip install -q "scann[tf]" tensorflow-recommenders tensorflow-datasets
print("✅ Installation Complete.")"""
    add_cell(cell_2)

    # --- CELL 3: Load Data ---
    cell_3 = """# @title 💾 Step 2: Load Data from Public Bucket
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text, List
from tensorflow.keras.layers import StringLookup, Embedding, Dense

print(f"TensorFlow Version: {tf.__version__}")

# Paths (Reading from DATA_GCS_PATH)
ARTICLES_PATH = os.path.join(DATA_GCS_PATH, 'articles.csv')
CUSTOMERS_PATH = os.path.join(DATA_GCS_PATH, 'customers.csv')
TRANSACTIONS_PATH = os.path.join(DATA_GCS_PATH, 'transactions.csv')

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

val_start_date = pd.to_datetime('2020-09-09')
train_start_date = val_start_date - pd.DateOffset(years=1)
train_df = transactions_df[
    (transactions_df['t_dat'] < val_start_date) &
    (transactions_df['t_dat'] >= train_start_date)
].copy()

train_df['month_of_year'] = train_df['t_dat'].dt.month.astype(str)
train_df['week_of_month'] = ((train_df['t_dat'].dt.day - 1) // 7 + 1).astype(str)
interactions_df = train_df[['customer_id', 'article_id', 'month_of_year', 'week_of_month']]

print(f"✅ Training dataset ready: {len(interactions_df)} rows.")
del transactions_df, train_df"""
    add_cell(cell_3)

    # --- CELL 4: Preprocessing ---
    cell_4 = """# @title 🔧 Step 3: Preprocessing
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

articles_ds = tf.data.Dataset.from_tensor_slices(dict(articles_df))
interactions_ds = tf.data.Dataset.from_tensor_slices(dict(interactions_df))

def add_features(features):
    features['age_bin'] = cust_age_table.lookup(features['customer_id'])
    for col in feature_cols:
        features[col] = article_tables[col].lookup(features['article_id'])
    return features

interactions_ds = interactions_ds.map(add_features, num_parallel_calls=tf.data.AUTOTUNE)
print("✅ Lookup tables created.")"""
    add_cell(cell_4)

    # --- CELL 5: Define Model Architecture ---
    cell_5 = """# @title 🧠 Step 4: Define Two-Tower Model
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

print("✅ Model architecture defined.")"""
    add_cell(cell_5)

    # --- CELL 6: Training ---
    cell_6 = """# @title 🏋️ Step 5: Train the Model
cached_train = interactions_ds.shuffle(100_000).batch(16384).cache().prefetch(tf.data.AUTOTUNE)
model = HMRModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE))
print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(cached_train, epochs=EPOCHS)
print("✅ Training finished.")"""
    add_cell(cell_6)

    # --- CELL 7: ScaNN Index ---
    cell_7 = """# @title 🔍 Step 6: Build ScaNN Index
scann_index = tfrs.layers.factorized_top_k.ScaNN(
    model.user_model,
    num_reordering_candidates=500,
    num_leaves=1000,
    num_leaves_to_search=30,
    k=50
)
candidate_dataset = articles_ds.batch(2048).map(lambda x: (x["article_id"], model.item_model(x)))
scann_index.index_from_dataset(candidate_dataset)

sample_query = {
    "customer_id": tf.constant([customer_ids[0]]),
    "age_bin": tf.constant([age_groups[0]]),
    "month_of_year": tf.constant(["9"]),
    "week_of_month": tf.constant(["2"])
}
_ = scann_index(sample_query)
print("✅ ScaNN index built successfully.")"""
    add_cell(cell_7)

    # --- CELL 8: Save Model ---
    cell_8 = """# @title 💾 Step 7: Save Model to User Bucket
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
_ = serving_model(sample_query)

# SAVING THE MODEL TO PRIVATE WORK BUCKET
MODEL_SAVE_PATH = os.path.join(WORK_GCS_PATH, 'models/two-tower-model')
print(f"Saving model to: {MODEL_SAVE_PATH}")
tf.saved_model.save(serving_model, MODEL_SAVE_PATH)
print("✅ Model saved successfully.")"""
    add_cell(cell_8)

    # Save to file
    with open('hm_two_tower_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print("Successfully created 'hm_two_tower_training.ipynb'.")

if __name__ == "__main__":
    create_notebook()