import streamlit as st
import pandas as pd
import requests
import math
import os
from google.cloud import storage


# ==============================================================================
# 1. AYARLAR & KONFİGÜRASYON
# ==============================================================================
st.set_page_config(page_title="H&M AI Shop", layout="wide")

# Cloud Run'a deploy ederken environment variable olarak vereceğiz
BUCKET_NAME = os.environ.get("BUCKET_NAME", "hm-recommendation-workshop") 

# ==============================================================================
# 2. YARDIMCI FONKSİYONLAR
# ==============================================================================
def get_image_url(article_id):
    """Generates the correct image URL from Article ID."""
    article_id = str(article_id)
    base_url = "https://repo.hops.works/dev/jdowling/h-and-m/images"
    folder = article_id[:3] 
    return f"{base_url}/{folder}/{article_id}.jpg"

def sigmoid(x):
    """Squeezes the logit score between 0-1."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

# ==============================================================================
# 3. VERİ YÜKLEME (GCS & LOCAL)
# ==============================================================================
@st.cache_data
def load_data():
    files = ['app_customers.parquet', 'app_articles.parquet', 'val_truth.parquet']
    
    # Check if files exist, if not try to download from GCS
    for f in files:
        if not os.path.exists(f):
            try:
                # Artifacts are stored under models/ranking_model in the bucket
                gcs_path = f"models/ranking_model/{f}"
                st.toast(f"Downloading {f} from GCS...", icon="☁️")
                download_blob(BUCKET_NAME, gcs_path, f)
            except Exception as e:
                st.warning(f"Could not download {f}: {e}")

    try:
        cust = pd.read_parquet('app_customers.parquet')
        art = pd.read_parquet('app_articles.parquet')
        try:
            truth = pd.read_parquet('val_truth.parquet')
        except:
            truth = pd.DataFrame(columns=['customer_id', 'article_id'])
        return cust, art, truth
    except Exception as e:
        st.error(f"Error loading data files. Make sure the training notebook ran successfully and artifacts are in GCS. Error: {e}")
        st.stop()

customers, articles_df, truth_df = load_data()

# ==============================================================================
# 4. API İLETİŞİMİ
# ==============================================================================
def get_recommendations(api_url, uid, history):
    payload = {"user_id": str(uid), "history": history}
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        
        try:
            data = response.json()
        except:
            st.error(f"Backend returned non-JSON response: {response.text}")
            return []

        if 'recommendations' in data:
            return data['recommendations']
        elif 'error' in data:
            st.error(f"Backend Error: {data['error']}")
            return []
        else:
            return []

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

# ==============================================================================
# 5. ARAYÜZ (UI)
# ==============================================================================
st.title("🛍️ H&M AI Recommender System")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")

# Dynamic API URL Input
default_api = "https://YOUR-CLOUD-RUN-URL.run.app/predict"
api_url = st.sidebar.text_input("Backend API URL:", value=default_api, help="Paste the URL of your deployed Ranking Service here.")

st.sidebar.divider()
st.sidebar.header("👤 User Panel")

valid_users = truth_df['customer_id'].unique()
all_users = customers['customer_id'].unique()
# Prioritize users with history
sorted_users = list(valid_users) + [u for u in all_users if u not in valid_users]

selected_user = st.sidebar.selectbox("Select Customer:", sorted_users)

# Session State
if 'active_user' not in st.session_state:
    st.session_state.active_user = selected_user
    st.session_state.cart = [] 

if st.session_state.active_user != selected_user:
    st.session_state.cart = [] 
    st.session_state.active_user = selected_user
    st.toast(f"User changed to: {selected_user[:10]}...", icon="🔄")

if selected_user in customers['customer_id'].values:
    user_info = customers[customers['customer_id'] == selected_user].iloc[0]
    st.sidebar.info(f"**Age:** {int(user_info['age'])}\n**Club Status:** {user_info['club_member_status']}")

# --- SIDEBAR: CART ---
st.sidebar.divider()
st.sidebar.subheader(f"🛒 Cart ({len(st.session_state.cart)})")

if st.session_state.cart:
    for item_id in st.session_state.cart:
        p_row = articles_df[articles_df['article_id'] == item_id]
        c1, c2 = st.sidebar.columns([1, 3])
        with c1:
            st.image(get_image_url(item_id), use_container_width=True)
        with c2:
            if not p_row.empty:
                st.markdown(f"**{p_row.iloc[0]['prod_name']}**")
            else:
                st.write(f"ID: {item_id}")

    if st.sidebar.button("🗑️ Clear Cart", use_container_width=True):
        st.session_state.cart = []
        st.rerun()
else:
    st.sidebar.caption("Cart is empty.")

# --- MAIN SCREEN ---

# 1. Ground Truth
if not truth_df.empty:
    actual_purchases = truth_df[truth_df['customer_id'] == selected_user]['article_id'].tolist()
    actual_purchases = [str(x) for x in actual_purchases]
else:
    actual_purchases = []

# 2. Fetch Recommendations
if "YOUR-CLOUD-RUN-URL" in api_url or not api_url:
    st.warning("⚠️ Please enter your valid Backend API URL in the sidebar.")
    recommendations = []
else:
    with st.spinner(f"🤖 AI is analyzing profile for {selected_user[:10]}..."):
        recommendations = get_recommendations(api_url, selected_user, st.session_state.cart)

recommended_ids = [str(item['article_id']) for item in recommendations]

# 3. Display Recommendations
st.subheader(f"Recommended for You")

if recommendations:
    cols = st.columns(4)
    for i, item in enumerate(recommendations):
        with cols[i % 4]:
            aid = str(item['article_id'])
            
            product_info = articles_df[articles_df['article_id'] == aid]
            if not product_info.empty:
                prod_name = product_info.iloc[0]['prod_name']
                category = product_info.iloc[0].get('product_group_name', 'General')
            else:
                prod_name = item['prod_name']
                category = "Unknown"

            # Check if prediction matches ground truth
            if aid in actual_purchases:
                st.success("✅ HIT!")
            
            st.image(get_image_url(aid), use_container_width=True)
            st.markdown(f"**{prod_name}**")
            st.caption(f"{category}")
            
            # Score Bar
            score_val = sigmoid(item['score'])
            st.progress(score_val, text=f"Score: {score_val:.2f}")
            
            if st.button("Add to Cart 🛒", key=f"rec_{aid}", use_container_width=True):
                st.session_state.cart.append(aid)
                st.rerun()
elif "YOUR-CLOUD-RUN-URL" not in api_url:
    st.info("No recommendations found or backend is warming up.")

# 4. Display Ground Truth
st.divider()
st.subheader("🛒 Actual Purchases (Test Data)")

if actual_purchases:
    st.caption(f"User bought {len(actual_purchases)} items in the test week.")
    cols_actual = st.columns(6)
    
    for i, aid in enumerate(actual_purchases):
        with cols_actual[i % 6]:
            img_url = get_image_url(aid)
            
            if aid in recommended_ids:
                st.markdown(":green[**✅ Found**]")
            else:
                st.markdown(":red[**❌ Missed**]")

            st.image(img_url, use_container_width=True)
else:
    st.info("No purchase history for this user in the test set.")