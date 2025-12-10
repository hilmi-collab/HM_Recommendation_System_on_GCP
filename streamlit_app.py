import streamlit as st
import pandas as pd
import requests
import math
import os

# ==============================================================================
# 1. SETTINGS
# ==============================================================================
st.set_page_config(page_title="H&M AI Shop", layout="wide", page_icon="🛍️")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_image_url(article_id):
    # Whatever the ID is (int or str), convert to string first
    # Then strip whitespace
    # Then pad with zeros to ensure 10 digits
    article_id = str(article_id).strip().zfill(10)
    
    base_url = "https://repo.hops.works/dev/jdowling/h-and-m/images"
    folder = article_id[:3] 
    return f"{base_url}/{folder}/{article_id}.jpg"

def sigmoid(x):
    try: return 1 / (1 + math.exp(-x))
    except: return 0.0

# ==============================================================================
# 3. LOAD DATA
# ==============================================================================
@st.cache_data
def load_data():
    try:
        # Check if local parquet files exist
        if not os.path.exists('app_customers.parquet'):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        cust = pd.read_parquet('app_customers.parquet')
        art = pd.read_parquet('app_articles.parquet')
        truth = pd.read_parquet('val_truth.parquet')
        return cust, art, truth
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

customers, articles_df, truth_df = load_data()

# ==============================================================================
# 4. API COMMUNICATION
# ==============================================================================
def get_recommendations(api_url, uid, history):
    # Ensure URL ends with /predict
    if not api_url.endswith("/predict"):
        api_url = f"{api_url.rstrip('/')}/predict"

    payload = {"user_id": str(uid), "history": history}
    
    try:
        response = requests.post(api_url, json=payload, timeout=240)
        
        if response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return []
            
        data = response.json()
        
        if isinstance(data, list): return data
        elif isinstance(data, dict): return data.get('recommendations', [])
        else: return []
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

# ==============================================================================
# 5. USER INTERFACE
# ==============================================================================
st.title("🛍️ H&M AI Recommender System")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
default_api = "https://YOUR-CLOUD-RUN-URL.run.app"
api_url = st.sidebar.text_input("Backend API URL:", value=default_api)

st.sidebar.divider()
st.sidebar.header("👤 Customer Selection")

# --- SMART USER FILTER ---
if not truth_df.empty:
    # Filter users who shopped during the Validation week
    # And sort by most active users (Active users on top)
    active_users = truth_df['customer_id'].value_counts().index.tolist()
    
    # Put the top 2000 active users in the dropdown
    display_users = active_users[:2000]
    
    st.sidebar.info(f"There are {len(active_users)} active users in the test week.")
else:
    display_users = []
    st.sidebar.warning("Validation data not found.")

# Search and Select
manual_id = st.sidebar.text_input("🔍 Search by ID (Paste):")
dropdown_user = st.sidebar.selectbox("🏆 Top Active Users:", display_users)

# Selection Logic
if manual_id:
    selected_user = manual_id.strip()
else:
    selected_user = dropdown_user

# Session State
if 'active_user' not in st.session_state:
    st.session_state.active_user = selected_user
    st.session_state.cart = [] 

if st.session_state.active_user != selected_user:
    st.session_state.cart = [] 
    st.session_state.active_user = selected_user

# User Info / Metadata
if selected_user and not customers.empty:
    user_row = customers[customers['customer_id'] == selected_user]
    
    if not user_row.empty:
        age = int(user_row.iloc[0]['age'])
        status = user_row.iloc[0]['club_member_status']
        st.sidebar.success(f"Profile Loaded")
        st.sidebar.info(f"**Age:** {age} | **Status:** {status}")
    else:
        st.sidebar.warning("User metadata not found (Cold User).")

# Cart
st.sidebar.divider()
st.sidebar.subheader(f"🛒 Cart ({len(st.session_state.cart)})")
if st.sidebar.button("Clear Cart"):
    st.session_state.cart = []
    st.rerun()

# --- MAIN SCREEN ---
if "YOUR-CLOUD-RUN-URL" in api_url:
    st.warning("👈 Please enter your Backend URL.")
elif selected_user:
    # Real Ground Truth Data
    real_purchases = []
    if not truth_df.empty:
        real_purchases = truth_df[truth_df['customer_id'] == selected_user]['article_id'].tolist()
        real_purchases = [str(x) for x in real_purchases]

    st.markdown(f"### 🤖 AI Recommendations: `{selected_user[:10]}...`")
    
    with st.spinner("Preparing recommendations..."):
        recs = get_recommendations(api_url, selected_user, st.session_state.cart)
    
    rec_ids = []
    for r in recs:
        if isinstance(r, dict):
            rec_ids.append(str(r.get('article_id')))

    if recs:
        cols = st.columns(4)
        for i, item in enumerate(recs):
            with cols[i % 4]:
                if not isinstance(item, dict): continue
                
                aid = str(item.get('article_id', 'Unknown'))
                img = get_image_url(aid)
                score_val = sigmoid(item.get('score', 0))
                
                st.image(img, use_container_width=True)
                st.markdown(f"**{item.get('prod_name', 'Product')}**")
                
                if aid in real_purchases:
                    st.success(f"🎯 HIT! ({score_val:.2f})")
                else:
                    st.caption(f"Confidence: {score_val:.2f}")
                
                if st.button("Add", key=f"btn_{i}_{aid}"):
                    st.session_state.cart.append(aid)
                    st.rerun()
    else:
        st.info("No recommendations found for this user.")
    
    st.divider()
    
    # Ground Truth Section
    if real_purchases:
        st.subheader(f"🛒 What did they actually buy? ({len(real_purchases)} Items)")
        cols = st.columns(6)
        for i, aid in enumerate(real_purchases):
            with cols[i%6]:
                st.image(get_image_url(aid), use_container_width=True)
                if aid in rec_ids:
                    st.markdown(":green[**✅ Model Hit**]")
                else:
                    st.markdown(":red[**❌ Missed**]")
    else:
        st.info("This user did not make any purchases during the test week (Sept 2020).")