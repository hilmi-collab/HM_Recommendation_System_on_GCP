#!/bin/bash

# ==============================================================================
# H&M RECOMMENDATION WORKSHOP - SETUP SCRIPT (HIGH PERFORMANCE - DEBUG MODU)
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'  # Hata mesajları için kırmızı renk eklendi
NC='\033[0m' 

# 🔧 CONFIGURATION
GITHUB_BASE_URL="https://raw.githubusercontent.com/hilmi-collab/HM_Recommendation_System_on_GCP/main"
# Note: Ensure these filenames match the ones in your repository
NOTEBOOK_1="hm_two_tower_training.ipynb"
NOTEBOOK_2="hm_ranking_lightgbm_training.ipynb"

# Frontend Files
APP_FILE="streamlit_app.py"
REQ_FILE="requirements.txt"
DOCKER_FILE="Dockerfile"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   H&M Workshop Setup (High Performance Mode)         ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. PROJECT INFO
PROJECT_ID=$(gcloud config get-value project)
USER_EMAIL=$(gcloud config get-value account)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
WORK_BUCKET_NAME="hm-workshop-${PROJECT_ID}"

echo -e "${YELLOW}[Info]${NC} Project: ${GREEN}$PROJECT_ID${NC}"
echo -e "${YELLOW}[Info]${NC} Work Bucket: ${GREEN}$WORK_BUCKET_NAME${NC}"

# 2. APIs
echo -e "${BLUE}[Step 1/5] Enabling APIs...${NC}"
gcloud services enable aiplatform.googleapis.com dataform.googleapis.com compute.googleapis.com run.googleapis.com cloudbuild.googleapis.com serviceusage.googleapis.com storage-component.googleapis.com > /dev/null 2>&1

# 3. IAM
echo -e "${BLUE}[Step 2/5] Configuring IAM...${NC}"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/aiplatform.colabEnterpriseAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/storage.admin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/run.admin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/iam.serviceAccountUser" > /dev/null 2>&1

COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/storage.objectAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/aiplatform.user" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/artifactregistry.writer" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/cloudbuild.builds.builder" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/run.admin" > /dev/null 2>&1

# 4. RUNTIME TEMPLATES (GÜNCELLENMİŞ FONKSİYON)
echo -e "${BLUE}[Step 3/5] Creating High-Performance Runtime Templates...${NC}"
REGION="us-central1"


create_template() {
    local T_NAME=$1
    local M_TYPE=$2
    

    local NETWORK_URI="projects/$PROJECT_ID/global/networks/default"
    local SUBNET_URI="projects/$PROJECT_ID/regions/$REGION/subnetworks/default"
    
    if gcloud colab runtime-templates list --region=$REGION --filter="displayName=$T_NAME" --format="value(name)" 2>/dev/null | grep -q "$T_NAME"; then
        echo -e "${YELLOW}[Info]${NC} Template '$T_NAME' exists."
    else
        echo -e "Creating template: ${YELLOW}$T_NAME${NC} ($M_TYPE)..."
        

        gcloud colab runtime-templates create \
            --project=$PROJECT_ID \
            --region=$REGION \
            --display-name=$T_NAME \
            --machine-type=$M_TYPE \
            --accelerator-type=NVIDIA_TESLA_T4 \
            --accelerator-count=1 \
            --network=$NETWORK_URI \
            --subnetwork=$SUBNET_URI > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then 
            echo -e "${GREEN}✔ $T_NAME created.${NC}"
        else 
            echo -e "${YELLOW}[Warning] Standard creation failed. Retrying with 'gcloud beta'...${NC}"

            gcloud beta colab runtime-templates create \
                --project=$PROJECT_ID \
                --region=$REGION \
                --display-name=$T_NAME \
                --machine-type=$M_TYPE \
                --accelerator-type=NVIDIA_TESLA_T4 \
                --accelerator-count=1 \
                --network=$NETWORK_URI \
                --subnetwork=$SUBNET_URI
            
            if [ $? -ne 0 ]; then
                echo -e "${RED}❌ Error: Template creation failed.${NC}"
            fi
        fi
    fi
}



# RAM is crucial for Retrieval (n1-standard-8 -> 30GB RAM)
create_template "hm-retrieval-gpu-template" "n1-standard-8"

# CPU is crucial for Ranking (n1-standard-16 -> 16 vCPU, 60GB RAM)
# This power is required for processing 6 weeks of data.
create_template "hm-ranking-gpu-template" "n1-standard-16"

# 5. WORK BUCKET & NOTEBOOKS
echo -e "${BLUE}[Step 4/5] Setting up User Bucket & Notebooks...${NC}"
if ! gsutil ls -b gs://$WORK_BUCKET_NAME > /dev/null 2>&1; then
    gsutil mb -l us-central1 gs://$WORK_BUCKET_NAME > /dev/null 2>&1
    echo -e "${GREEN}✔ Bucket created: $WORK_BUCKET_NAME${NC}"
fi

upload_notebook() {
    wget -q "$GITHUB_BASE_URL/$1" -O $1
    if [ -f "$1" ]; then
        gsutil cp $1 gs://$WORK_BUCKET_NAME/notebooks/$1 > /dev/null 2>&1
        rm $1
        echo -e "${GREEN}  -> $1 uploaded.${NC}"
    fi
}
upload_notebook $NOTEBOOK_1
upload_notebook $NOTEBOOK_2

# 6. FRONTEND FILES
echo -e "${BLUE}[Step 5/5] Downloading Frontend Files...${NC}"
mkdir -p hm_frontend
cd hm_frontend
wget -q "$GITHUB_BASE_URL/$APP_FILE" -O $APP_FILE
wget -q "$GITHUB_BASE_URL/$REQ_FILE" -O $REQ_FILE
wget -q "$GITHUB_BASE_URL/$DOCKER_FILE" -O $DOCKER_FILE
cd ..

echo -e "${GREEN}======================================================${NC}"
echo -e "🚀  **READY (HIGH PERFORMANCE)**"
echo -e "Notebooks: gs://$WORK_BUCKET_NAME/notebooks"