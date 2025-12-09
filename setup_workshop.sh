#!/bin/bash

# ==============================================================================
# H&M RECOMMENDATION WORKSHOP - SETUP SCRIPT
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

# 🔧 CONFIGURATION
GITHUB_BASE_URL="https://raw.githubusercontent.com/hilmi-collab/HM_Recommendation_System_on_GCP/main"
NOTEBOOK_1="hm_two_tower_training.ipynb"
NOTEBOOK_2="hm_ranking_lightgbm_training.ipynb"

# Frontend Files
APP_FILE="streamlit_app.py"
REQ_FILE="requirements.txt"
DOCKER_FILE="Dockerfile"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   H&M Recommendation Workshop - Automated Setup      ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. INFO
PROJECT_ID=$(gcloud config get-value project)
USER_EMAIL=$(gcloud config get-value account)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
BUCKET_NAME="hm-workshop-${PROJECT_ID}"

echo -e "${YELLOW}[Info]${NC} Project: ${GREEN}$PROJECT_ID${NC}"
echo -e "${YELLOW}[Info]${NC} Bucket: ${GREEN}$BUCKET_NAME${NC}"

# 2. APIs
echo -e "${BLUE}[Step 1/6] Enabling APIs...${NC}"
gcloud services enable aiplatform.googleapis.com dataform.googleapis.com compute.googleapis.com run.googleapis.com cloudbuild.googleapis.com serviceusage.googleapis.com storage-component.googleapis.com > /dev/null 2>&1
echo -e "${GREEN}✔ APIs enabled.${NC}"

# 3. IAM
echo -e "${BLUE}[Step 2/6] Configuring IAM...${NC}"
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
echo -e "${GREEN}✔ Permissions assigned.${NC}"

# 4. RUNTIME TEMPLATES (UPDATED)
echo -e "${BLUE}[Step 3/6] Creating Colab Runtime Templates...${NC}"
REGION="us-central1"

# Fonksiyon: Template oluşturucu (Kod tekrarını önlemek için)
create_template() {
    local T_NAME=$1
    local M_TYPE=$2
    
    # Check if exists
    if gcloud colab runtime-templates list --region=$REGION --filter="displayName=$T_NAME" --format="value(name)" 2>/dev/null | grep -q "$T_NAME"; then
        echo -e "${YELLOW}[Info]${NC} Template '$T_NAME' already exists. Skipping."
    else
        echo -e "Creating template: ${YELLOW}$T_NAME${NC} ($M_TYPE)..."
        
        # Try Standard
        gcloud colab runtime-templates create \
            --project=$PROJECT_ID \
            --region=$REGION \
            --display-name=$T_NAME \
            --machine-type=$M_TYPE \
            --accelerator-type=NVIDIA_TESLA_T4 \
            --accelerator-count=1 > /dev/null 2>&1
            
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✔ $T_NAME created.${NC}"
        else
            # Try Beta (Fallback)
            echo -e "${YELLOW}[Info] Standard command failed, trying BETA...${NC}"
            gcloud beta colab runtime-templates create \
                --project=$PROJECT_ID \
                --region=$REGION \
                --display-name=$T_NAME \
                --machine-type=$M_TYPE \
                --accelerator-type=NVIDIA_TESLA_T4 \
                --accelerator-count=1 > /dev/null 2>&1
                
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✔ $T_NAME created (via Beta).${NC}"
            else
                echo -e "${RED}[Error] Failed to create $T_NAME.${NC}"
            fi
        fi
    fi
}

# 1. Retrieval Template (High RAM - n1-standard-8)
create_template "hm-retrieval-gpu-template" "n1-standard-8"

# 2. Ranking Template (Standard RAM - n1-standard-4)
create_template "hm-ranking-gpu-template" "n1-standard-4"

# 5. STORAGE & NOTEBOOKS
echo -e "${BLUE}[Step 4/6] Setting up Storage & Notebooks...${NC}"
if ! gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    gsutil mb -l us-central1 gs://$BUCKET_NAME > /dev/null 2>&1
fi

upload_notebook() {
    wget -q "$GITHUB_BASE_URL/$1" -O $1
    if [ -f "$1" ]; then
        gsutil cp $1 gs://$BUCKET_NAME/notebooks/$1 > /dev/null 2>&1
        rm $1
        echo -e "${GREEN}  -> $1 uploaded.${NC}"
    fi
}
upload_notebook $NOTEBOOK_1
upload_notebook $NOTEBOOK_2

# 6. FRONTEND FILES
echo -e "${BLUE}[Step 5/6] Downloading Frontend Files...${NC}"
mkdir -p hm_frontend
cd hm_frontend
wget -q "$GITHUB_BASE_URL/$APP_FILE" -O $APP_FILE
wget -q "$GITHUB_BASE_URL/$REQ_FILE" -O $REQ_FILE
wget -q "$GITHUB_BASE_URL/$DOCKER_FILE" -O $DOCKER_FILE
cd ..
echo -e "${GREEN}✔ Frontend files downloaded to 'hm_frontend' folder.${NC}"

echo -e "${BLUE}[Step 6/6] Done!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "🚀  **READY**"
echo -e "Notebooks are in: gs://$BUCKET_NAME/notebooks"
echo -e "Frontend code is in: ~/hm_frontend"
echo -e ""
echo -e "Runtime Templates Created:"
echo -e "1. ${YELLOW}hm-retrieval-gpu-template${NC} (Use for Two-Tower Model)"
echo -e "2. ${YELLOW}hm-ranking-gpu-template${NC} (Use for Ranking Model)"