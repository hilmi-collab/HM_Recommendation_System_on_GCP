#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# REPLACE THIS URL with your actual GitHub raw URL for the notebook
NOTEBOOK_URL="https://raw.githubusercontent.com/hilmi-collab/HM_Recommendation_System_on_GCP/main/hm_two_tower_training.ipynb"
NOTEBOOK_FILENAME="hm_two_tower_training.ipynb"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   H&M Recommendation Workshop - Automated Setup      ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. Get Project Info
PROJECT_ID=$(gcloud config get-value project)
USER_EMAIL=$(gcloud config get-value account)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Unique Bucket Name: hm-workshop-[PROJECT_ID]
BUCKET_NAME="hm-workshop-${PROJECT_ID}"

echo -e "${YELLOW}[Info]${NC} Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "${YELLOW}[Info]${NC} Target Bucket: ${GREEN}$BUCKET_NAME${NC}"
echo ""

# 2. Enable APIs
echo -e "${BLUE}[Step 1/5] Enabling Google Cloud APIs...${NC}"
gcloud services enable \
    aiplatform.googleapis.com \
    dataform.googleapis.com \
    compute.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    serviceusage.googleapis.com \
    storage-component.googleapis.com > /dev/null 2>&1

echo -e "${GREEN}✔ APIs enabled.${NC}"

# 3. IAM Roles
echo -e "${BLUE}[Step 2/5] Configuring IAM Permissions...${NC}"
# User roles
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/aiplatform.colabEnterpriseAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/storage.admin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/serviceusage.serviceUsageAdmin" > /dev/null 2>&1

# Default Compute SA roles
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/storage.objectAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/aiplatform.user" > /dev/null 2>&1

echo -e "${GREEN}✔ Permissions assigned.${NC}"

# 4. Create Runtime Template
echo -e "${BLUE}[Step 3/5] Creating Colab Runtime Template...${NC}"
TEMPLATE_NAME="hm-workshop-gpu-template"

if gcloud ai runtime-templates list --region=us-central1 --filter="displayName=$TEMPLATE_NAME" --format="value(name)" | grep -q "$TEMPLATE_NAME"; then
    echo -e "${YELLOW}[Info]${NC} Template '$TEMPLATE_NAME' already exists."
else
    gcloud ai runtime-templates create $TEMPLATE_NAME \
        --project=$PROJECT_ID \
        --region=us-central1 \
        --display-name=$TEMPLATE_NAME \
        --machine-type=n1-standard-4 \
        --accelerator-type=NVIDIA_TESLA_T4 \
        --accelerator-count=1 > /dev/null 2>&1
    echo -e "${GREEN}✔ Runtime Template created.${NC}"
fi

# 5. Create Bucket & Upload Notebook
echo -e "${BLUE}[Step 4/5] Setting up Storage & Notebooks...${NC}"

# Check if bucket exists, if not create it
if gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    echo -e "${YELLOW}[Info]${NC} Bucket gs://$BUCKET_NAME already exists."
else
    gsutil mb -l us-central1 gs://$BUCKET_NAME > /dev/null 2>&1
    echo -e "${GREEN}✔ Bucket created: gs://$BUCKET_NAME${NC}"
fi

# Download Notebook from GitHub
echo -e "${YELLOW}[Info]${NC} Downloading notebook from GitHub..."
wget -q $NOTEBOOK_URL -O $NOTEBOOK_FILENAME

# Upload to Bucket
echo -e "${YELLOW}[Info]${NC} Uploading notebook to Colab storage..."
gsutil cp $NOTEBOOK_FILENAME gs://$BUCKET_NAME/notebooks/$NOTEBOOK_FILENAME > /dev/null 2>&1

# Clean up local file
rm $NOTEBOOK_FILENAME

echo -e "${GREEN}✔ Notebook uploaded to: gs://$BUCKET_NAME/notebooks/$NOTEBOOK_FILENAME${NC}"

# 6. Finish
echo -e "${BLUE}[Step 5/5] Setup Complete!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "🚀  **READY TO START!**"
echo -e "1. Open Colab Enterprise in Cloud Console."
echo -e "2. Click 'Open from Cloud Storage'."
echo -e "3. Navigate to bucket: ${YELLOW}$BUCKET_NAME${NC} -> notebooks"
echo -e "4. Open ${YELLOW}$NOTEBOOK_FILENAME${NC}"
echo -e "${GREEN}======================================================${NC}"
