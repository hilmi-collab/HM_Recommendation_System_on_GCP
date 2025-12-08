#!/bin/bash

# ==============================================================================
# H&M RECOMMENDATION WORKSHOP - SETUP SCRIPT
# ==============================================================================

# Color definitions for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# 🔧 CONFIGURATION (UPDATE THESE URLs)
# ------------------------------------------------------------------------------
# Replace 'YOUR_GITHUB_USER' and 'YOUR_REPO_NAME' with your actual details.
GITHUB_BASE_URL="https://raw.githubusercontent.com/hilmi-collab/HM_Recommendation_System_on_GCP/main"

NOTEBOOK_1="hm_two_tower_training.ipynb"
NOTEBOOK_2="hm_ranking_lightgbm_training.ipynb"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   H&M Recommendation Workshop - Automated Setup      ${NC}"
echo -e "${BLUE}======================================================${NC}"

# ------------------------------------------------------------------------------
# 1. PROJECT & USER INFO
# ------------------------------------------------------------------------------
PROJECT_ID=$(gcloud config get-value project)
USER_EMAIL=$(gcloud config get-value account)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Create a unique bucket name to avoid conflicts
BUCKET_NAME="hm-workshop-${PROJECT_ID}"

echo -e "${YELLOW}[Info]${NC} Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "${YELLOW}[Info]${NC} User Email: ${GREEN}$USER_EMAIL${NC}"
echo -e "${YELLOW}[Info]${NC} Target Bucket: ${GREEN}$BUCKET_NAME${NC}"
echo ""

# ------------------------------------------------------------------------------
# 2. ENABLE APIs
# ------------------------------------------------------------------------------
echo -e "${BLUE}[Step 1/5] Enabling Google Cloud APIs...${NC}"
gcloud services enable \
    aiplatform.googleapis.com \
    dataform.googleapis.com \
    compute.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    serviceusage.googleapis.com \
    storage-component.googleapis.com > /dev/null 2>&1

echo -e "${GREEN}✔ APIs enabled successfully.${NC}"

# ------------------------------------------------------------------------------
# 3. IAM ROLES & PERMISSIONS
# ------------------------------------------------------------------------------
echo -e "${BLUE}[Step 2/5] Configuring IAM Permissions...${NC}"

# Grant necessary roles to the user
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/aiplatform.colabEnterpriseAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/storage.admin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/serviceusage.serviceUsageAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/run.admin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/iam.serviceAccountUser" > /dev/null 2>&1

# Grant permissions to the default Compute Service Account (used by Colab runtime)
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/storage.objectAdmin" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/aiplatform.user" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/artifactregistry.writer" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/cloudbuild.builds.builder" > /dev/null 2>&1
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$COMPUTE_SA" --role="roles/run.admin" > /dev/null 2>&1

echo -e "${GREEN}✔ IAM permissions assigned.${NC}"

# ------------------------------------------------------------------------------
# 4. CREATE COLAB RUNTIME TEMPLATE (FIXED WITH BETA)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[Step 3/5] Creating Colab Runtime Template...${NC}"
TEMPLATE_NAME="hm-workshop-gpu-template"

# Hata düzeltmesi: 'gcloud ai' yerine 'gcloud beta ai' kullanıyoruz.
if gcloud beta ai runtime-templates list --region=us-central1 --filter="displayName=$TEMPLATE_NAME" --format="value(name)" 2>/dev/null | grep -q "$TEMPLATE_NAME"; then
    echo -e "${YELLOW}[Info]${NC} Template '$TEMPLATE_NAME' already exists. Skipping."
else
    # Create template using BETA command to avoid "Invalid choice" error
    gcloud beta ai runtime-templates create $TEMPLATE_NAME \
        --project=$PROJECT_ID \
        --region=us-central1 \
        --display-name=$TEMPLATE_NAME \
        --machine-type=n1-standard-4 \
        --accelerator-type=NVIDIA_TESLA_T4 \
        --accelerator-count=1 > /dev/null 2>&1
        
    # Check if creation was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✔ Runtime Template ($TEMPLATE_NAME) created.${NC}"
    else
        echo -e "${RED}[Error] Failed to create Runtime Template. Please check quotas or permissions.${NC}"
        # Scripti durdurmuyoruz, manuel olarak oluşturabilirler diye devam ediyor.
    fi
fi

# ------------------------------------------------------------------------------
# 5. SETUP STORAGE & NOTEBOOKS
# ------------------------------------------------------------------------------
echo -e "${BLUE}[Step 4/5] Setting up Storage and Uploading Notebooks...${NC}"

# Create Bucket if it doesn't exist
if gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    echo -e "${YELLOW}[Info]${NC} Bucket gs://$BUCKET_NAME already exists."
else
    gsutil mb -l us-central1 gs://$BUCKET_NAME > /dev/null 2>&1
    echo -e "${GREEN}✔ Bucket created: gs://$BUCKET_NAME${NC}"
fi

# Function to download and upload notebook
upload_notebook() {
    local NB_NAME=$1
    local URL="$GITHUB_BASE_URL/$NB_NAME"
    
    echo -e "${YELLOW}[Info]${NC} Processing $NB_NAME..."
    
    # Download
    wget -q $URL -O $NB_NAME
    
    if [ -f "$NB_NAME" ]; then
        # Upload to GCS
        gsutil cp $NB_NAME gs://$BUCKET_NAME/notebooks/$NB_NAME > /dev/null 2>&1
        rm $NB_NAME
        echo -e "${GREEN}  -> Uploaded to Colab Storage.${NC}"
    else
        echo -e "${RED}[Error] Failed to download $NB_NAME. Check URL.${NC}"
    fi
}

# Upload both notebooks
upload_notebook $NOTEBOOK_1
upload_notebook $NOTEBOOK_2

# ------------------------------------------------------------------------------
# 6. COMPLETION
# ------------------------------------------------------------------------------
echo -e "${BLUE}[Step 5/5] Setup Complete!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "🚀  **WORKSHOP ENVIRONMENT READY**"
echo -e ""
echo -e "1. Go to **Colab Enterprise** in Google Cloud Console."
echo -e "2. Click **'File' -> 'Open notebook'**."
echo -e "3. Select the **'Google Cloud Storage'** tab."
echo -e "4. Navigate to: ${YELLOW}$BUCKET_NAME > notebooks${NC}"
echo -e "5. You will see both files:"
echo -e "   - ${GREEN}$NOTEBOOK_1${NC} (Start with this)"
echo -e "   - ${GREEN}$NOTEBOOK_2${NC}"
echo -e "${GREEN}======================================================${NC}"