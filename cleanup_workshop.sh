#!/bin/bash

# ==============================================================================
# H&M WORKSHOP - RESOURCE CLEANUP SCRIPT
# ==============================================================================

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}======================================================${NC}"
echo -e "${RED}   H&M Workshop - RESOURCE CLEANUP                    ${NC}"
echo -e "${RED}======================================================${NC}"
echo -e "${YELLOW}INFO: This process will NOT delete your GCP Project. It only cleans up workshop resources.${NC}"
echo ""

# 1. Get Project Information
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="hm-workshop-${PROJECT_ID}"
REGION="us-central1"

# Service and Template Names
SERVICE_BACKEND="hm-recommender-service"
SERVICE_FRONTEND="hm-streamlit-ui"
TEMPLATE_RETRIEVAL="hm-retrieval-gpu-template"
TEMPLATE_RANKING="hm-ranking-gpu-template"

echo -e "Project to Process: ${BLUE}$PROJECT_ID${NC} (Project will be PRESERVED)"
echo -e "Bucket to Delete:   ${RED}$BUCKET_NAME${NC}"
echo ""

# 2. Request Confirmation
read -p "Do you want to clean up the resources? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi
echo ""

# ------------------------------------------------------------------------------
# 3. Delete Cloud Run Services
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Removing Cloud Run Services...${NC}"

gcloud run services delete $SERVICE_BACKEND --region=$REGION --quiet 2>/dev/null
if [ $? -eq 0 ]; then echo -e "${GREEN}✔ Backend Service deleted.${NC}"; else echo -e "${YELLOW}⚠ Backend service does not exist.${NC}"; fi

gcloud run services delete $SERVICE_FRONTEND --region=$REGION --quiet 2>/dev/null
if [ $? -eq 0 ]; then echo -e "${GREEN}✔ Frontend Service deleted.${NC}"; else echo -e "${YELLOW}⚠ Frontend service does not exist.${NC}"; fi


# ------------------------------------------------------------------------------
# 4. Delete Container Registry Images
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Cleaning Docker Images...${NC}"

delete_image() {
    local IMG_NAME="gcr.io/$PROJECT_ID/$1"
    # List and delete tags
    gcloud container images list-tags $IMG_NAME --format='get(digest)' 2>/dev/null | while read digest; do
        gcloud container images delete "$IMG_NAME@$digest" --force-delete-tags --quiet 2>/dev/null
    done
    # Delete repo
    gcloud container images delete $IMG_NAME --force-delete-tags --quiet 2>/dev/null
    echo -e "${GREEN}  -> $1 images cleaned.${NC}"
}

delete_image "hm-recommender-app" 
delete_image "hm-streamlit-app"   


# ------------------------------------------------------------------------------
# 5. Delete Colab Runtime Templates
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Deleting Colab Runtime Templates...${NC}"

delete_template() {
    local T_NAME=$1
    gcloud colab runtime-templates delete $T_NAME --region=$REGION --quiet 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✔ Template deleted: $T_NAME${NC}"
    else
        # Beta fallback
        gcloud beta colab runtime-templates delete $T_NAME --region=$REGION --quiet 2>/dev/null
        if [ $? -eq 0 ]; then
             echo -e "${GREEN}✔ Template deleted (Beta): $T_NAME${NC}"
        else
             echo -e "${YELLOW}⚠ Template not found: $T_NAME${NC}"
        fi
    fi
}

delete_template $TEMPLATE_RETRIEVAL
delete_template $TEMPLATE_RANKING


# ------------------------------------------------------------------------------
# 6. Delete GCS Bucket
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Cleaning Cloud Storage Bucket...${NC}"

if gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    gsutil -m rm -r gs://$BUCKET_NAME
    echo -e "${GREEN}✔ Bucket and all contents deleted.${NC}"
else
    echo -e "${YELLOW}⚠ Bucket does not exist ($BUCKET_NAME).${NC}"
fi


# ------------------------------------------------------------------------------
# 7. Clean Local Files
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Cleaning local files...${NC}"

rm -rf hm_frontend 2>/dev/null
rm -f hm_two_tower_training.ipynb 2>/dev/null
rm -f hm_ranking_lightgbm_training.ipynb 2>/dev/null
rm -f setup_workshop.sh 2>/dev/null

echo -e "${GREEN}✔ Cloud Shell working directory cleaned.${NC}"

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}   CLEANUP COMPLETED (PROJECT ACTIVE) 🧹   ${NC}"
echo -e "${GREEN}======================================================${NC}"