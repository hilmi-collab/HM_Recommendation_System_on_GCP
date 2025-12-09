#!/bin/bash

# ==============================================================================
# H&M WORKSHOP - KAYNAK TEMİZLİĞİ (CLEANUP SCRIPT)
# ==============================================================================

# Renk tanımlamaları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}======================================================${NC}"
echo -e "${RED}   H&M Workshop - KAYNAK TEMİZLİĞİ (CLEANUP)          ${NC}"
echo -e "${RED}======================================================${NC}"
echo -e "${YELLOW}UYARI: Bu işlem oluşturulan tüm verileri, modelleri ve servisleri kalıcı olarak silecektir!${NC}"
echo ""

# 1. Proje Bilgilerini Al
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="hm-workshop-${PROJECT_ID}"
REGION="us-central1"

# Servis ve Template İsimleri (Setup script ile eşleşmeli)
SERVICE_BACKEND="hm-recommender-service"
SERVICE_FRONTEND="hm-streamlit-ui"
TEMPLATE_RETRIEVAL="hm-retrieval-gpu-template"
TEMPLATE_RANKING="hm-ranking-gpu-template"

echo -e "Silinecek Proje: ${BLUE}$PROJECT_ID${NC}"
echo -e "Silinecek Bucket: ${BLUE}$BUCKET_NAME${NC}"
echo ""

# 2. Onay İste
read -p "Devam etmek istiyor musunuz? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "İşlem iptal edildi."
    exit 1
fi
echo ""

# ------------------------------------------------------------------------------
# 3. Cloud Run Servislerini Sil (Backend + Frontend)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Cloud Run Servisleri siliniyor...${NC}"

# Backend Servisi
gcloud run services delete $SERVICE_BACKEND --region=$REGION --quiet 2>/dev/null
if [ $? -eq 0 ]; then echo -e "${GREEN}✔ Backend Servisi ($SERVICE_BACKEND) silindi.${NC}"; else echo -e "${YELLOW}⚠ Backend servisi bulunamadı.${NC}"; fi

# Frontend Servisi
gcloud run services delete $SERVICE_FRONTEND --region=$REGION --quiet 2>/dev/null
if [ $? -eq 0 ]; then echo -e "${GREEN}✔ Frontend Servisi ($SERVICE_FRONTEND) silindi.${NC}"; else echo -e "${YELLOW}⚠ Frontend servisi bulunamadı.${NC}"; fi


# ------------------------------------------------------------------------------
# 4. Container Registry İmajlarını Sil
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Docker İmajları siliniyor...${NC}"

delete_image() {
    local IMG_NAME="gcr.io/$PROJECT_ID/$1"
    # Tag'leri listele ve hepsini sil
    gcloud container images list-tags $IMG_NAME --format='get(digest)' 2>/dev/null | while read digest; do
        gcloud container images delete "$IMG_NAME@$digest" --force-delete-tags --quiet 2>/dev/null
    done
    # Repo'yu sil
    gcloud container images delete $IMG_NAME --force-delete-tags --quiet 2>/dev/null
    echo -e "${GREEN}  -> $1 imajları temizlendi.${NC}"
}

delete_image "hm-recommender-app" # Backend Image
delete_image "hm-streamlit-app"   # Frontend Image


# ------------------------------------------------------------------------------
# 5. Colab Runtime Template'lerini Sil (İkisi de)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Colab Runtime Template'leri siliniyor...${NC}"

delete_template() {
    local T_NAME=$1
    # Standart silme komutu
    gcloud colab runtime-templates delete $T_NAME --region=$REGION --quiet 2>/dev/null
    
    # Eğer hata verirse (bazen beta gerekebilir veya zaten silinmiştir)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✔ Template silindi: $T_NAME${NC}"
    else
        # Beta ile dene (Cloud Shell bazen beta gerektirir)
        gcloud beta colab runtime-templates delete $T_NAME --region=$REGION --quiet 2>/dev/null
        if [ $? -eq 0 ]; then
             echo -e "${GREEN}✔ Template silindi (Beta): $T_NAME${NC}"
        else
             echo -e "${YELLOW}⚠ Template bulunamadı: $T_NAME${NC}"
        fi
    fi
}

delete_template $TEMPLATE_RETRIEVAL
delete_template $TEMPLATE_RANKING


# ------------------------------------------------------------------------------
# 6. GCS Bucket'ını Sil
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Cloud Storage Bucket siliniyor...${NC}"

if gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    gsutil -m rm -r gs://$BUCKET_NAME
    echo -e "${GREEN}✔ Bucket ve tüm içeriği silindi.${NC}"
else
    echo -e "${YELLOW}⚠ Bucket bulunamadı ($BUCKET_NAME).${NC}"
fi


# ------------------------------------------------------------------------------
# 7. Yerel Dosyaları Temizle (Cloud Shell)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Yerel dosyalar temizleniyor...${NC}"

rm -rf hm_frontend 2>/dev/null
rm -f hm_two_tower_training.ipynb 2>/dev/null
rm -f hm_ranking_lightgbm_training.ipynb 2>/dev/null
rm -f setup_workshop.sh 2>/dev/null

echo -e "${GREEN}✔ Cloud Shell yerel klasörü temizlendi.${NC}"

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}   TEMİZLİK İŞLEMİ TAMAMLANDI! 🧹   ${NC}"
echo -e "${GREEN}======================================================${NC}"