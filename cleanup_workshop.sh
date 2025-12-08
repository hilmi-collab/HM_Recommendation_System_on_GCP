#!/bin/bash

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

# 3. Cloud Run Servisini Sil
echo -e "${BLUE}[1/4] Cloud Run Servisi siliniyor...${NC}"
gcloud run services delete hm-recommender-service \
    --region=$REGION \
    --quiet 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✔ Servis silindi.${NC}"
else
    echo -e "${YELLOW}⚠ Servis bulunamadı veya zaten silinmiş.${NC}"
fi

# 4. Container Registry İmajını Sil
echo -e "${BLUE}[2/4] Docker İmajı siliniyor...${NC}"
IMAGE_NAME="gcr.io/$PROJECT_ID/hm-recommender-app"

# İmajın digest'larını bul ve sil (Tag'li ve tag'siz tüm versiyonlar)
gcloud container images list-tags $IMAGE_NAME --format='get(digest)' | while read digest; do
    gcloud container images delete "$IMAGE_NAME@$digest" --force-delete-tags --quiet 2>/dev/null
done
# Son olarak repository'i temizle
gcloud container images delete $IMAGE_NAME --force-delete-tags --quiet 2>/dev/null

echo -e "${GREEN}✔ Docker imajları temizlendi.${NC}"


# 5. Colab Runtime Template'ini Sil
echo -e "${BLUE}[3/4] Colab Runtime Template siliniyor...${NC}"
TEMPLATE_NAME="hm-workshop-gpu-template"

gcloud ai runtime-templates delete $TEMPLATE_NAME \
    --region=$REGION \
    --quiet 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✔ Runtime Template silindi.${NC}"
else
    echo -e "${YELLOW}⚠ Template bulunamadı veya zaten silinmiş.${NC}"
fi


# 6. GCS Bucket'ını Sil (İçindekilerle Birlikte)
echo -e "${BLUE}[4/4] Cloud Storage Bucket siliniyor...${NC}"

if gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    gsutil -m rm -r gs://$BUCKET_NAME
    echo -e "${GREEN}✔ Bucket ve tüm içeriği silindi.${NC}"
else
    echo -e "${YELLOW}⚠ Bucket bulunamadı ($BUCKET_NAME).${NC}"
fi

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}   TEMİZLİK İŞLEMİ TAMAMLANDI! 🧹   ${NC}"
echo -e "${GREEN}======================================================${NC}"