FROM python:3.9-slim

# Temizlik ve Kurulum
RUN apt-get update && apt-get clean

WORKDIR /app

# Önce gereksinimleri kopyala ve kur (Cache avantajı için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- KRİTİK DÜZELTME BURADA ---
# Sadece .py dosyasını değil, klasördeki TÜM dosyaları (.parquet'ler dahil) kopyalıyoruz.
COPY . .

EXPOSE 8080

CMD streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0