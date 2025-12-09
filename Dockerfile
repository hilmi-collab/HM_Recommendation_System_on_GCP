FROM python:3.9-slim

# Install system dependencies if needed (none strictly needed for pure python libs used)
RUN apt-get update && apt-get clean

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY streamlit_app.py .

# Streamlit runs on port 8501 by default
EXPOSE 8080

# Configure Streamlit to run on port 8080 (Cloud Run requirement)
# and address 0.0.0.0
CMD streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0