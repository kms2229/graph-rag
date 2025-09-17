FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in the correct order
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy>=1.26.0 pandas>=2.2.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
