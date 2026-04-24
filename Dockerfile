FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY agent2.py .

# ChromaDB-Verzeichnis erstellen
RUN mkdir -p /data/chroma_db

# Port
EXPOSE 8000

# Umgebungsvariablen (werden von Railway überschrieben)
ENV CHROMA_PATH=/data/chroma_db
ENV PORT=8000

# Start
CMD ["python3", "-m", "uvicorn", "agent2:app", "--host", "0.0.0.0", "--port", "8000"]
