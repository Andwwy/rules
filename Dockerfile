FROM python:3.12-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (the .db files and .env are bind-mounted at runtime, see compose)
COPY annotate.py .

# Listen on all interfaces inside the container; published to the host by compose.
ENV HOST=0.0.0.0 \
    PORT=5002

EXPOSE 5002

CMD ["python", "annotate.py"]
