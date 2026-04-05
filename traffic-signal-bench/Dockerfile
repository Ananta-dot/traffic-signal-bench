FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import requests; requests.get('http://localhost:7860/')" || exit 1

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
