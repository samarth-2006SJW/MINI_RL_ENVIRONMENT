# ============================================================
# Stage 1: Build Frontend (Node.js)
# ============================================================
FROM node:18-slim AS builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ .
RUN npm run build

# ============================================================
# Stage 2: Build Backend & Server (Python)
# Uses Python 3.10 slim for a small, production-ready image
# ============================================================
FROM python:3.10-slim

# Set environment variables to prevent .pyc files and enable log flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Hugging Face Spaces runs as a non-root user (UID 1000)
# Create the user upfront to match HF's runtime expectations
RUN useradd -m -u 1000 appuser

# Set the working directory inside the container
WORKDIR /app
ENV PYTHONPATH=/app

# ============================================================
# Stage 3: Dependency Installation
# Copy requirements first for Docker layer caching efficiency
# ============================================================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 4: Copy Application Code & Frontend Build
# ============================================================
COPY --chown=appuser:appuser . .
# Overwrite the dist directory with the one built in Stage 1
COPY --from=builder --chown=appuser:appuser /app/frontend/dist ./frontend/dist

# ============================================================
# Stage 5: Runtime Configuration
# Hugging Face Spaces REQUIRES port 7860
# OPENAI_API_KEY must be set as a Space Secret (never hardcode)
# ============================================================
EXPOSE 7860

# Switch to non-root user for security
USER appuser

# Start the FastAPI server via Uvicorn on the required port
CMD ["python", "backend/api/app.py"]
