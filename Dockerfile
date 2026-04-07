# ============================================================
# Stage 1: Build Stage
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

# ============================================================
# Stage 2: Dependency Installation
# Copy requirements first for Docker layer caching efficiency
# ============================================================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 3: Copy Application Code
# ============================================================
COPY --chown=appuser:appuser . .

# ============================================================
# Stage 4: Runtime Configuration
# Hugging Face Spaces REQUIRES port 7860
# OPENAI_API_KEY must be set as a Space Secret (never hardcode)
# ============================================================
EXPOSE 7860

# Switch to non-root user for security
USER appuser

# Start the Gradio dashboard on the required port
CMD ["python", "app.py"]
