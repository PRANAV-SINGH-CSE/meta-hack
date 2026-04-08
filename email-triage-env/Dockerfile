# ── Email Triage OpenEnv — Dockerfile ─────────────────────────────────────────
# Compatible with Hugging Face Spaces (port 7860)
# Build : docker build -t email-triage-env .
# Run   : docker run -p 7860:7860 email-triage-env

FROM python:3.11-slim

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

# Create __init__ files so Python treats dirs as packages
RUN touch env/__init__.py data/__init__.py

USER appuser

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
