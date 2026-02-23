# ── Build stage: install dependencies ─────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install Poetry + export plugin (used only in this stage to resolve/install deps)
RUN pip install --no-cache-dir poetry poetry-plugin-export

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml poetry.lock ./

# Export dependencies to a plain requirements.txt so the final stage
# doesn't need Poetry at runtime.
RUN poetry export --without-hashes --format=requirements.txt -o /tmp/requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Install runtime Python dependencies
COPY --from=builder /tmp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY app/  ./app/
COPY ui/   ./ui/
COPY run.py ./

# Create mount-point directories so Docker volumes attach cleanly
RUN mkdir -p /app/chroma_db /app/data

# Expose both service ports (the actual port used depends on CMD)
EXPOSE 8000 8501
