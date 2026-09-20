# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Unbuffered stdout so `docker logs` shows output immediately.
# HOST/PORT default to container-friendly values; override at runtime if needed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes,
# so editing agent.py does not reinstall scikit-learn.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Application source. examples/ must be included — it is the seed source that
# agent.py copies into data/ on first start.
COPY agent.py healthcheck.py ./
COPY infrastructure/ ./infrastructure/
COPY ingestion/ ./ingestion/
COPY review/ ./review/
COPY examples/ ./examples/
COPY index.html app.js styles.css ./

# Runtime state (opportunities + knowledge). Mount a volume here to persist
# across container restarts — otherwise every `docker run` starts from the
# examples/ seed.
VOLUME ["/app/data"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "healthcheck.py"]

CMD ["python", "agent.py", "serve"]
