# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVENTLET_NO_GREENDNS=1

# System deps (clean build; wheels cover most libs like orjson/eventlet)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# If you have a requirements.txt, use it; else pip install from pyproject
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Your code
COPY . /app

# Cloud Run provides $PORT; we’ll bind gunicorn to it
ENV APP_PATH="wsgi:app" \
    MODE="single" \
    SOCKIO_FORCE_BASE=1 \
    SOCKIO_COMPRESS=0 \
    FLASK_SKIP_DOTENV=1

# Small, Cloud Run–friendly entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
