# games/spoiled_broth/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVENTLET_NO_GREENDNS=1 \
    PORT=8080 \
    FLASK_SKIP_DOTENV=1 \
    SOCKIO_FORCE_BASE=1 \
    SOCKIO_COMPRESS=0

ARG REQ_FLAVOR=none

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
RUN python -m pip install -U pip

# Engine wheel staged by deploy script
COPY wheels/ /tmp/wheels/

# Install only the newest cooked-engine wheel
RUN LATEST_WHEEL="$(ls -t /tmp/wheels/cooked_engine-*.whl | head -n 1)" && \
    echo "Installing wheel: $LATEST_WHEEL" && \
    python -m pip install --no-cache-dir "$LATEST_WHEEL"

WORKDIR /app
# So Python can import /app/games/<pkg>
ENV PYTHONPATH="/app/games:/app"

# Base requirements (always)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Optional extra requirements (place files under games/spoiled_broth/requirements/*.txt)
COPY requirements/ /app/requirements/
RUN if [ "${REQ_FLAVOR}" != "none" ] && [ -f "/app/requirements/${REQ_FLAVOR}.txt" ]; then \
      echo "Installing extra requirements from /app/requirements/${REQ_FLAVOR}.txt"; \
      python -m pip install --no-cache-dir -r "/app/requirements/${REQ_FLAVOR}.txt"; \
    else \
      echo "No extra requirements to install (REQ_FLAVOR=${REQ_FLAVOR})."; \
    fi

# Web server + async worker
RUN python -m pip install --no-cache-dir gunicorn eventlet

# Copy game package into /app/games/spoiled_broth (preserve package name)
COPY . /app/games/spoiled_broth/

# Start: module path is spoiled_broth.wsgi:app
CMD ["bash","-lc","exec gunicorn -k eventlet -w 1 -b :${PORT} spoiled_broth.wsgi:app"]
