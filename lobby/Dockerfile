FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir flask gunicorn
COPY main.py /app/main.py
ENV PORT=8080
CMD exec gunicorn -k sync -w 1 main:app --bind 0.0.0.0:${PORT} --access-logfile - --error-logfile -
