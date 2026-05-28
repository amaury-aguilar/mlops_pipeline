FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY src/model_deploy.py ./src/model_deploy.py
COPY src/ft_engineering.py ./src/ft_engineering.py
COPY model_artifacts ./model_artifacts

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]