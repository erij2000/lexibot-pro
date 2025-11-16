# Backend - FastAPI
FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y gcc libpq-dev && \
    apt-get clean

# Copy dependencies
COPY requirements_fastapi.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app

EXPOSE 8000

# Uvicorn server
CMD ["uvicorn", "lexibot_api_final:app", "--host", "0.0.0.0", "--port", "8000"]
