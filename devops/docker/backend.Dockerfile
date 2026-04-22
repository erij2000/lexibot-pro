# backend.Dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Packages système pour compiler asyncpg si besoin
RUN apt-get update && \
    apt-get install -y gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# PUISQUE LE FICHIER EST À LA RACINE DU PROJET (le context ..) :
# On le copie directement depuis la racine vers l'image
COPY requirements_fastapi.txt ./requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code du dossier backend vers le dossier /app du conteneur
COPY backend/ .

EXPOSE 8000

# Au lieu de "main:app", on utilise le chemin du module "api.main:app"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]