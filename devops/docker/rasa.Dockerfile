# rasa.Dockerfile
FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Installer gcc et make
RUN apt-get update && \
    apt-get install -y gcc g++ make && \
    apt-get clean

# Copier requirements depuis la racine
COPY ../../requirements_rasa.txt ./requirements.txt

# Installer Rasa
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code Rasa
COPY ../rasa /app

EXPOSE 5005

CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"]