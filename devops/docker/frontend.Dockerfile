FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer gcc
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean

# Installer Streamlit et requests
RUN pip install --no-cache-dir streamlit==1.40.0 requests

# Copier le code frontend (relative to context: ..)
COPY frontend /app

EXPOSE 8501

RUN pip install langdetect
CMD ["streamlit", "run", "lexibot_app.py", "--server.address=0.0.0.0"]