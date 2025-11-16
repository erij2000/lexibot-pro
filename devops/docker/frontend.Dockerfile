FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean

# Install Streamlit
RUN pip install --no-cache-dir streamlit==1.40.0 requests

# Copy frontend code
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "lexibot_app.py", "--server.address=0.0.0.0"]

