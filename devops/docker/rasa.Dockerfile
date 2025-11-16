FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install required system deps
RUN apt-get update && \
    apt-get install -y gcc g++ make && \
    apt-get clean

# Copy requirements
COPY requirements_rasa.txt /app/requirements.txt

# Install Rasa dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full Rasa project
COPY . /app

EXPOSE 5005

CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"]

