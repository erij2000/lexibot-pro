FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ make && \
    apt-get clean

# Copy requirements
COPY ../requirements_rasa.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy actions folder only
COPY . /app/actions

EXPOSE 5055

CMD ["rasa", "run", "actions"]
