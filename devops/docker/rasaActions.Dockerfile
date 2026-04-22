FROM rasa/rasa-sdk:latest

USER root

# Installation des dépendances
RUN pip install --no-cache-dir langdetect pyspellchecker

# Copie des actions
COPY rasa/actions /app/actions

# Configuration du PATH
ENV PYTHONPATH="/app:/app/actions"

USER 1001

EXPOSE 5055

# FIX: Use python -m rasa_sdk instead of rasa command
CMD ["python", "-m", "rasa_sdk", "--actions", "actions"]