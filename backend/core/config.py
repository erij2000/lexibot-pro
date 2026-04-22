from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    # --- App ---
    PROJECT_NAME: str = "Lexibot Pro API"
    API_V1_STR: str = "/api/v1"

    # --- Security & DB ---
    # SECRET: On met une valeur par défaut pour le développement, 
    # mais elle sera écrasée par le .env en prod.
    SECRET: str = Field(default="dev_secret_key_change_me", validation_alias="SECRET") 
    
    # DATABASE_URL : 
    # TRÈS IMPORTANT : Dans Docker, on utilise le nom du SERVICE (db) et non 127.0.0.1
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://erij_user:super_secret_erij_mdp@db:5432/lexibot_db",
        validation_alias="DATABASE_URL"
    )

    # --- Ollama ---
    # Sur Windows avec Docker, on utilise host.docker.internal pour sortir du container vers l'hôte
    OLLAMA_API_BASE_URL: str = os.getenv("OLLAMA_API_BASE_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:latest")

    # --- Rasa ---
    # Si Rasa est dans un container nommé 'rasa', on utilise son nom
    RASA_API_BASE_URL: str = os.getenv(
        "RASA_API_BASE_URL", "http://rasa:5005/webhooks/rest/webhook"
    )
    RASA_TIMEOUT: int = int(os.getenv("RASA_TIMEOUT", 10))

    # --- Admin ---
    SUPER_USER_EMAIL: str = os.getenv("SUPER_USER_EMAIL", "admin@lexibot.com")
    SUPER_USER_PASSWORD: str = os.getenv("SUPER_USER_PASSWORD", "SuperSecure123")

    # --- Configuration ---
    # Pydantic Settings chargera automatiquement le fichier .env s'il existe
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

settings = Settings()

# Petit hack pour cleaner les logs au démarrage
if os.getenv("DEBUG_MODE") == "true":
    print(f"🔍 DATABASE_URL utilisée: {settings.DATABASE_URL}")