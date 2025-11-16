from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

# --- 1. DÉFINITION DES RÉGLAGES D'APPLICATION ---

class Settings(BaseSettings):
    # Paramètres de l'application
    PROJECT_NAME: str = "Lexibot Pro API"
    API_V1_STR: str = "/api/v1"
    
    # Sécurité et Base de Données
    SECRET: str = Field(..., env="SECRET") # Clé secrète FastAPI-Users/JWT
    
    # Utilisez directement la variable d'environnement pour l'URL de la base de données
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://erij_user:super_secret_erij_mdp@localhost/lexibot_db",
        env="DATABASE_URL"
    )
    
    # Paramètres Ollama
    OLLAMA_API_BASE_URL: str = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    
    # Paramètres Rasa (INCLUS ICI)
    RASA_API_BASE_URL: str = os.getenv("RASA_API_BASE_URL", "http://localhost:5005/webhooks/rest/webhook") # Endpoint standard pour l'interaction
    RASA_TIMEOUT: int = int(os.getenv("RASA_TIMEOUT", 10)) # Timeout pour les requêtes Rasa

    # Paramètres Admin
    SUPER_USER_EMAIL: str = os.getenv("SUPER_USER_EMAIL", "admin@lexibot.com")
    SUPER_USER_PASSWORD: str = os.getenv("SUPER_USER_PASSWORD", "SuperSecure123")

    # Configuration des variables d'environnement (lit le fichier .env)
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()