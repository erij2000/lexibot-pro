# backend/core/logging_config.py - Configuration du logging
import logging
import sys
import os # Ajout de l'import os qui manquait pour le path
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Créer le dossier logs s'il n'existe pas
# Utilisation d'un chemin absolu plus sûr, en supposant que le fichier est dans backend/core/
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configuration des formats
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configuration du logger principal
def setup_logging(
    log_level: str = "INFO",
    log_file: str = "lexibot.log"
):
    """Configure le système de logging"""
    
    # Niveau de log
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # S'assurer que les handlers ne sont pas ajoutés plusieurs fois
    if not root_logger.handlers:
        
        # Handler console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            fmt="%(levelname)s:     %(message)s",
            datefmt=DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        
        # Handler fichier avec rotation
        file_handler = RotatingFileHandler(
            LOGS_DIR / log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        
        # Ajouter les handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    # Désactiver les logs verbeux de certaines bibliothèques
    # On force le niveau WARNING pour uvicorn.access pour éviter trop de bruit
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING) 
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING) 
    
    return root_logger

# Logger spécifique pour Ollama
def get_ollama_logger():
    """Logger pour les requêtes Ollama"""
    logger = logging.getLogger("ollama")
    
    # S'assurer que l'handler est ajouté une seule fois
    if not any(isinstance(h, RotatingFileHandler) and 'ollama' in h.baseFilename for h in logger.handlers):
        # Handler fichier spécifique
        ollama_handler = RotatingFileHandler(
            LOGS_DIR / "ollama_logs.txt",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        ollama_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(ollama_handler)
    
    return logger

# Logger spécifique pour l'authentification
def get_auth_logger():
    """Logger pour l'authentification"""
    logger = logging.getLogger("auth")
    
    # S'assurer que l'handler est ajouté une seule fois
    if not any(isinstance(h, RotatingFileHandler) and 'auth' in h.baseFilename for h in logger.handlers):
        # Handler fichier spécifique
        auth_handler = RotatingFileHandler(
            LOGS_DIR / "auth.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8"
        )
        auth_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(auth_handler)
    
    return logger

# Initialisation par défaut
# Note: Nous pourrions vouloir appeler cette fonction dans main.py pour
# avoir plus de contrôle sur le moment où le logging commence.
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
ollama_logger = get_ollama_logger()
auth_logger = get_auth_logger()