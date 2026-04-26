import os
import sys
import psutil
import torch
from pprint import pprint

# On s'assure que le moteur est dans le path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from lexios_engine import config
    print("✅ Importation de config.py réussie.")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    sys.exit(1)

def check_system_resources():
    print("\n=== ÉTAT DU SYSTÈME (MSI) ===")
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    
    print(f"RAM Totale détectée par l'OS : {total_gb:.2f} GB")
    print(f"RAM Disponible actuellement : {available_gb:.2f} GB")
    
    if total_gb < 20:
        print("⚠️ Attention : L'OS ne semble pas voir tes 24 GB de RAM.")
    else:
        print("🚀 Les 24 GB de RAM sont bien reconnus.")

def validate_settings_logic():
    print("\n=== VÉRIFICATION DES PARAMÈTRES LEXIBOT ===")
    s = config.settings
    
    # Test de la logique de Batch Size (Santé du CPU/GPU)
    print(f"Device utilisé : {s.EMBED_DEVICE}")
    print(f"Batch Size configuré : {s.EMBED_BATCH_SIZE}")
    
    # Test de la taille du cache
    print(f"Capacité du Cache : {s.CACHE_MAX_SIZE} entrées")
    if s.CACHE_MAX_SIZE >= 2000:
        print("✅ Cache optimisé pour 24GB.")
    
    # Test de la fenêtre de contexte BGE-M3
    print(f"Fenêtre BGE-M3 : {s.EMBED_MAX_LENGTH} tokens")
    
    # Vérification GPU (VRAM vs RAM)
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM GPU dédiée : {vram:.2f} GB")
    else:
        print("ℹ️ Mode CPU actif (Le projet utilisera tes 24GB de RAM système).")

if __name__ == "__main__":
    check_system_resources()
    validate_settings_logic()
    
    print("\n=== RÉSULTAT FINAL ===")
    errors = config.validate_config()
    if not errors:
        print("✨ Ta configuration est SAINE et prête pour le développement.")
    else:
        print("❌ Des erreurs ont été détectées dans la configuration.")