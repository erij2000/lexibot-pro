# test_lecture_seule.py — Aucun modèle IA, aucun risque
from pathlib import Path

LEXIOS_PATH = Path(r"G:\Mon Drive\lexiOS")
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".heic", ".heif"}

def scanner():
    if not LEXIOS_PATH.exists():
        print("❌ Dossier inaccessible. Drive Desktop n'est pas prêt.")
        return
    
    fichiers = []
    for ext in EXTENSIONS:
        fichiers.extend(LEXIOS_PATH.rglob(f"*{ext}"))
    
    print(f"✅ Dossier accessible : {LEXIOS_PATH}")
    print(f"📁 {len(fichiers)} fichiers trouvés :\n")
    
    for f in fichiers[:20]:  # Affiche les 20 premiers seulement
        profondeur = len(f.relative_to(LEXIOS_PATH).parts)
        indent = "  " * (profondeur - 1)
        print(f"{indent}└── {f.name} ({f.stat().st_size / 1024:.1f} Ko)")
    
    if len(fichiers) > 20:
        print(f"\n... et {len(fichiers) - 20} autres fichiers")

if __name__ == "__main__":
    scanner()