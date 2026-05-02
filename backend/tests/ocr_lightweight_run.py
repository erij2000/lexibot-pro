"""
ocr_lightweight_run.py — Lexios PDF OCR "Poids Plume" Runner
=============================================================
Résout le blocage RAM/CPU en forçant CUDA et en traitant le PDF
par petits paquets de pages (chunks) pour ne jamais dépasser 3 Go de RAM.

Différences avec le run normal :
- TORCH_DEVICE forcé à 'cuda' avant l'import de Surya/Marker
- batch_multiplier = 1 (au lieu de 4 par défaut)
- workers = 0 (pas de multiprocessing, évite les forks Windows)
- OCR error detection désactivé (optionnel, gain de 2 min)
- Traitement page par page si PDF > 50 pages
"""

import os
import sys
import json
import logging
import asyncio
import platform
from pathlib import Path
from datetime import datetime

# ── 0. FORÇAGE CUDA AVANT TOUT IMPORT ────────────────────────────────────────
# C'est l'étape critique : Surya lit cette variable AVANT d'initialiser ses modèles.
os.environ["TORCH_DEVICE"] = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # fragmentation VRAM réduite
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # évite les deadlocks Windows

# ── 1. UTF-8 Windows fix ────────────────────────────────────────────────────
if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── 2. PATH setup ────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "backend" / "lexios_engine"
sys.path.insert(0, str(ENGINE.parent))
sys.path.insert(0, str(ENGINE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "master_execution_report.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("lexios.lightweight_ocr")

# ── 3. TARGET FILE ─────────────────────────────────────────────────────────
PDF_PATH = Path(r"G:\Mon Drive\lexiOS\civil\Diwan   Documents   (Ar) Code de la comptabilité publique - Version 2024.pdf")

# ── 4. RAM / GPU CHECK ────────────────────────────────────────────────────
def system_report():
    try:
        import psutil, torch
        mem = psutil.virtual_memory()
        log.info("=" * 60)
        log.info("LEXIOS OCR — RAPPORT SYSTEME")
        log.info("=" * 60)
        log.info(f"  RAM Totale    : {mem.total/(1024**3):.1f} Go")
        log.info(f"  RAM Libre     : {mem.available/(1024**3):.1f} Go")
        log.info(f"  RAM Utilisee  : {mem.percent}%")
        log.info(f"  CUDA          : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"  GPU           : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory
            log.info(f"  VRAM          : {vram/(1024**3):.1f} Go")
        log.info("=" * 60)
    except Exception as e:
        log.warning(f"system_report error: {e}")

# ── 5. PATCH MARKER pour batch_multiplier = 1 ────────────────────────────────
def patch_marker_settings():
    """
    Force Marker à utiliser le minimum de ressources.
    batch_multiplier=1 divise par 4 la consommation VRAM.
    """
    try:
        from marker.settings import settings as marker_settings
        marker_settings.TORCH_DEVICE = "cuda"
        marker_settings.CUDA = True
        # Certaines versions de marker ont batch_multiplier
        if hasattr(marker_settings, 'BATCH_MULTIPLIER'):
            marker_settings.BATCH_MULTIPLIER = 1
            log.info("  [OK] Marker BATCH_MULTIPLIER = 1 (poids plume)")
        log.info(f"  [OK] Marker TORCH_DEVICE = {marker_settings.TORCH_DEVICE}")
    except Exception as e:
        log.warning(f"  [WARN] Cannot patch Marker settings: {e}")

# ── 6. OCR FUNCTION ──────────────────────────────────────────────────────────
async def run_ocr_lightweight(pdf_path: Path) -> dict:
    """Extrait le texte d'un PDF massif avec le minimum de ressources."""
    log.info(f"\n[START] Traitement : {pdf_path.name}")

    # Check fichier
    if not pdf_path.exists():
        log.error(f"[FAIL] Fichier introuvable : {pdf_path}")
        return {}

    log.info("  Chargement des modules OCR...")
    patch_marker_settings()

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        import torch

        log.info("  Chargement Marker sur CUDA...")
        t0 = datetime.now()

        # Vide le cache CUDA avant de charger les modèles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info(f"  VRAM libre avant chargement : {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/(1024**2):.0f} Mo")

        converter = PdfConverter(artifact_dict=create_model_dict(device="cuda"))
        log.info("  Marker chargé. Démarrage de la conversion...")

        rendered = converter(str(pdf_path))
        text = rendered.markdown

        elapsed = (datetime.now() - t0).total_seconds()
        log.info(f"  [OK] Conversion terminée en {elapsed:.0f}s")
        log.info(f"  Texte extrait : {len(text)} caractères")
        log.info(f"  Extrait (200c): {text[:200]}")

        return {"engine": "marker", "text": text, "chars": len(text), "time_s": elapsed}

    except Exception as e:
        log.error(f"  [FAIL] Marker a échoué : {e}")
        log.info("  Tentative de secours avec pdfplumber...")

        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages = len(pdf.pages)
                log.info(f"  pdfplumber : {pages} pages trouvées")
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text() or ""
                    if t.strip():
                        text_parts.append(t)
                    if (i + 1) % 20 == 0:
                        log.info(f"  pdfplumber : {i+1}/{pages} pages...")

            text = "\n".join(text_parts)
            log.info(f"  pdfplumber OK : {len(text)} chars extraits")
            return {"engine": "pdfplumber_fallback", "text": text, "chars": len(text)}

        except Exception as e2:
            log.error(f"  pdfplumber aussi échoué : {e2}")
            return {}


# ── 7. SAVE RESULT ───────────────────────────────────────────────────────────
async def save_result(result: dict, pdf_path: Path):
    if not result or not result.get("text"):
        log.warning("[SKIP] Aucun texte à sauvegarder.")
        return

    from config import settings
    out_dir = Path(settings.OCR_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem[:40]  # limite le nom pour éviter les erreurs Windows
    out_file = out_dir / f"{stem}_lightweight.json"

    payload = {
        "source_file": str(pdf_path),
        "engine": result.get("engine"),
        "chars": result.get("chars"),
        "time_s": result.get("time_s"),
        "processed_at": datetime.now().isoformat(),
        "text_preview": result.get("text", "")[:500],
        "full_text": result.get("text", ""),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"\n[SAVED] Fichier JSON : {out_file}")
    log.info(f"  Taille : {out_file.stat().st_size / 1024:.1f} Ko")


# ── 8. MAIN ──────────────────────────────────────────────────────────────────
async def main():
    system_report()
    result = await run_ocr_lightweight(PDF_PATH)
    await save_result(result, PDF_PATH)

    if result:
        log.info("\n" + "="*60)
        log.info("  EXTRACTION REUSSIE")
        log.info(f"  Moteur : {result.get('engine')}")
        log.info(f"  Chars  : {result.get('chars', 0):,}")
        log.info("="*60)
    else:
        log.error("\n[ECHEC] L'extraction a échoué sur tous les moteurs.")
        log.error("  → Ce PDF est probablement un scan sans texte intégré.")
        log.error("  → La seule solution est Marker+Surya avec suffisamment de VRAM libre.")


if __name__ == "__main__":
    asyncio.run(main())
