"""
ocr_pipeline.py — Lexios Brain One v9 (Article-Aware)
======================================================
Extraction OCR avec structuration article-aware.
"""

import os
import json
import asyncio
import logging
import hashlib
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from PIL import Image
import aiohttp

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')

try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.common.surya.schema import TaskNames
    HAS_SURYA = True
except ImportError:
    HAS_SURYA = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False

from config import settings
from core_embedder import count_tokens

try:
    from doc_detector import DocumentDetector
except ImportError:
    class DetResult:
        confidence = 0.5
        doc_type = "Inconnu"
        category = "Général"

    class DocumentDetector:
        @staticmethod
        def detect(text, filename):
            return DetResult()

import threading
_surya_lock = threading.Lock()


def safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}


log = logging.getLogger("lexios.ocr")
SUPPORTED_PDF = {".pdf"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".heic", ".heif"}


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class Chunk:
    """Chunk structuré avec article_id pour article-aware retrieval."""
    text: str
    normalized_text: str
    index: int
    chunk_id: str
    article_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "index": self.index,
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "page": self.page,
            "section": self.section,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "word_count": self.word_count
        }


@dataclass
class DriveContext:
    hierarchy: List[str]
    context_path: str
    category: str
    subcategory: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hierarchy": self.hierarchy,
            "context_path": self.context_path,
            "category": self.category,
            "subcategory": self.subcategory
        }


@dataclass
class LexiosDoc:
    uid: str
    source_file: str
    file_type: str
    ocr_engine: str
    language: str
    pages: int
    drive: DriveContext
    nature: str
    doc_type_detected: str
    doc_type_confidence: float
    is_structured: bool
    entities: Dict[str, Any]
    structured_data: Dict[str, Any]
    chunks: List[Chunk]
    raw_text: str
    processed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "ocr_engine": self.ocr_engine,
            "language": self.language,
            "pages": self.pages,
            "drive": self.drive.to_dict(),
            "nature": self.nature,
            "doc_type_detected": self.doc_type_detected,
            "doc_type_confidence": self.doc_type_confidence,
            "is_structured": self.is_structured,
            "entities": self.entities,
            "structured_data": self.structured_data,
            "chunks": [c.to_dict() for c in self.chunks],
            "raw_text": self.raw_text,
            "processed_at": self.processed_at
        }


# ============================================================================
# TEXT PROCESSING
# ============================================================================

class TextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    @staticmethod
    def detect_section(text: str) -> Optional[str]:
        patterns = [
            r'^(?:CHAPITRE|Chapitre|chapter)\s+\w+',
            r'^(?:SECTION|Section|section)\s+\w+',
            r'^(?:TITRE|Titre|title)\s+\w+',
            r'^(?:ARTICLE|Article|art\.?|المادة)\s+\d+',
            r'^[\u0600-\u06FF]{3,20}\s*$',
        ]
        for pattern in patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return "header"
        return None

    @staticmethod
    def extract_article_number(text: str) -> Optional[str]:
        m = re.search(
            r'(?:Article|Art\.?|art\.?|المادة|الفصل)\s*(?:n°|numéro|number|رقم)?\s*(\d+)',
            text,
            re.IGNORECASE
        )
        if m:
            return m.group(1)
        return None


class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.normalizer = TextNormalizer()
        assert overlap < chunk_size

    def chunk(self, text: str, uid: str, page_breaks: Optional[List[int]] = None) -> List[Chunk]:
        text_tokens = count_tokens(text)
        if text_tokens <= self.chunk_size:
            normalized = self.normalizer.normalize(text)
            return [Chunk(
                text=text,
                normalized_text=normalized,
                index=0,
                chunk_id=f"{uid}_chunk_0",
                article_id=uid,
                page=1,
                section=self.normalizer.detect_section(text),
                char_start=0,
                char_end=len(text),
                word_count=len(text.split())
            )]

        paragraphs = self._split_paragraphs_with_positions(text)
        chunks = []
        current_texts = []
        current_positions = []
        current_size = 0
        char_offset = 0
        chunk_idx = 0
        current_article_id = f"{uid}_art_0"
        article_counter = 0

        for para, para_start, para_end in paragraphs:
            para_tokens = count_tokens(para)

            section = self.normalizer.detect_section(para)
            art_num = self.normalizer.extract_article_number(para) if section == "header" else None
            if art_num:
                article_counter += 1
                current_article_id = f"{uid}_art_{hashlib.md5(art_num.encode()).hexdigest()[:8]}"

            if para_tokens > self.chunk_size:
                if current_texts:
                    chunk = self._create_chunk(
                        current_texts, current_positions, char_offset,
                        chunk_idx, uid, page_breaks, current_article_id
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    char_offset = current_positions[-1][1] if current_positions else char_offset

                sub_chunks = self._split_large_paragraph(
                    para, para_start, uid, chunk_idx, current_article_id
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                char_offset = para_end

                current_texts = []
                current_positions = []
                current_size = 0
                continue

            if current_size + para_tokens > self.chunk_size and current_texts:
                chunk = self._create_chunk(
                    current_texts, current_positions, char_offset,
                    chunk_idx, uid, page_breaks, current_article_id
                )
                chunks.append(chunk)
                chunk_idx += 1

                overlap_texts, overlap_positions = self._compute_overlap(current_texts, current_positions)
                current_texts = overlap_texts + [para]
                current_positions = overlap_positions + [(para_start, para_end)]
                current_size = sum(count_tokens(t) for t in current_texts)
                char_offset = overlap_positions[0][0] if overlap_positions else para_start
            else:
                current_texts.append(para)
                current_positions.append((para_start, para_end))
                current_size += para_tokens
                char_offset = char_offset or para_start

        if current_texts:
            chunk = self._create_chunk(
                current_texts, current_positions, char_offset,
                chunk_idx, uid, page_breaks, current_article_id
            )
            chunks.append(chunk)

        return chunks

    def _split_paragraphs_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        splits = re.split(r'(\n\s*\n|\r\n\s*\r\n)', text)
        paragraphs = []
        pos = 0
        for i, part in enumerate(splits):
            if i % 2 == 0 and part.strip():
                paragraphs.append((part.strip(), pos, pos + len(part)))
                pos += len(part)
            else:
                pos += len(part)
        return paragraphs

    def _split_large_paragraph(self, para: str, start_pos: int, uid: str,
                                start_idx: int, article_id: str) -> List[Chunk]:
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_text = ""
        current_start = start_pos
        char_pos = start_pos
        local_idx = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)
            if count_tokens(current_text) + sent_tokens > self.chunk_size and current_text:
                normalized = self.normalizer.normalize(current_text)
                chunks.append(Chunk(
                    text=current_text.strip(),
                    normalized_text=normalized,
                    index=start_idx + local_idx,
                    chunk_id=f"{uid}_chunk_{start_idx + local_idx}",
                    article_id=article_id,
                    page=None,
                    section=None,
                    char_start=current_start,
                    char_end=char_pos,
                    word_count=len(current_text.split())
                ))
                local_idx += 1

                last_sent = current_text.rsplit(". ", 1)[-1] if ". " in current_text else ""
                current_text = last_sent + " " + sent if last_sent else sent
                current_start = char_pos - len(last_sent) if last_sent else char_pos
            else:
                current_text += " " + sent if current_text else sent
            char_pos += len(sent) + 1

        if current_text.strip():
            normalized = self.normalizer.normalize(current_text)
            chunks.append(Chunk(
                text=current_text.strip(),
                normalized_text=normalized,
                index=start_idx + local_idx,
                chunk_id=f"{uid}_chunk_{start_idx + local_idx}",
                article_id=article_id,
                page=None,
                section=None,
                char_start=current_start,
                char_end=start_pos + len(para),
                word_count=len(current_text.split())
            ))
        return chunks

    def _create_chunk(self, texts: List[str], positions: List[Tuple[int, int]],
                      char_offset: int, idx: int, uid: str,
                      page_breaks: Optional[List[int]], article_id: str) -> Chunk:
        full_text = '\n\n'.join(texts)
        normalized = self.normalizer.normalize(full_text)
        page = self._detect_page(positions[0][0], page_breaks) if page_breaks else None
        section = self.normalizer.detect_section(full_text)

        return Chunk(
            text=full_text,
            normalized_text=normalized,
            index=idx,
            chunk_id=f"{uid}_chunk_{idx}",
            article_id=article_id,
            page=page,
            section=section,
            char_start=positions[0][0],
            char_end=positions[-1][1],
            word_count=len(full_text.split())
        )

    def _compute_overlap(self, texts: List[str], positions: List[Tuple[int, int]]) -> Tuple[List[str], List[Tuple[int, int]]]:
        overlap_size = 0
        overlap_texts = []
        overlap_positions = []
        for text, pos in reversed(list(zip(texts, positions))):
            if overlap_size + len(text) <= self.overlap:
                overlap_texts.insert(0, text)
                overlap_positions.insert(0, pos)
                overlap_size += len(text)
            else:
                remaining = self.overlap - overlap_size
                if remaining > 50:
                    partial = text[-remaining:]
                    overlap_texts.insert(0, partial)
                    overlap_positions.insert(0, (pos[1] - remaining, pos[1]))
                break
        return overlap_texts, overlap_positions

    def _detect_page(self, char_pos: int, page_breaks: List[int]) -> Optional[int]:
        for i, break_pos in enumerate(page_breaks):
            if char_pos < break_pos:
                return i + 1
        return len(page_breaks) + 1


# ============================================================================
# OCR ENGINE
# ============================================================================

class LegalOCR:
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(settings.OCR_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_OCR)
        self._session: Optional[aiohttp.ClientSession] = None
        self._llm_semaphore = asyncio.Semaphore(5)
        self.chunker = TextChunker(
            chunk_size=settings.OCR_CHUNK_SIZE,
            overlap=settings.OCR_CHUNK_OVERLAP
        )
        self._marker_models = None
        self._surya_models = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.LLM_TIMEOUT)
            )
        return self._session

    def _get_marker(self):
        if self._marker_models is None:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            log.info("Chargement Marker (Modern API)...")
            # Use CPU by default if GPU not detected
            device = settings.EMBED_DEVICE
            self._marker_models = PdfConverter(artifact_dict=create_model_dict(device=device))
        return self._marker_models

    def _get_surya(self):
        with _surya_lock:
            if self._surya_models is None and HAS_SURYA:
                log.info("Chargement Surya (Modern API)...")
                foundation = FoundationPredictor()
                self._surya_models = {
                    "det": DetectionPredictor(),
                    "rec": RecognitionPredictor(foundation)
                }
            return self._surya_models

    def _detect_language(self, text: str) -> str:
        if not text:
            return "fr"
        ar_count = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        ratio = ar_count / len(text) if text else 0
        return "ar" if ratio > 0.5 else ("mixed" if ratio > 0.1 else "fr")

    def _extract_topology(self, path: Path) -> DriveContext:
        try:
            parts = list(path.parts)
            pivot_idx = -1
            for i, part in enumerate(parts):
                if "lexios".lower() in part.lower():
                    pivot_idx = i
                    break
            if pivot_idx < 0:
                pivot_idx = max(0, len(parts) - 4)

            hierarchy = parts[pivot_idx + 1:-1] if pivot_idx < len(parts) - 1 else parts[:-1]
            category = "Général"
            if hierarchy:
                first = hierarchy[0].lower()
                if any(k in first for k in ["civil", "مدني"]):
                    category = "Civil"
                elif any(k in first for k in ["penal", "pénal", "جنائي"]):
                    category = "Pénal"
                elif any(k in first for k in ["admin", "إداري"]):
                    category = "Administratif"

            subcategory = hierarchy[1] if len(hierarchy) > 1 else (hierarchy[0] if hierarchy else "général")
            context_path = " > ".join(hierarchy) if hierarchy else "racine"

            return DriveContext(
                hierarchy=list(hierarchy),
                context_path=context_path,
                category=category,
                subcategory=subcategory
            )
        except Exception as e:
            log.error(f"Topology error: {e}")
            return DriveContext([], "unknown", "Général", "général")

    def _compute_uid(self, content: str, file_path: str) -> str:
        return hashlib.sha256((content[:5000] + file_path).encode()).hexdigest()[:16]

    def _extract_entities_simple(self, text: str) -> Dict[str, Any]:
        entities = {"articles": [], "dates": [], "montants": []}
        article_pattern = r'(Art(?:icle)?\.?\s*\d+|\bالمادة\s*\d+)'
        entities["articles"] = list(set(re.findall(article_pattern, text, re.IGNORECASE)))[:5]
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
        entities["dates"] = list(set(re.findall(date_pattern, text)))[:3]
        montant_pattern = r'(\d[\d\s,.]*)\s*(?:TND|DT|دينار|dinars?)'
        entities["montants"] = re.findall(montant_pattern, text, re.IGNORECASE)[:3]
        return entities

    async def _call_llm_with_retry(self, text: str, context: str, max_retries: int = 2) -> Dict[str, Any]:
        async with self._llm_semaphore:
            prompt = f"""Analyse ce texte juridique tunisien. Retourne UNIQUEMENT JSON.

Contexte: {context}
Texte: {text[:2000]}

Format: {{"classification": "...", "parties": [], "articles_cites": [], "dates_importantes": [], "resume": "", "mots_cles": []}}"""

            for attempt in range(max_retries):
                try:
                    session = await self._get_session()
                    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
                        payload = {
                            "model": settings.GROQ_MODEL,
                            "messages": [
                                {"role": "system", "content": "Expert juridique. JSON uniquement."},
                                {"role": "user", "content": prompt}
                            ],
                            "response_format": {"type": "json_object"},
                            "temperature": 0.1
                        }
                        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
                        async with session.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            json=payload,
                            headers=headers
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                content = data["choices"][0]["message"]["content"]
                                return safe_json_parse(content)
                    else:
                        payload = {
                            "model": settings.OLLAMA_MODEL,
                            "messages": [
                                {"role": "system", "content": "Expert juridique. JSON uniquement."},
                                {"role": "user", "content": prompt}
                            ],
                            "format": "json",
                            "stream": False,
                            "options": {"temperature": 0.1}
                        }
                        async with session.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                content = data["message"]["content"]
                                return safe_json_parse(content)
                except Exception as e:
                    log.warning(f"LLM attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)

            log.warning("LLM extraction returned empty after all retries")
            return {}

    # ========================================================================
    # MÉTHODE PRINCIPALE : process_file
    # ========================================================================

    async def process_file(self, path: Path) -> Optional[LexiosDoc]:
        path = Path(path)
        """Traite un fichier PDF ou image et retourne un LexiosDoc structuré."""
        if not path.exists():
            log.error(f"Fichier introuvable: {path}")
            return None

        ext = path.suffix.lower()
        text = ""
        engine = "unknown"
        pages = 1
        page_breaks = []

        # =========================================================
        # OCR PROCESSING (PDF + IMAGE)
        # =========================================================

        if ext in SUPPORTED_PDF:
            try:
                converter = self._get_marker()
                # marker-pdf 1.10.2 returns a RenderedDocument object
                rendered = converter(str(path))
                text = rendered.markdown
                
                print("TEXT OCR (PDF) =", text[:200])
                print("LEN =", len(text))

                engine = "marker"
                pages = getattr(converter, "page_count", 1)

                page_breaks = [m.start() for m in re.finditer(r"(?i)\bPage\s+\d+\b", text)]

                if len(page_breaks) < pages - 1:
                    avg_page_len = len(text) // pages if pages > 1 else len(text)
                    page_breaks = [i * avg_page_len for i in range(1, pages)]

            except Exception as e:
                log.error(f"Marker failed: {e}")

                try:
                    import pdfplumber

                    text_parts = []
                    with pdfplumber.open(path) as pdf:
                        pages = len(pdf.pages)
                        for p in pdf.pages:
                            t = p.extract_text()
                            if t:
                                text_parts.append(t)

                    text = "\n".join(text_parts)
                    engine = "pdfplumber_fallback"

                    page_breaks = [m.start() for m in re.finditer(r"(?i)\bPage\s+\d+\b", text)]

                    if len(page_breaks) < pages - 1:
                        avg_page_len = len(text) // pages if pages > 1 else len(text)
                        page_breaks = [i * avg_page_len for i in range(1, pages)]

                except Exception as pdf_e:
                    log.error(f"pdfplumber fallback failed: {pdf_e}")
                    return None

        elif ext in SUPPORTED_IMAGE:
            if not HAS_SURYA:
                log.error("Surya not installed")
                return None

            try:
                log.info(f"OCR image: {path.name}")

                # 1. Ouverture de l'image (HEIC ou JPG iPhone)
                with Image.open(path) as raw_img:
                    # Conversion en RGB
                    rgb_img = raw_img.convert("RGB")
                    
                    # TECHNIQUE ANTI-BUG : On recrée l'image à partir des pixels bruts 
                    # Cela supprime les métadonnées EXIF d'Apple qui font planter l'encodeur JPEG
                    img = Image.frombytes('RGB', rgb_img.size, rgb_img.tobytes())

                    # 2. Sauvegarde sécurisée du fichier temporaire
                    tmp_path = path.with_suffix(".tmp.jpg")
                    # On force des paramètres standards pour éviter l'erreur des 16 arguments
                    img.save(tmp_path, format="JPEG", quality=95, optimize=True)
                
                # 3. Réouverture du fichier propre pour Surya
                img = Image.open(tmp_path)
                
                # Chargement des modèles Surya
                models = self._get_surya()
                print("HAS_SURYA =", HAS_SURYA)
                print("MODELS LOADED =", bool(models))

                predictions_by_image = models["rec"](
                    [img],
                    task_names=[TaskNames.ocr_with_boxes],
                    det_predictor=models["det"]
                )

                text = "\n".join(
                    line.text
                    for p in predictions_by_image
                    for line in p.text_lines
                    if line.text.strip()
                )

                print("TEXT OCR (IMAGE) =", text[:200])
                print("LEN =", len(text))

                engine = "surya"
                pages = 1

            except Exception as e:
                log.error(f"Surya failed on {path.name}: {e}", exc_info=True)
                return None

        else:
            log.error(f"Format non supporté: {ext}")
            return None

        # =========================================================
        # POST-TRAITEMENT & STRUCTURATION
        # =========================================================

        if not text or not text.strip():
            log.warning(f"Texte vide après OCR: {path.name}")
            return None

        drive = self._extract_topology(path)
        lang = self._detect_language(text)
        det = DocumentDetector.detect(text, path.name)
        uid = self._compute_uid(text, str(path))

        chunks = self.chunker.chunk(text, uid, page_breaks)
        entities = self._extract_entities_simple(text)
        structured = await self._call_llm_with_retry(text, drive.context_path)

        nature = structured.get("classification", "Inconnu")
        if nature == "Inconnu" and det.confidence >= 0.6:
            nature = det.doc_type

        doc = LexiosDoc(
            uid=uid,
            source_file=str(path),
            file_type=ext.lstrip("."),
            ocr_engine=engine,
            language=lang,
            pages=pages,
            drive=drive,
            nature=nature,
            doc_type_detected=det.doc_type,
            doc_type_confidence=det.confidence,
            is_structured=bool(structured),
            entities=entities,
            structured_data=structured,
            chunks=chunks,
            raw_text=text,
            processed_at=datetime.now().isoformat()
        )

        await self._write_output(doc, path)
        return doc

    async def _write_output(self, doc: LexiosDoc, path: Path):
        json_path = self.output_dir / f"{path.stem}_{doc.uid}.json"
        json_path.write_text(
            json.dumps(doc.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    async def process_folder(self, input_dir: str) -> List[LexiosDoc]:
        root = Path(input_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dossier introuvable: {root}")

        files = []
        for ext in (SUPPORTED_PDF | SUPPORTED_IMAGE):
            files.extend(root.rglob(f"*{ext}"))
        files = [f for f in files if not f.name.startswith(".")]

        log.info(f"OCR démarré: {len(files)} fichiers")
        sem = asyncio.Semaphore(10)

        async def process_with_sem(f):
            async with sem:
                return await self.process_file(f)

        results = await asyncio.gather(*[process_with_sem(f) for f in files], return_exceptions=True)
        docs = [r for r in results if isinstance(r, LexiosDoc)]
        log.info(f"OCR terminé: {len(docs)}/{len(files)} succès")
        return docs

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ============================================================================
# FONCTION UTILITAIRE RACINE
# ============================================================================

async def process_lexios_drive(output_dir: str = None) -> List[LexiosDoc]:
    LEXIOS_PATH = r"G:\Mon Drive\lexiOS"
    if not Path(LEXIOS_PATH).exists():
        raise FileNotFoundError(f"Dossier lexiOS non trouvé: {LEXIOS_PATH}")
    ocr = LegalOCR(output_dir=output_dir)
    try:
        return await ocr.process_folder(LEXIOS_PATH)
    finally:
        await ocr.close()