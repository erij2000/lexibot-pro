import os
import sys
import platform

# Windows console unicode fix
if platform.system() == "Windows":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter

# Add engine to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

TEST_DATA_DIR = Path("backend/data/test_inputs")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_test_image(text, filename, size=(800, 1000), bg_color="white", text_color="black"):
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    # Default font is usually tiny, but for OCR testing it's okay or we can try to find a system font
    draw.text((50, 50), text, fill=text_color)
    img_path = TEST_DATA_DIR / filename
    img.save(img_path)
    print(f"Generated image: {img_path}")
    return img_path

def create_test_pdf_from_images(image_paths, pdf_filename):
    writer = PdfWriter()
    for img_path in image_paths:
        # Pypdf doesn't directly convert image to pdf page easily without helper
        # Actually, pypdf is for merging/splitting. 
        # Since I have pillow, I can save as PDF directly.
        pass
    
    pdf_path = TEST_DATA_DIR / pdf_filename
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    if imgs:
        imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
    print(f"Generated PDF: {pdf_path}")
    return pdf_path

def generate_all():
    # 1. Clean Text Image (to be wrapped in PDF)
    txt_clean = """
    ARTICLE 1: DEFINITIONS
    The 'Service' refers to the Lexibot Professional Suite.
    'User' refers to the legal professional using the system.

    ARTICLE 2: SCOPE
    This agreement covers the use of OCR and RAG technologies for legal research.
    """
    img_clean = create_test_image(txt_clean, "clean_doc.png")
    create_test_pdf_from_images([img_clean], "test_digital_like.pdf")

    # 2. Scanned Style (Noisy)
    txt_scanned = "ARTICLE 3: LIMITATION OF LIABILITY\nThe provider is not liable for hallucinations."
    img_scanned = create_test_image(txt_scanned, "scanned_doc.png", bg_color="lightgray")
    create_test_pdf_from_images([img_scanned], "test_scanned.pdf")

    # 3. Multilingual
    # Note: Arabic might need a specific font to render in PIL, 
    # but for "surya" testing, even if it's just characters it might work.
    # If I can't render Arabic easily, I'll just use French for now.
    txt_multi = "ARTICLE 4: DISPUTE RESOLUTION\nLe présent contrat est régi par le droit français.\nالقانون واجب التطبيق هو القانون التونسي."
    create_test_image(txt_multi, "test_multilingual.png")

    # 4. Corrupted file (just a random text file renamed)
    with open(TEST_DATA_DIR / "corrupted.pdf", "w") as f:
        f.write("This is not a real PDF.")

    print("\n✅ All test data generated in:", TEST_DATA_DIR.absolute())

if __name__ == "__main__":
    generate_all()
