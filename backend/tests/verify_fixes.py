
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add engine to path
ENGINE_PATH = Path("backend/lexios_engine").resolve()
sys.path.append(str(ENGINE_PATH.parent))
sys.path.append(str(ENGINE_PATH))

class TestLexiosFixes(unittest.TestCase):
    def test_config_unicode_fix(self):
        """Verify that config.py now handles stdout reconfiguration without crashing."""
        import config
        # If we can import it without UnicodeEncodeError, it's a good sign
        self.assertTrue(hasattr(config, 'settings'))

    @patch('marker.converters.pdf.PdfConverter')
    @patch('marker.models.create_model_dict')
    @patch('surya.recognition.RecognitionPredictor')
    @patch('surya.detection.DetectionPredictor')
    @patch('surya.foundation.FoundationPredictor')
    def test_ocr_pipeline_refactor(self, mock_found, mock_det, mock_rec, mock_marker_dict, mock_pdf_conv):
        """Verify that the refactored LegalOCR uses the correct modern APIs."""
        # Setup mocks
        mock_pdf_instance = MagicMock()
        mock_pdf_conv.return_value = mock_pdf_instance
        mock_pdf_instance.return_value.markdown = "# Test Content"
        mock_pdf_instance.page_count = 1
        
        from ocr_pipeline import LegalOCR
        ocr = LegalOCR()
        
        # Test PDF processing (mocked)
        with patch('pathlib.Path.exists', return_value=True):
            # We use a dummy string because we mocked the converter
            import asyncio
            
            async def run_test():
                # Mocking the actual file reading part if needed, but PdfConverter is mocked
                doc = await ocr.process_file(Path("test.pdf"))
                return doc

            doc = asyncio.run(run_test())
            
            # Verifications
            self.assertIsNotNone(doc)
            self.assertEqual(doc.raw_text, "# Test Content")
            self.assertEqual(doc.ocr_engine, "marker")
            
            # Check if modern API was called
            mock_pdf_conv.assert_called()
            mock_marker_dict.assert_called()

    def test_doc_detector_logic(self):
        """Verify that doc_detector correctly identifies Tunisian legal patterns."""
        from doc_detector import DocumentDetector
        
        # Test Case 1: Jugement
        text = "AU NOM DU PEUPLE TUNISIEN\nLe tribunal de première instance..."
        res = DocumentDetector.detect(text)
        self.assertEqual(res.doc_type, "Jugement")
        
        # Test Case 2: Code de loi
        text = "مجلة الجزاء\nالفصل الأول: يطبق هذا القانون"
        res = DocumentDetector.detect(text)
        self.assertEqual(res.doc_type, "Code_de_loi")
        self.assertEqual(res.language, "ar")

    def test_data_generator_outputs(self):
        """Verify that the test data generator created the expected files."""
        base_path = Path("backend/data/test_inputs")
        expected_files = [
            "test_digital_like.pdf",
            "test_scanned.pdf",
            "test_multilingual.png",
            "corrupted.pdf"
        ]
        for f in expected_files:
            self.assertTrue((base_path / f).exists(), f"Missing test file: {f}")

if __name__ == "__main__":
    unittest.main()
