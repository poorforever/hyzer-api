import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock
import numpy as np

# Ajout du chemin src pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock des dépendances avant d'importer l'app pour éviter le chargement des modèles
with patch('src.model.EmbeddingExtractor.__init__', return_value=None):
    with patch('src.model.RecommenderModel.__init__', return_value=None):
        from src.api import app
        from fastapi.testclient import TestClient

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def test_recommend_endpoint(self):
        response = client.post(
            "/recommend",
            data={"user_id": "user123", "history": ["song1", "song2"]}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["user_id"], "user123")
        self.assertIn("recommendations", data)

    @patch('src.audio_processor.AudioProcessor.extract_all_features')
    @patch('src.audio_processor.AudioProcessor.load_audio')
    @patch('src.model.EmbeddingExtractor.get_embedding')
    def test_analyze_endpoint(self, mock_get_embedding, mock_load_audio, mock_extract):
        # Configurer les mocks
        mock_extract.return_value = {
            'mel': np.zeros((128, 10)),
            'chroma': np.zeros((12, 10)),
            'ssm': np.zeros((10, 10))
        }
        mock_load_audio.return_value = (np.zeros(1000), 22050)
        mock_get_embedding.return_value = np.zeros(512)
        
        # Créer un fichier audio fictif
        file_content = b"fake audio content"
        files = {"file": ("test.wav", file_content, "audio/wav")}
        
        # S'assurer que le dossier data existe
        if not os.path.exists("data"):
            os.makedirs("data")
            
        response = client.post("/analyze", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pattern_score", data)
        self.assertIn("embedding_sample", data)
        self.assertEqual(data["filename"], "test.wav")

if __name__ == '__main__':
    unittest.main()
