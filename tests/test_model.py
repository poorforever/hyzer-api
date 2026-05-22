import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Ajout du chemin src pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import EmbeddingExtractor, RecommenderModel

class TestModel(unittest.TestCase):
    def test_recommender_model_forward(self):
        input_dim = 512
        model = RecommenderModel(input_dim=input_dim)
        batch_size = 4
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 1))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def test_embedding_extractor_with_mock(self, mock_processor, mock_model):
        # Configurer les mocks
        extractor = EmbeddingExtractor()
        
        # Simuler un embedding
        fake_audio = np.random.rand(16000)
        
        # Test de la méthode get_embedding
        # Si le modèle est None (chargement échoué), il doit retourner un vecteur aléatoire
        extractor.model = None
        embedding = extractor.get_embedding(fake_audio)
        self.assertEqual(len(embedding), 512)
        self.assertIsInstance(embedding, np.ndarray)

    def test_embedding_extractor_fallback(self):
        # Test du fallback quand le modèle ne charge pas
        with patch('transformers.AutoModel.from_pretrained', side_effect=Exception("Error")):
            extractor = EmbeddingExtractor()
            self.assertIsNone(extractor.model)
            embedding = extractor.get_embedding(np.zeros(100))
            self.assertEqual(len(embedding), 512)

if __name__ == '__main__':
    unittest.main()
