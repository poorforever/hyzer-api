import unittest
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Ajout du chemin src pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor(sr=22050)
        # Créer un signal audio synthétique (1 seconde de silence/bruit)
        self.duration = 1.0
        self.y = np.random.uniform(-1, 1, int(self.processor.sr * self.duration))

    def test_get_mel_spectrogram(self):
        mel = self.processor.get_mel_spectrogram(self.y)
        self.assertIsInstance(mel, np.ndarray)
        self.assertEqual(mel.shape[0], 128) # n_mels

    def test_get_chromagram(self):
        chroma = self.processor.get_chromagram(self.y)
        self.assertIsInstance(chroma, np.ndarray)
        self.assertEqual(chroma.shape[0], 12) # 12 notes chromatiques

    def test_get_ssm(self):
        ssm = self.processor.get_ssm(self.y)
        self.assertIsInstance(ssm, np.ndarray)
        self.assertEqual(ssm.ndim, 2)
        self.assertEqual(ssm.shape[0], ssm.shape[1])
        # Vérifier la normalisation
        self.assertGreaterEqual(np.min(ssm), 0.0)
        self.assertLessEqual(np.max(ssm), 1.0)

    def test_compute_pattern_score(self):
        ssm = np.eye(10) # Matrice identité (diagonale parfaite)
        score = self.processor.compute_pattern_score(ssm)
        self.assertIsInstance(score, (float, np.float32, np.float64))
        self.assertEqual(score, 0.1) # Moyenne de 10/100

    @patch('librosa.load')
    def test_extract_all_features(self, mock_load):
        mock_load.return_value = (self.y, self.processor.sr)
        
        # On utilise un chemin fictif
        features = self.processor.extract_all_features("fake_path.wav")
        
        self.assertIn('mel', features)
        self.assertIn('chroma', features)
        self.assertIn('ssm', features)
        mock_load.assert_called_once_with("fake_path.wav", sr=self.processor.sr)

if __name__ == '__main__':
    unittest.main()
