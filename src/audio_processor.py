import librosa
import numpy as np
from scipy.spatial.distance import cdist

class AudioProcessor:
    def __init__(self, sr=22050):
        self.sr = sr

    def load_audio(self, file_path):
        """Charge un fichier audio et retourne le signal et le sample rate."""
        y, sr = librosa.load(file_path, sr=self.sr)
        return y, sr

    def get_mel_spectrogram(self, y):
        """Calcule le Mel-Spectrogramme pour la sonorité et le timbre."""
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    def get_chromagram(self, y):
        """Calcule le Chromagramme pour les notes et accords."""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        return chroma

    def get_ssm(self, y, metric='euclidean'):
        """
        Calcule la Matrice d'Auto-Similarité (SSM) pour les patterns vs irrégularités.
        Utilise les MFCC comme base pour la comparaison temporelle.
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        # Transpose pour avoir (frames, features)
        mfcc = mfcc.T
        # Calcul de la distance entre chaque paire de frames
        ssm = cdist(mfcc, mfcc, metric=metric)
        # Normalisation
        ssm = (ssm - np.min(ssm)) / (np.max(ssm) - np.min(ssm))
        return ssm

    def extract_all_features(self, file_path):
        """Extrait toutes les features nécessaires pour une chanson."""
        y, sr = self.load_audio(file_path)
        
        features = {
            'mel': self.get_mel_spectrogram(y),
            'chroma': self.get_chromagram(y),
            'ssm': self.get_ssm(y)
        }
        return features

    def compute_pattern_score(self, ssm):
        """
        Calcule un score de 'pattern' basé sur la structure de la SSM.
        Un score élevé indique une forte répétitivité (diagonales nettes).
        Ceci est une heuristique simplifiée avant le fine-tuning.
        """
        # On peut mesurer la variance ou la densité des structures diagonales
        # Pour l'instant, c'est un placeholder pour illustrer le concept
        return np.mean(ssm) # À affiner avec le modèle
