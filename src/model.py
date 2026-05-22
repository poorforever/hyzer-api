import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoProcessor as CLAPProcessor

class EmbeddingExtractor:
    def __init__(self, model_name="laion/clap-htsat-unfused"):
        """
        Initialise le modèle de base (CLAP par défaut) pour l'extraction d'embeddings.
        CLAP est excellent car il lie l'audio au texte et capture des concepts sémantiques.
        """
        # Note: Dans un environnement réel, cela nécessiterait une connexion internet pour télécharger le modèle
        try:
            self.processor = CLAPProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.model = None
            self.processor = None

    def get_embedding(self, audio_data, sr=48000):
        """
        Extrait le vecteur ADN (embedding) du morceau audio.
        """
        if self.model is None:
            # Retourne un vecteur aléatoire de taille standard (ex: 512) pour le développement si le modèle n'est pas chargé
            return np.random.rand(512)

        inputs = self.processor(audios=audio_data, return_tensors="pt", sampling_rate=sr)
        with torch.no_grad():
            outputs = self.model.get_audio_features(**inputs)
        
        return outputs.numpy()[0]

class RecommenderModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        """
        Petit réseau de neurones pour le fine-tuning.
        Il prend l'embedding du modèle de base + éventuellement des features manuelles (SSM score, etc.)
        """
        super(RecommenderModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # Sortie : score de 'match' ou classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
