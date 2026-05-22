from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import os
from .audio_processor import AudioProcessor
from .model import EmbeddingExtractor, RecommenderModel
import torch

app = FastAPI(title="Hyzer API", description="Moteur de recommandation musical hyperspécifique")

# Initialisation des composants (Singleton style)
processor = AudioProcessor()
extractor = EmbeddingExtractor()
model = RecommenderModel() # À charger avec les poids fine-tunés ultérieurement

class Interaction(BaseModel):
    track_id: str
    play_count: int
    duration_ms: int
    is_hit: Optional[bool] = None

class AgentInput(BaseModel):
    user_id: str
    history: List[Interaction]
    limit: Optional[int] = 10

@app.post("/analyze")
async def analyze_track(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio et extrait ses caractéristiques hyperspécifiques.
    """
    # Sauvegarde temporaire du fichier
    temp_path = f"data/temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 1. Extraction des features audio
        features = processor.extract_all_features(temp_path)
        
        # 2. Extraction de l'embedding (ADN)
        y, sr = processor.load_audio(temp_path)
        embedding = extractor.get_embedding(y, sr=sr)
        
        # 3. Calcul du score pattern (exemple heuristique)
        pattern_score = processor.compute_pattern_score(features['ssm'])
        
        return {
            "filename": file.filename,
            "pattern_score": float(pattern_score),
            "embedding_sample": embedding[:5].tolist(), # Juste un aperçu
            "message": "Analyse complétée avec succès"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/recommend")
async def get_recommendations(data: AgentInput):
    """
    Détermine des matches basés sur l'historique d'écoute et les patterns préférés du user.
    """
    # Ici, on chargerait les embeddings de la librairie de chansons
    # et on utiliserait le RecommenderModel pour trouver les meilleurs scores.
    return {
        "user_id": data.user_id,
        "recommendations": [
            {"track_id": "rec_1", "match_score": 0.95, "reason": "Forte similarité de pattern (SSM)"},
            {"track_id": "rec_2", "match_score": 0.88, "reason": "Timbre spectral correspondant (Mel-Spectro)"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
