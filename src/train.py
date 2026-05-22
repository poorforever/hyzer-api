import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from .model import RecommenderModel, EmbeddingExtractor
from .audio_processor import AudioProcessor
import numpy as np

class SongDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def train_recommender(embeddings, labels, epochs=20, lr=0.001):
    """
    Entraîne la couche de fine-tuning sur les labels spécifiques de l'utilisateur.
    """
    dataset = SongDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = RecommenderModel(input_dim=embeddings.shape[1])
    criterion = nn.BCELoss() # Binary Cross Entropy pour classification hit/non-hit
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_emb, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_emb)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    return model

if __name__ == "__main__":
    # Exemple de données simulées pour démonstration
    # Supposons 10 chansons avec des embeddings de taille 512
    fake_embeddings = np.random.rand(10, 512)
    fake_labels = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0] # 1 = Hit pour cet utilisateur
    
    trained_model = train_recommender(fake_embeddings, fake_labels)
    torch.save(trained_model.state_dict(), "models/user_fine_tuned.pth")
    print("Modèle fine-tuné sauvegardé dans models/user_fine_tuned.pth")
