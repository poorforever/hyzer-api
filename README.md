# Hyzer API - Moteur de Recommandation Musicale

API pour l'analyse hyperspécifique de morceaux de musique et la recommandation basée sur des patterns (SSM, Mel-Spectrogram).

## 🚀 Utilisation de l'Agent

L'agent Hyzer s'exécute en envoyant une requête POST à `/recommend` avec le payload JSON suivant. 

### Spécification des Données d'Entrée (JSON)

Ce format est conçu pour être facilement copié et parsé par des clients automatiques ou d'autres agents.

<details open>
<summary>📋 JSON d'entrée type (copiable)</summary>

```json
{
  "user_id": "user_12345_sample",
  "history": [
    {
      "track_id": "track_unique_id_001",
      "play_count": 15,
      "duration_ms": 210000,
      "is_hit": true
    },
    {
      "track_id": "track_unique_id_002",
      "play_count": 2,
      "duration_ms": 35000,
      "is_hit": false
    }
  ],
  "limit": 5
}
```

</details>

## 🛠 Installation

Voir [SETUP.md](SETUP.md) pour les instructions d'installation détaillées.
