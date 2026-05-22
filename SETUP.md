# Guide d'Installation de l'Environnement Hyzer API

Ce projet nécessite Python 3.9+ et plusieurs bibliothèques de traitement audio et de Deep Learning.

## 1. Installation de Python

Si Python n'est pas installé, téléchargez-le sur [python.org](https://www.python.org/downloads/).
Assurez-vous de cocher **"Add Python to PATH"** lors de l'installation.

## 2. Création d'un environnement virtuel

Il est fortement recommandé d'utiliser un environnement virtuel :

```powershell
python -m venv venv
.\venv\Scripts\activate
```

## 3. Installation des dépendances

Une fois l'environnement activé, installez les paquets requis :

```powershell
pip install -r requirements.txt
```

*Note : L'installation de `torch` et `torchaudio` peut prendre du temps en fonction de votre connexion.*

## 4. Installation de FFmpeg (Crucial pour Librosa)

`librosa` utilise `FFmpeg` ou `PySoundFile` pour lire les fichiers audio (mp3, etc.).
Sur Windows :
1. Téléchargez FFmpeg via [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Décompressez et ajoutez le dossier `bin` à votre variable d'environnement `PATH`.

## 5. Lancer l'API

Pour démarrer le serveur de développement :

```powershell
uvicorn src.api:app --reload
```

L'API sera accessible sur `http://localhost:8000`. Vous pouvez consulter la documentation interactive sur `http://localhost:8000/docs`.
