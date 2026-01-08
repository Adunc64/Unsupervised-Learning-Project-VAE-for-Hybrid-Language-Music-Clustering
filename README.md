# Music Representation Learning & Clustering (VAE-based) — Easy / Medium / Hard

This repository implements an end-to-end pipeline for unsupervised representation learning and clustering of music-related data using Variational Autoencoders (VAEs). It covers **Easy**, **Medium**, and **Hard** project tasks:

- **Easy:** Lyrics-only (English + Bangla) → TF-IDF → MLP VAE → KMeans + PCA baseline
- **Medium:** Audio mel-spectrograms → ConvVAE → clustering (KMeans/Agglo/DBSCAN) + multimodal fusion (audio + lyrics)
- **Hard:** Multimodal baselines + **β-VAE** on fused vectors with intrinsic + extrinsic clustering metrics


---

## Setup

NB:## Dataset (Audio not included)##
The `data/audio/` folder is not tracked in this repository because it is ~1.23GB.
-> Go to https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
-> Download the data or zip file download link https://drive.google.com/file/d/14VOpH-TGAw2gSEvX5PrCUPPq9KjpcEDf/view?usp=sharing
-> Rename the "genres_original" folder name to "audio"
-> Move the "audio" folder inside the "data" folder inside the project folder 
-> Place it like this:
data/audio/<genre>/<track.wav>


### 1) Create a virtual environment (recommended)
bash
python -m venv venv
# Windows:
# venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate


### 2) Install dependencies
pip install -r requirements.txt

### 3) Dataset
Easy --> data/meta.csv
Medium/Hard --> data/audio/<genre>/<track.wav>
            --> data/lyrics/lyrics.csv

### 4) Code run command
Easy --> python run.py --task easy --k 10 --latent-dim 16 --epochs 30 (advanced)

Medium --> run.py --task medium --k 10 --latent-dim 16 --epochs 20

Hard --> run.py --task hard --k 10 --latent-dim 16 --epochs 20 --beta 4.0

