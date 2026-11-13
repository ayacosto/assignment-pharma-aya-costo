# Pharma RAG Assistant

Assistant agentique (RAG) qui répond à des questions sur un corpus de médicaments (DailyMed / MedlinePlus).


## Installation

```bash
pip install -r requirements.txt
```

## Construire les index FAISS

```bash
python -m ingestion.build_vectorstore chemin\vers\corpus
```

## Lancer avec Docker

```bash
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

API disponible sur : http://localhost:8000


## Frontend

Ouvrir `frontend/index.html` dans un navigateur.