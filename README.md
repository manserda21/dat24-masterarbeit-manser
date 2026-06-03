## Git Branching Workflow

Ich verwende einen `main`-Branch für stabile Versionen und einen `dev`-Branch für aktive Entwicklung.

### Wichtige Befehle

#### Dev-Branch erstellen 

```bash
git checkout -b dev         # erstellt und wechselt zum 'dev'-Branch
git push -u origin dev      # veröffentlicht den Branch auf GitHub
```

#### Zwischen branches wechseln
```bash
git checkout dev            # wechselt zur Entwicklung
git checkout main           # wechselt zum Haupt-Branch
```

### venv aktivieren
```bash
source /data/manser/venvs/venv_masterarbeit/bin/activate
```

### Trainingsskript starten

```bash
python3 -m src.train \
--data /data/manser/datasets/real_players_dataset_v1_30_frames/data.yaml \
--name real_players_v1 \
--model yolov8m.pt \
--imgsz 1280 \
--batch 4 \
--epochs 50
```