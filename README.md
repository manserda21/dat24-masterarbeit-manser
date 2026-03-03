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