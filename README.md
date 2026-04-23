# Agent 2 – Mission Mindset Content Matcher

Semantischer Content-Matcher für die Mission Mindset Content Pipeline.

## Deployment auf Railway

### Schritt 1: GitHub Repository erstellen
```bash
git init
git add .
git commit -m "Agent 2 initial deployment"
git remote add origin https://github.com/DEIN-USERNAME/mission-mindset-agent2.git
git push -u origin main
```

### Schritt 2: Railway Projekt erstellen
1. Gehe zu [railway.app](https://railway.app) und logge dich ein
2. Klicke auf **"New Project"** → **"Deploy from GitHub repo"**
3. Wähle das Repository `mission-mindset-agent2`
4. Railway erkennt das Dockerfile automatisch

### Schritt 3: Umgebungsvariablen setzen
In Railway unter **Variables** folgende Werte eintragen:

| Variable | Wert |
|---|---|
| `AIRTABLE_TOKEN` | (deinen Airtable Personal Access Token eintragen) |
| `AIRTABLE_BASE_ID` | appg3lq5xmatwBoDd |
| `OPENAI_API_KEY` | (dein OpenAI API Key) |
| `CHROMA_PATH` | /data/chroma_db |

### Schritt 4: Volume für ChromaDB hinzufügen
In Railway unter **Volumes**:
- Mount Path: `/data`
- Damit bleibt die Vektordatenbank auch nach Neustarts erhalten

### Schritt 5: Initiales Befüllen der Vektordatenbank
Nach dem ersten Deployment einmalig aufrufen:
```
POST https://DEINE-RAILWAY-URL/sync
```

## API Endpoints

| Endpoint | Methode | Beschreibung |
|---|---|---|
| `/health` | GET | Status-Check |
| `/stats` | GET | Statistiken + Eintrags-Liste |
| `/match` | POST | Hook → passenden Inhalt finden |
| `/sync` | POST | Neue Transkripte indexieren |
| `/feedback` | POST | Feedback speichern |

## Lokaler Start
```bash
pip install -r requirements.txt
python3 agent2.py
```
