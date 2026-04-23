#!/usr/bin/env python3
"""
Agent 2: Content-Matcher (Mission Mindset)
==========================================
Empfängt einen viralen Hook (aus der Content Pipeline Tabelle in Airtable),
sucht semantisch den passendsten Inhalt in der Content Bibliothek,
begründet die Auswahl mit GPT-4 und speichert das Ergebnis zurück in Airtable.

Lernfähigkeit: Feedback aus dem Feld "Notizen / Feedback" wird bei zukünftigen
Anfragen als Kontext mitgegeben, sodass der Agent aus Korrekturen lernt.

Trigger: Polling alle 2 Minuten – prüft ob neue Records mit Status "🔍 Hook bereit"
vorhanden sind und verarbeitet sie automatisch.
"""

import os
import time
import json
import threading
import requests
import chromadb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ─── Konfiguration ───────────────────────────────────────────────────────────
AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN", "")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "appg3lq5xmatwBoDd")
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}
CONTENT_TABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/Content%20Bibliothek"
PIPELINE_TABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/Content%20Pipeline"
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/home/ubuntu/agent2/chroma_db")
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
POLL_INTERVAL = 120  # Sekunden zwischen Polling-Durchläufen

# Airtable Status-Werte (angepasst an vorhandene Optionen)
STATUS_HOOK_BEREIT = "💡 Idee"               # Agent 1 hat Hook eingetragen (Trigger)
STATUS_IN_BEARBEITUNG = "🔍 Recherche läuft " # Agent 2 verarbeitet gerade
STATUS_INHALT_GEFUNDEN = "✍️ Text wird erstellt" # Agent 2 fertig, Agent 3 kann starten
STATUS_FEHLER = "💡 Idee"                     # Bei Fehler zurück auf Idee

# ─── Initialisierung ─────────────────────────────────────────────────────────
print("Initialisiere Agent 2 (Mission Mindset Content-Matcher)...")

print("  Lade Embedding-Modell...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("  Verbinde mit Vektordatenbank...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="content_bibliothek",
    metadata={"hnsw:space": "cosine"}
)
print(f"  Vektordatenbank: {collection.count()} Einträge")

llm_client = OpenAI()

app = FastAPI(
    title="Agent 2 – Content Matcher (Mission Mindset)",
    description="Findet semantisch passenden Inhalt für virale Hooks",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Agent 2 bereit! ✅\n")

# ─── Datenmodelle ─────────────────────────────────────────────────────────────
class HookRequest(BaseModel):
    hook: str
    pipeline_record_id: Optional[str] = None
    format: Optional[str] = "Karussell"
    zielgruppe: Optional[str] = "Angehende Life- und Mindset-Coaches"

class DashboardFeedbackRequest(BaseModel):
    hook: str
    chosen_content_id: Optional[str] = None
    rating: int  # 1 = positiv, -1 = negativ
    comment: Optional[str] = ""

class FeedbackRequest(BaseModel):
    pipeline_record_id: str
    bewertung: str  # "👍 Gut" oder "👎 Schlecht"
    kommentar: Optional[str] = ""
    besserer_inhalt_titel: Optional[str] = ""

# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────
def get_feedback_history(limit: int = 15) -> str:
    """Holt die letzten Feedback-Einträge aus dem Notizen-Feld der Pipeline."""
    try:
        resp = requests.get(
            PIPELINE_TABLE_URL,
            headers=AIRTABLE_HEADERS,
            params={
                "filterByFormula": "AND(NOT({Notizen / Feedback} = ''), NOT({Quelle aus Bibliothek} = ''))",
                "maxRecords": limit,
                "fields[]": ["Hook", "Quelle aus Bibliothek", "Notizen / Feedback"]
            },
            timeout=15
        )
        if resp.status_code != 200:
            return ""
        
        records = resp.json().get('records', [])
        if not records:
            return ""
        
        feedback_text = "Bisheriges Feedback (lerne daraus für bessere Entscheidungen):\n"
        for r in records:
            f = r['fields']
            hook = (f.get('Hook', '') or '')[:80]
            quelle = f.get('Quelle aus Bibliothek', '') or ''
            feedback = f.get('Notizen / Feedback', '') or ''
            if hook and feedback:
                feedback_text += f"- Hook: '{hook}' → Gewählt: '{quelle}' | Feedback: {feedback[:150]}\n"
        
        return feedback_text if len(feedback_text) > 80 else ""
    except Exception as e:
        print(f"  Feedback-History Fehler: {e}")
        return ""

def semantic_search(hook: str, n_results: int = 5) -> list:
    """Sucht semantisch ähnliche Inhalte in der Vektordatenbank."""
    if collection.count() == 0:
        return []
    
    query_embedding = embedding_model.encode([hook])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, collection.count())
    )
    
    matches = []
    for i, (doc_id, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - distance
        matches.append({
            "rank": i + 1,
            "airtable_id": metadata.get('airtable_id', doc_id),
            "titel": metadata.get('titel', ''),
            "typ": metadata.get('typ', ''),
            "url": metadata.get('url', ''),
            "similarity": round(similarity, 3),
            "transkript_preview": metadata.get('transkript_preview', '')[:2000]
        })
    
    return matches

def reason_with_gpt(hook: str, candidates: list, format: str, zielgruppe: str, feedback_history: str) -> dict:
    """Nutzt GPT-4 um den besten Inhalt auszuwählen und zu begründen."""
    
    candidates_text = ""
    for c in candidates:
        candidates_text += f"""
Kandidat {c['rank']} (Semantische Ähnlichkeit: {c['similarity']:.0%}):
- Titel: {c['titel']}
- Typ: {c['typ']}
- Transkript-Auszug: {c['transkript_preview'][:800]}
---"""
    
    system_prompt = """Du bist Agent 2 im Mission Mindset Content-System von Jonas Küng.

Mission Mindset hilft angehenden Life- und Mindset-Coaches dabei, ein erfolgreiches 
Coaching-Business aufzubauen. Die Zielgruppe sind Menschen die Coach werden wollen 
oder bereits als Coach tätig sind und mehr Kunden und Umsatz gewinnen wollen.

Deine Aufgabe: Analysiere den Hook und finde den Inhalt aus der Content Bibliothek 
der am besten dazu passt – nicht nur nach Keywords, sondern nach tiefem inhaltlichem 
Verständnis. Der Inhalt soll die Kernaussage des Hooks mit echten Beispielen, 
Geschichten und Expertise aus Jonas' Arbeit untermauern.

Antworte IMMER im JSON-Format."""

    user_prompt = f"""Hook: "{hook}"
Content-Format: {format}
Zielgruppe: {zielgruppe}

{feedback_history}

Kandidaten aus der Content Bibliothek:
{candidates_text}

Wähle den BESTEN Kandidaten. Antworte im JSON-Format:
{{
    "gewählter_kandidat": <Nummer 1-{len(candidates)}>,
    "gewählter_titel": "<exakter Titel>",
    "gewählter_airtable_id": "<Airtable Record ID>",
    "begründung": "<Warum passt dieser Inhalt am besten? 2-3 präzise Sätze>",
    "kernbotschaft": "<Die eine Botschaft die Hook + Inhalt verbindet. 1 kraftvoller Satz>",
    "content_winkel": "<Welchen spezifischen Aspekt/Moment/Zitat aus dem Transkript soll Agent 3 nutzen?>",
    "vertrauens_score": <0-100>,
    "alternativer_kandidat": <Nummer des zweitbesten>
}}"""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"  GPT-Fehler: {e}")
        # Fallback: Besten semantischen Match nehmen
        return {
            "gewählter_kandidat": 1,
            "gewählter_titel": candidates[0]['titel'],
            "gewählter_airtable_id": candidates[0]['airtable_id'],
            "begründung": f"Höchste semantische Ähnlichkeit ({candidates[0]['similarity']:.0%})",
            "kernbotschaft": "",
            "content_winkel": "",
            "vertrauens_score": int(candidates[0]['similarity'] * 100),
            "alternativer_kandidat": 2
        }

def save_to_pipeline(pipeline_record_id: str, hook: str, match_result: dict, candidates: list) -> bool:
    """Speichert das Ergebnis in der Content Pipeline Tabelle."""
    gewählter_idx = match_result.get('gewählter_kandidat', 1) - 1
    if gewählter_idx < 0 or gewählter_idx >= len(candidates):
        gewählter_idx = 0
    
    gewählter = candidates[gewählter_idx]
    
    # Zusammenfassung für das Notizen-Feld
    zusammenfassung = f"""🤖 Agent 2 Analyse:

📌 Gewählter Inhalt: {match_result.get('gewählter_titel', '')}
🎯 Kernbotschaft: {match_result.get('kernbotschaft', '')}
📐 Content-Winkel: {match_result.get('content_winkel', '')}
💯 Vertrauens-Score: {match_result.get('vertrauens_score', 0)}/100

📝 Begründung: {match_result.get('begründung', '')}

📚 Transkript-Auszug:
{gewählter.get('transkript_preview', '')[:3000]}"""
    
    update_data = {
        "fields": {
            "Status": "✍️ Text wird erstellt",
            "Quelle aus Bibliothek": match_result.get('gewählter_titel', gewählter['titel']),
            "Thema / Idee": zusammenfassung[:10000]
        }
    }
    
    try:
        resp = requests.patch(
            f"{PIPELINE_TABLE_URL}/{pipeline_record_id}",
            headers=AIRTABLE_HEADERS,
            json=update_data,
            timeout=20
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"  Airtable-Speicher-Fehler: {e}")
        return False

def sync_vector_db() -> int:
    """Synchronisiert neue Transkripte aus Airtable in die Vektordatenbank."""
    existing_ids = set(collection.get()['ids'])
    new_records = []
    offset = None
    
    while True:
        params = {
            "filterByFormula": "{Status Transkript} = '\u2705 Transkribiert'",
            "maxRecords": 100,
            "fields[]": ["Titel", "Transkript", "Typ", "Notizen", "URL / Link"]
        }
        if offset:
            params['offset'] = offset
        
        resp = requests.get(CONTENT_TABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=30)
        data = resp.json()
        
        for r in data.get('records', []):
            t = r['fields'].get('Transkript', '')
            if (t and len(t) > 200 
                and not t.startswith('Starting video analysis')
                and r['id'] not in existing_ids):
                new_records.append(r)
        
        offset = data.get('offset')
        if not offset:
            break
        time.sleep(0.2)
    
    if not new_records:
        return 0
    
    texts, ids, metadatas = [], [], []
    for record in new_records:
        title = record['fields'].get('Titel', '')
        transcript = record['fields'].get('Transkript', '')
        doc_text = f"Titel: {title}\nTyp: {record['fields'].get('Typ', '')}\n\nInhalt:\n{transcript[:3000]}"
        texts.append(doc_text)
        ids.append(record['id'])
        metadatas.append({
            "airtable_id": record['id'],
            "titel": title[:500],
            "typ": record['fields'].get('Typ', ''),
            "url": (record['fields'].get('URL / Link', '') or '')[:500],
            "notizen": (record['fields'].get('Notizen', '') or '')[:500],
            "transkript_laenge": len(transcript),
            "transkript_preview": transcript[:8000]
        })
    
    embeddings = embedding_model.encode(texts, batch_size=16)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=[t[:2000] for t in texts],
        metadatas=metadatas
    )
    
    return len(new_records)

def process_single_record(record_id: str, hook: str, format_type: str):
    """Verarbeitet einen einzelnen Pipeline-Record."""
    print(f"\n  Verarbeite: '{hook[:60]}...'")
    
    # Status auf "In Bearbeitung" setzen
    requests.patch(
        f"{PIPELINE_TABLE_URL}/{record_id}",
        headers=AIRTABLE_HEADERS,
        json={"fields": {"Status": STATUS_IN_BEARBEITUNG}},
        timeout=10
    )
    
    # Semantische Suche
    candidates = semantic_search(hook, n_results=5)
    if not candidates:
        requests.patch(
            f"{PIPELINE_TABLE_URL}/{record_id}",
            headers=AIRTABLE_HEADERS,
            json={"fields": {"Status": STATUS_FEHLER, "Thema / Idee": "Keine passenden Inhalte in der Vektordatenbank gefunden."}},
            timeout=10
        )
        return
    
    # Feedback-History
    feedback_history = get_feedback_history()
    
    # GPT-4 Reasoning
    match_result = reason_with_gpt(hook, candidates, format_type, "Angehende Life- und Mindset-Coaches", feedback_history)
    
    # In Airtable speichern
    saved = save_to_pipeline(record_id, hook, match_result, candidates)
    
    if saved:
        print(f"  ✅ Gespeichert: '{match_result.get('gewählter_titel', '')[:50]}' (Score: {match_result.get('vertrauens_score', 0)})")
    else:
        print(f"  ❌ Speichern fehlgeschlagen")

def polling_loop():
    """Polling-Loop: Prüft alle 2 Minuten auf neue Hooks in der Content Pipeline."""
    print(f"Polling-Loop gestartet (alle {POLL_INTERVAL}s)\n")
    
    while True:
        try:
            # Neue Records mit Status "Hook bereit" holen
            resp = requests.get(
                PIPELINE_TABLE_URL,
                headers=AIRTABLE_HEADERS,
                params={
                    "filterByFormula": f"AND({{Status}} = '{STATUS_HOOK_BEREIT}', NOT({{Hook}} = ''))",
                    "maxRecords": 10,
                    "fields[]": ["Hook", "Status", "Content Format"]
                },
                timeout=15
            )
            
            if resp.status_code == 200:
                records = resp.json().get('records', [])
                if records:
                    print(f"[Polling] {len(records)} neue Hook(s) gefunden!")
                    for record in records:
                        hook = record['fields'].get('Hook', '')
                        format_type = record['fields'].get('Content Format', 'Karussell')
                        if hook:
                            process_single_record(record['id'], hook, format_type)
                
                # Auch Vektordatenbank synchronisieren (alle 10 Minuten)
                if int(time.time()) % 600 < POLL_INTERVAL:
                    new_count = sync_vector_db()
                    if new_count > 0:
                        print(f"[Sync] {new_count} neue Transkripte in Vektordatenbank aufgenommen")
        
        except Exception as e:
            print(f"[Polling] Fehler: {e}")
        
        time.sleep(POLL_INTERVAL)

# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "agent": "Agent 2 – Content Matcher (Mission Mindset)",
        "status": "aktiv",
        "vektordatenbank": f"{collection.count()} Einträge",
        "polling": f"alle {POLL_INTERVAL}s",
        "trigger_status": f"Warte auf Records mit Status '{STATUS_HOOK_BEREIT}'"
    }

@app.get("/health")
def health():
    return {"status": "ok", "db_size": collection.count()}

@app.post("/match")
def match_content(request: HookRequest):
    """Manueller Endpunkt: Hook eingeben und sofort Ergebnis erhalten."""
    hook = request.hook.strip()
    if not hook:
        raise HTTPException(status_code=400, detail="Hook darf nicht leer sein")
    
    if collection.count() == 0:
        raise HTTPException(status_code=503, detail="Vektordatenbank ist leer")
    
    candidates = semantic_search(hook, n_results=5)
    if not candidates:
        raise HTTPException(status_code=404, detail="Keine passenden Inhalte gefunden")
    
    feedback_history = get_feedback_history()
    match_result = reason_with_gpt(hook, candidates, request.format, request.zielgruppe, feedback_history)
    
    saved = False
    if request.pipeline_record_id:
        saved = save_to_pipeline(request.pipeline_record_id, hook, match_result, candidates)
    
    gewählter_idx = match_result.get('gewählter_kandidat', 1) - 1
    if gewählter_idx < 0 or gewählter_idx >= len(candidates):
        gewählter_idx = 0
    
    return {
        "hook": hook,
        "title": match_result.get('gewählter_titel'),
        "core_message": match_result.get('kernbotschaft'),
        "reasoning": match_result.get('begründung'),
        "transcript_excerpt": candidates[gewählter_idx].get('transkript_preview', '')[:2000],
        "score": match_result.get('vertrauens_score'),
        "content_id": match_result.get('gewählter_airtable_id'),
        "gewählter_inhalt": {
            "titel": match_result.get('gewählter_titel'),
            "airtable_id": match_result.get('gewählter_airtable_id'),
            "begründung": match_result.get('begründung'),
            "kernbotschaft": match_result.get('kernbotschaft'),
            "content_winkel": match_result.get('content_winkel'),
            "vertrauens_score": match_result.get('vertrauens_score'),
            "transkript_auszug": candidates[gewählter_idx].get('transkript_preview', '')[:2000]
        },
        "alle_kandidaten": [
            {"rang": c['rank'], "titel": c['titel'], "ähnlichkeit": f"{c['similarity']:.0%}"}
            for c in candidates
        ],
        "in_airtable_gespeichert": saved
    }

@app.post("/sync")
def sync_database():
    """Synchronisiert neue Transkripte in die Vektordatenbank."""
    new_count = sync_vector_db()
    return {
        "items_added": new_count,
        "neue_einträge": new_count,
        "gesamt_in_db": collection.count(),
        "message": f"{new_count} neue Transkripte aufgenommen",
        "nachricht": f"{new_count} neue Transkripte aufgenommen"
    }

@app.get("/stats")
def get_stats():
    """Gibt Statistiken und Liste aller indexierten Inhalte zurück."""
    try:
        all_items = collection.get(include=["metadatas"])
        entries = []
        for i, meta in enumerate(all_items.get("metadatas", [])):
            if meta:
                entries.append({
                    "id": all_items["ids"][i] if i < len(all_items.get("ids", [])) else str(i),
                    "title": meta.get("titel", meta.get("title", "Unbekannt")),
                    "platform": meta.get("plattform", meta.get("platform", "")),
                    "type": meta.get("typ", meta.get("type", "")),
                    "char_count": meta.get("zeichenanzahl", meta.get("char_count", 0)),
                })
    except Exception as e:
        entries = []
    return {
        "total_entries": collection.count(),
        "vektordatenbank_einträge": collection.count(),
        "embedding_modell": EMBEDDING_MODEL,
        "trigger_status": STATUS_HOOK_BEREIT,
        "poll_interval_sekunden": POLL_INTERVAL,
        "entries": entries
    }

@app.post("/feedback")
def submit_feedback(request: DashboardFeedbackRequest):
    """Empfängt Feedback vom Dashboard und speichert es für zukünftiges Lernen."""
    rating_text = "👍 Gut" if request.rating > 0 else "👎 Schlecht"
    comment = request.comment or ""
    feedback_text = f"{rating_text}: {comment}".strip(": ")
    
    # Wenn wir eine Airtable-ID haben, direkt updaten
    if request.chosen_content_id:
        try:
            resp = requests.patch(
                f"{PIPELINE_TABLE_URL}/{request.chosen_content_id}",
                headers=AIRTABLE_HEADERS,
                json={"fields": {"Notizen / Feedback": feedback_text}},
                timeout=10
            )
        except Exception as e:
            print(f"[Feedback] Airtable-Update fehlgeschlagen: {e}")
    
    return {
        "success": True,
        "message": f"Feedback gespeichert: {rating_text}",
        "rating": request.rating
    }

@app.on_event("startup")
async def startup_event():
    """Startet den Polling-Loop beim App-Start."""
    thread = threading.Thread(target=polling_loop, daemon=True)
    thread.start()
    print("Polling-Loop gestartet!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
