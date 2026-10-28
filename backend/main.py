import base64, json, uuid
from html import escape
from pathlib import Path
import cv2
import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

MODEL = "claude-sonnet-5-5"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
app = FastAPI(title="CrimeVision AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA = Path("data"); EVID = DATA / "evidence"; EVID.mkdir(parents=True, exist_ok=True)
CASES = DATA / "cases.json"  # JSON store

SYSTEM = ("You are a forensic analysis assistant. Describe only what is visible or stated; never guess identities. "
          "Reply with JSON only, keys: crime_type (Robbery, Assault, Accident, Theft, Vandalism or None), "
          "confidence (0-1), summary, objects (list), people (list of short descriptions), weapons (list), "
          "timeline (list of {time, event}), transcript (string, empty if none).")

def ask(content, system=SYSTEM):
    r = client.messages.create(model=MODEL, max_tokens=1500, system=system,
                               messages=[{"role": "user", "content": content}])
    return r.content[0].text

def parse(text):
    return json.loads(text[text.find("{"): text.rfind("}") + 1])

def img_block(data, mt="image/jpeg"):
    return {"type": "image", "source": {"type": "base64", "media_type": mt,
                                         "data": base64.b64encode(data).decode()}}

def load_all():
    return json.loads(CASES.read_text()) if CASES.exists() else {}

def save(kind, name, data, result):
    cid = uuid.uuid4().hex[:8]
    (EVID / f"{cid}_{name}").write_bytes(data)
    result.update(case_id=cid, kind=kind, filename=name)
    cases = load_all(); cases[cid] = result
    CASES.write_text(json.dumps(cases, indent=2))
    return result

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    out = parse(ask([img_block(data, file.content_type or "image/jpeg"),
                     {"type": "text", "text": "Analyze this crime scene image."}]))
    return save("image", file.filename, data, out)

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    data = await file.read()
    tmp = DATA / "tmp_video"; tmp.write_bytes(data)
    cap = cv2.VideoCapture(str(tmp)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = max(int(fps * 2), n // 8)
    content = []
    for i in range(0, n, step):  # sample about 8 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, frame = cap.read()
        if ok:
            content += [{"type": "text", "text": f"Frame at {i / fps:.0f}s:"},
                        img_block(cv2.imencode(".jpg", frame)[1].tobytes())]
    cap.release()
    if not content:
        raise HTTPException(400, "Could not read frames from this video")
    content.append({"type": "text", "text": "Build a timeline of events from these frames."})
    return save("video", file.filename, data, parse(ask(content)))

_w = None
def whisper():
    global _w
    if _w is None:
        from faster_whisper import WhisperModel
        _w = WhisperModel("base", compute_type="int8")
    return _w

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    data = await file.read()
    tmp = DATA / ("tmp_audio" + (Path(file.filename).suffix or ".wav")); tmp.write_bytes(data)
    segs, _ = whisper().transcribe(str(tmp))
    text = "\n".join(f"[{int(s.start)}s] {s.text.strip()}" for s in segs)
    out = parse(ask(f"Witness statement transcript:\n{text}\nSummarize and classify."))
    out["transcript"] = text
    return save("audio", file.filename, data, out)

class Q(BaseModel):
    case_id: str
    question: str

@app.post("/chat")
def chat(q: Q):
    case = load_all().get(q.case_id)
    if not case:
        raise HTTPException(404, "Unknown case")
    a = ask(f"Case evidence (JSON):\n{json.dumps(case)}\n\nInvestigator question: {q.question}",
            "Answer only from the case evidence. If it is not there, say so.")
    return {"answer": a}

@app.get("/report/{cid}")
def report(cid: str):
    c = load_all().get(cid)
    if not c:
        raise HTTPException(404, "Unknown case")
    st = getSampleStyleSheet()
    P = lambda t, s="BodyText": Paragraph(escape(str(t)), st[s])
    items = [P(f"Forensic Report: Case {cid}", "Title"), P(f"Evidence: {c['filename']} ({c['kind']})"),
             P(f"Classification: {c.get('crime_type')} (confidence {c.get('confidence')})"),
             P("Summary", "Heading2"), P(c.get("summary", "")),
             P("Objects: " + ", ".join(map(str, c.get("objects", [])))),
             P("People: " + "; ".join(map(str, c.get("people", [])))),
             P("Weapons: " + ", ".join(map(str, c.get("weapons", [])))), P("Timeline", "Heading2")]
    items += [P(f"{t.get('time')}: {t.get('event')}") for t in c.get("timeline", [])]
    if c.get("transcript"):
        items += [P("Transcript", "Heading2")] + [P(l) for l in c["transcript"].split("\n")]
    path = DATA / f"report_{cid}.pdf"
    SimpleDocTemplate(str(path)).build(items)
    return FileResponse(path, filename=path.name)

# Serves the React frontend at http://localhost:8000
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True))
