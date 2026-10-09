from fastapi import FastAPI
from controller import Controller
from backends.backend_openvino import OpenVinoBackend

app = FastAPI(title="Adaptive Precision LLM Inference Service")

# Initialize backend on startup and compile to Intel Arc GPU
b = OpenVinoBackend()
b.load()
ctl = Controller(b)

# Request history log for telemetry and monitoring
log = []


@app.post("/ask")
def ask(body: dict):
    """
    Accepts JSON payload: {"prompt": "Your question here"}
    Routes through adaptive controller, returns answer + metrics.
    """
    prompt = body.get("prompt", "")
    r, level, prec, retried = ctl.ask(prompt)

    entry = {
        "level": level,
        "precision": prec,
        "retried": retried,
        "latency": r.latency_s,
    }
    log.append(entry)

    return {
        "answer": r.text,
        "complexity": level,
        "precision": prec,
        "retried": retried,
        "latency_s": r.latency_s,
        "memory_mb": r.memory_mb,
    }


@app.get("/stats")
def stats():
    """Returns past request telemetry log."""
    return log
