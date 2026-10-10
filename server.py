import os
from fastapi import FastAPI
from controller import Controller

app = FastAPI(title="Adaptive Precision LLM Inference Service")

# Initialize backend dynamically based on environment or hardware availability
backend_choice = os.getenv("BACKEND", "openvino").lower()
if backend_choice == "openvino":
    try:
        from backends.backend_openvino import OpenVinoBackend
        b = OpenVinoBackend()
        b.load()
    except Exception as e:
        print(f"[!] OpenVINO backend unavailable ({e}). Falling back to CPU/Dummy.")
        try:
            from backends.backend_llamacpp import LlamaCppBackend
            b = LlamaCppBackend()
            b.load()
        except Exception:
            from common.dummy_backend import DummyBackend
            b = DummyBackend()
            b.load()
elif backend_choice == "cpu":
    from backends.backend_llamacpp import LlamaCppBackend
    b = LlamaCppBackend()
    b.load()
elif backend_choice == "cuda":
    from backends.backend_cuda import CudaBackend
    b = CudaBackend()
    b.load()
else:
    from common.dummy_backend import DummyBackend
    b = DummyBackend()
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
