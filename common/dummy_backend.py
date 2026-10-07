import random
import time
from common.interface import Backend, GenResult


class DummyBackend(Backend):
    name = "dummy"

    def load(self, precisions=("int4", "int8")):
        """Simulate loading models into memory without actual weight initialization."""
        pass

    def generate(self, prompt: str, precision: str, max_new_tokens: int = 128) -> GenResult:
        """Simulate inference latency and generate a dummy response for pipeline testing."""
        simulated_latency = 0.20 if precision == "int4" else 0.35
        time.sleep(simulated_latency)
        
        # Simulated confidence score between 0.30 and 0.95
        simulated_confidence = random.uniform(0.3, 0.95)
        
        return GenResult(
            text="dummy answer 42",
            latency_s=simulated_latency,
            tokens=10,
            memory_mb=1000.0,
            confidence=simulated_confidence
        )
