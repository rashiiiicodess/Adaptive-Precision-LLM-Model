import random
import time
from common.interface import Backend, GenResult


class DummyBackend(Backend):
    name = "dummy"

    def load(self, precisions=("int4", "int8")):
        pass

    def generate(self, prompt: str, precision: str, max_new_tokens: int = 128) -> GenResult:
        time.sleep(0.2 if precision == "int4" else 0.35)
        return GenResult(
            text="dummy answer 42",
            latency_s=0.2 if precision == "int4" else 0.35,
            tokens=10,
            memory_mb=1000.0,
            confidence=random.uniform(0.3, 0.95),
        )
