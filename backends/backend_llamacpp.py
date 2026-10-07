import math
import os
import psutil
import time
from common.interface import Backend, GenResult
from llama_cpp import Llama


PATHS = {
    "int4": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "int8": "models/qwen2.5-1.5b-instruct-q8_0.gguf",
}


class LlamaCppBackend(Backend):
    name = "cpu"

    def load(self, precisions=("int4", "int8")):
        """Load the GGUF models into memory with optimized thread counts."""
        # Use physical CPU core count (os.cpu_count() // 2) to prevent hyperthreading contention
        threads = max(1, (os.cpu_count() or 2) // 2)
        
        self.models = {}
        for p in precisions:
            if p not in PATHS:
                raise ValueError(f"Unknown precision '{p}'. Available: {list(PATHS.keys())}")
            
            model_path = PATHS[p]
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. Please ensure weights are downloaded."
                )
            
            self.models[p] = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=threads,
                logits_all=True,  # Required by llama-cpp-python for logprob / confidence extraction
                verbose=False,
            )

    def generate(self, prompt: str, precision: str, max_new_tokens: int = 128) -> GenResult:
        """Run greedy generation on the specified quantized GGUF model and measure metrics."""
        if precision not in self.models:
            raise RuntimeError(
                f"Precision '{precision}' not loaded. Currently loaded: {list(self.models.keys())}"
            )

        model = self.models[precision]
        
        # Measure wall-clock inference latency
        start_time = time.perf_counter()
        output = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,  # Greedy deterministic decoding
            logprobs=True,
            top_logprobs=1,
        )
        latency_s = time.perf_counter() - start_time

        # Extract generated text and token count
        choice = output["choices"][0]
        text = choice["message"]["content"]
        tokens_generated = output["usage"]["completion_tokens"]

        # Extract and compute mean token confidence from log-probabilities
        # P = exp(logprob)
        conf = None
        logprobs_data = (choice.get("logprobs") or {}).get("content", [])
        if logprobs_data:
            extracted_lps = [
                x["logprob"] for x in logprobs_data
                if isinstance(x, dict) and "logprob" in x and x["logprob"] is not None
            ]
            if extracted_lps:
                conf = float(sum(math.exp(lp) for lp in extracted_lps) / len(extracted_lps))

        # Measure Process Resident Set Size (RSS) in Megabytes
        memory_mb = float(psutil.Process(os.getpid()).memory_info().rss / 1e6)

        return GenResult(
            text=text,
            latency_s=latency_s,
            tokens=tokens_generated,
            memory_mb=memory_mb,
            confidence=conf,
        )
