import os
import time
import psutil
import torch
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from common.interface import Backend, GenResult

# We can load the tokenizer directly from our local exported folder
PATHS = {
    "int4": "models/qwen-ov-int4",
    "int8": "models/qwen-ov-int8",
}


class OpenVinoBackend(Backend):
    name = "openvino"

    def load(self, precisions=("int4", "int8"), device="GPU"):
        # Load tokenizer from local disk
        self.tok = AutoTokenizer.from_pretrained(PATHS["int8"])
        # Compile and load both models onto your Intel Arc GPU
        self.models = {
            p: OVModelForCausalLM.from_pretrained(PATHS[p], device=device)
            for p in precisions
        }

    def generate(self, prompt: str, precision: str, max_new_tokens: int = 128) -> GenResult:
        m = self.models[precision]

        # 1. Format the question so Qwen knows a human is talking
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tok(text, return_tensors="pt")

        # 2. Start the stopwatch
        t0 = time.perf_counter()

        # 3. Ask the model to generate words
        out = m.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # 4. Stop the stopwatch
        dt = time.perf_counter() - t0

        # 5. Extract only the newly created tokens (cut off the original prompt)
        gen = out.sequences[0][inputs["input_ids"].shape[1]:]

        # 6. Calculate confidence score (how certain the model felt about its answer)
        try:
            scores = m.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )[0]
            conf = float(scores.exp().mean())
        except Exception:
            # Fallback if compute_transition_scores is not supported
            try:
                confs = []
                for i, token_id in enumerate(gen):
                    step_probs = torch.softmax(out.scores[i][0], dim=-1)
                    confs.append(float(step_probs[token_id]))
                conf = sum(confs) / len(confs) if confs else None
            except Exception:
                conf = None

        # 7. Check how much computer memory (RAM) is being used in MB
        mem = psutil.Process(os.getpid()).memory_info().rss / 1e6

        # 8. Turn computer tokens back into human English
        decoded_text = self.tok.decode(gen, skip_special_tokens=True)

        return GenResult(
            text=decoded_text,
            latency_s=dt,
            tokens=len(gen),
            memory_mb=mem,
            confidence=conf,
        )
