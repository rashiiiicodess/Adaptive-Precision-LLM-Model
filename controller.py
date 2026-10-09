from classifier import classify

CONF_THRESHOLD = 0.55  # Minimum acceptable confidence score before triggering INT8 retry


class Controller:
    def __init__(self, backend):
        self.b = backend

    def ask(self, prompt: str, max_new_tokens: int = None):
        # 1. Classify prompt complexity
        level = classify(prompt)

        # 2. Hard questions (GSM8K) need more tokens to reason step-by-step
        if max_new_tokens is None:
            tokens_to_gen = 256 if level == "hard" else 128
        else:
            tokens_to_gen = max_new_tokens

        # 3. Pick precision: hard goes to int8, easy/medium goes to int4
        precision = "int8" if level == "hard" else "int4"

        # 4. Generate response with selected precision
        r = self.b.generate(prompt, precision, max_new_tokens=tokens_to_gen)
        retried = False

        # 5. Adaptive Feedback Loop: if int4 confidence is low, retry with int8
        if precision == "int4" and r.confidence is not None and r.confidence < CONF_THRESHOLD:
            r = self.b.generate(prompt, "int8", max_new_tokens=tokens_to_gen)
            precision = "int8"
            retried = True

        return r, level, precision, retried
