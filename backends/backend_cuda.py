import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from common.interface import Backend, GenResult


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class CudaBackend(Backend):
    name = "cuda"

    def load(self, precisions=("int4", "int8")):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.models = {}

        cfg = {
            "int4": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "int8": BitsAndBytesConfig(
                load_in_8bit=True,
            ),
        }

        for precision in precisions:
            if precision == "fp16":
                self.models[precision] = (
                    AutoModelForCausalLM.from_pretrained(
                        MODEL,
                        torch_dtype=torch.float16,
                        device_map="cuda",
                    )
                )

            elif precision in cfg:
                load_kwargs = {
                    "quantization_config": cfg[precision],
                    "device_map": "cuda",
                }

                # Use FP16 for non-quantized modules in the INT8 model.
                if precision == "int8":
                    load_kwargs["dtype"] = torch.float16

                self.models[precision] = AutoModelForCausalLM.from_pretrained(
                    MODEL,
                    **load_kwargs,
                )

            else:
                raise ValueError(
                    f"Unsupported precision: {precision}"
                )

    def generate(self, prompt, precision, max_new_tokens=128):
        if precision not in self.models:
            raise ValueError(
                f"Precision '{precision}' has not been loaded."
            )

        model = self.models[precision]

        text = self.tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tok(
            text,
            return_tensors="pt",
        ).to("cuda")

        torch.cuda.synchronize()
        start = time.perf_counter()

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_tokens = output.sequences[0][
            inputs.input_ids.shape[1]:
        ]

        scores = model.compute_transition_scores(
            output.sequences,
            output.scores,
            normalize_logits=True,
        )[0]

        confidence = float(scores.exp().mean())

        memory_mb = torch.cuda.memory_allocated() / 1e6

        return GenResult(
            self.tok.decode(
                generated_tokens,
                skip_special_tokens=True,
            ),
            elapsed,
            len(generated_tokens),
            memory_mb,
            confidence,
        )
