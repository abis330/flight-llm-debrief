"""Generators: fine-tuned HF model (+LoRA adapter), any OpenAI-compatible server (vLLM / Ollama /
TGI), or Anthropic Claude as a frontier zero-shot baseline / teacher."""
from __future__ import annotations

import json
import os

from .schema import SYSTEM_PROMPT


class HFGenerator:
    def __init__(self, model_dir: str, adapter: str | None = None, max_new_tokens: int = 600, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or model_dir)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(dev)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model, self.dev, self.max_new = model.eval(), dev, max_new_tokens

    def __call__(self, ex: dict) -> str:
        ids = self.tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAICompatGenerator:
    """e.g. `vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora --lora-modules debrief=outputs/final_adapter`"""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "debrief", api_key: str = "EMPTY"):
        import httpx

        self.http, self.url, self.model, self.key = httpx.Client(timeout=120), base_url.rstrip("/"), model, api_key

    def __call__(self, ex: dict) -> str:
        r = self.http.post(f"{self.url}/chat/completions", headers={"Authorization": f"Bearer {self.key}"},
                           json={"model": self.model, "messages": ex["prompt"], "temperature": 0, "max_tokens": 700})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicGenerator:
    """Frontier-model baseline (zero/few-shot). Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, few_shot: list[dict] | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.shots = []
        for ex in few_shot or []:
            self.shots += [{"role": "user", "content": ex["prompt"][-1]["content"]},
                           {"role": "assistant", "content": ex["completion"][0]["content"]}]

    def __call__(self, ex: dict) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=900, temperature=0, system=SYSTEM_PROMPT,
                                          messages=self.shots + [{"role": "user", "content": ex["prompt"][-1]["content"]}])
        return "".join(b.text for b in msg.content if b.type == "text")


def debrief_flight(lake_root: str, flight_id: str, generator, baseline_path: str) -> dict:
    from flightlake import Lake

    from .dataset import to_example
    from .digest import build_digest, load_baseline
    from .schema import parse_debrief

    d = build_digest(Lake(lake_root), flight_id, load_baseline(baseline_path))
    text = generator(to_example(d, None))
    obj, err = parse_debrief(text)
    return {"flight_id": flight_id, "digest": d["text"], "debrief": json.loads(obj.model_dump_json()) if obj else None,
            "error": err, "raw": text}
