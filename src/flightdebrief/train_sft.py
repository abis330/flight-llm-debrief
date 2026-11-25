"""LoRA / QLoRA supervised fine-tuning with TRL.

Real run (single T4/L4/A10 GPU, ~20-40 min for Qwen2.5-1.5B on a few thousand examples):
    python -m flightdebrief.train_sft --config configs/qwen2.5-1.5b-lora.yaml

Offline smoke test (no downloads; builds a tiny random model + tokenizer, runs 5 steps):
    python -m flightdebrief.train_sft --config configs/smoke.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _tiny_model_and_tokenizer(texts: list[str], out: Path):
    """Build a tiny Qwen2-architecture model and a BPE tokenizer from scratch (CI smoke tests)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special = ["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=special))
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>",
                                     pad_token="<|endoftext|>", bos_token=None)
    hf_tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    cfg = Qwen2Config(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      eos_token_id=hf_tok.eos_token_id, pad_token_id=hf_tok.pad_token_id)
    model = Qwen2ForCausalLM(cfg)
    out.mkdir(parents=True, exist_ok=True)
    hf_tok.save_pretrained(out)
    model.save_pretrained(out)
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data = load_dataset("json", data_files={"train": cfg["train_file"], "validation": cfg["val_file"]})
    data = data.remove_columns([c for c in data["train"].column_names if c not in ("prompt", "completion")])
    if cfg.get("max_train_examples"):
        data["train"] = data["train"].select(range(min(cfg["max_train_examples"], len(data["train"]))))
    model_name = cfg["base_model"]
    if a.smoke:
        texts = [m["content"] for ex in data["train"] for m in ex["prompt"] + ex["completion"]]
        model_name = _tiny_model_and_tokenizer(texts, Path(cfg["output_dir"]) / "tiny-base")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16)
    dtype = torch.bfloat16 if cfg.get("bf16") else (torch.float16 if cfg.get("fp16") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, quantization_config=quant)

    lora = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                      lora_dropout=cfg.get("lora_dropout", 0.05), task_type="CAUSAL_LM",
                      target_modules=cfg.get("target_modules", "all-linear"))
    kw = dict(
        output_dir=cfg["output_dir"], num_train_epochs=cfg.get("epochs", 2), max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("batch_size", 4), gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=float(cfg.get("lr", 2e-4)), lr_scheduler_type="cosine",
        logging_steps=cfg.get("logging_steps", 10), eval_strategy="steps", eval_steps=cfg.get("eval_steps", 100),
        save_strategy="steps", save_steps=cfg.get("eval_steps", 100), save_total_limit=2,
        load_best_model_at_end=not a.smoke, metric_for_best_model="eval_loss",
        bf16=bool(cfg.get("bf16")), fp16=bool(cfg.get("fp16")), max_length=cfg.get("max_seq_len", 2048),
        completion_only_loss=True, gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to=cfg.get("report_to", "none"), seed=cfg.get("seed", 0), use_cpu=a.smoke,
    )
    # transformers/TRL renamed a few arguments across releases -> adapt instead of pinning
    import inspect

    params = inspect.signature(SFTConfig.__init__).parameters
    kw["warmup_ratio" if "warmup_ratio" in params else "warmup_steps"] = 0.05
    dropped = [k for k in kw if k not in params]
    if dropped:
        print(f"[train_sft] ignoring args unsupported by this TRL version: {dropped}")
    args = SFTConfig(**{k: v for k, v in kw.items() if k in params})
    trainer = SFTTrainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["validation"],
                         processing_class=tok, peft_config=lora)
    trainer.train()
    final = Path(cfg["output_dir"]) / "final_adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    meta = {"base_model": model_name, "config": cfg, "log_history": trainer.state.log_history}
    (final / "training_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"adapter saved to {final}")
    if cfg.get("merge_and_save"):
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(cfg["output_dir"]) / "merged")
        tok.save_pretrained(Path(cfg["output_dir"]) / "merged")


if __name__ == "__main__":
    main()
