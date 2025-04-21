# fine_tune_4c_lora.py
"""
Supervised fine‑tuning (QLoRA) for 4‑C contextualisation extractor
------------------------------------------------------------------
* Base  : nousresearch/meta‑llama‑3‑8b‑instruct (8 k context)
* GPU   : single RTX 3090 Ti 24 GB
* TRL   : 0.16 or newer           (uses SFTTrainer + SFTConfig)
* Data  :   data/
              ├─ transcripts/  A001.txt …
              └─ targets/      A001.txt …   # ground‑truth extraction
            Prompt.txt          # 4‑C system prompt
"""

import argparse, warnings, json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LEN = 4096


# ──────────────────────────────────────────────────────────────
def make_chat_string(tok, sys_prompt: str, user: str, assistant: str) -> str:
    """Return a full ChatML string ready for tokenisation."""
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def build_dataset(data_root: Path, sys_prompt: str, tok) -> Dataset:
    """Load transcript/target pairs → Dataset with a single 'text' column."""
    rows = []
    for tr in sorted((data_root / "transcripts").glob("*.txt")):
        tgt = data_root / "targets" / tr.name
        if not tgt.exists():
            warnings.warn(f"✗ missing target for {tr.name}; skipping")
            continue
        user_txt = tr.read_text(encoding="utf-8")
        ass_txt = tgt.read_text(encoding="utf-8")
        chat_str = make_chat_string(tok, sys_prompt, user_txt, ass_txt)
        rows.append({"text": chat_str})
    if not rows:
        raise RuntimeError("No usable transcript/target pairs found.")
    return Dataset.from_list(rows)


# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--model",
                    default="nousresearch/meta-llama-3-8b-instruct")
    ap.add_argument("--output_dir", type=Path,
                    default=Path("4c_8b_lora"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    sys_prompt = args.prompt.read_text(encoding="utf-8").strip()

    # tokenizer first – needed for chat‑template
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,  # Use fast tokenizer for better performance
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Avoid issues with fp16 inference

    dataset = build_dataset(args.data_root, sys_prompt, tokenizer)

    # 4‑bit quantisation config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map={"":0}, # Use single GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Ensure model uses bfloat16
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        remove_unused_columns=False,
        dataset_text_field="text",  # Specify the text field for SFTTrainer
        label_names=["labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Clear GPU memory
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_cfg,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
