#!/usr/bin/env python3
"""
run_4c_extractor.py   (plain‑text version)
"""

import argparse, glob, os, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ──────────────────────────────────────────────────────────────
def load_model(name: str, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    return tok, mdl


def chat_prompt(system: str, user: str, tok):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(tok, mdl, prompt, do_sample=False):
    cfg = GenerationConfig(
        do_sample=do_sample,
        temperature=0.7 if do_sample else None,
        max_new_tokens=2048,
    )
    # ids = tok(prompt, return_tensors="pt").to(mdl.device)
    # out = mdl.generate(**ids, generation_config=cfg)
    # return tok.decode(out[0], skip_special_tokens=True).split("</assistant>")[-1].strip()
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    input_len = inputs.input_ids.shape[1]          # ← length of prompt
    output_ids = mdl.generate(**inputs, generation_config=cfg)

    # keep only newly generated tokens
    new_tokens = output_ids[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--transcripts", required=True, nargs="+")
    p.add_argument("--outdir", default="txt_out")
    p.add_argument("--model", default="nousresearch/meta-llama-3-8b-instruct")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--sample", action="store_true",
                   help="enable sampling (stochastic) generation")
    args = p.parse_args()

    paths = sorted(sum((glob.glob(pat) for pat in args.transcripts), []))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        sys.exit("No transcript files found.")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    tok, mdl = load_model(args.model, getattr(torch, args.dtype))
    sys_prompt = Path(args.prompt).read_text(encoding="utf‑8")

    for idx, file in enumerate(paths, 1):
        print(f"[{idx}/{len(paths)}] {Path(file).name}")
        transcript = Path(file).read_text(encoding="utf‑8")
        prompt = chat_prompt(sys_prompt, transcript, tok)
        reply = generate(tok, mdl, prompt, do_sample=args.sample)

        out_path = Path(args.outdir) / f"{Path(file).stem}.txt"
        out_path.write_text(reply, encoding="utf‑8")
        print(f"✓ saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
