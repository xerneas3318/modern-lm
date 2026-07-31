"""
publish_model.py — push one trained checkpoint to the Hub as a clean model repo.

Uploads EXTRACTED weights, not the raw training checkpoint: the raw .pt is 4.3GB
because it carries Muon momentum + AdamW moments + scheduler + loader state. What a
consumer needs is the state_dict + config (~1.9GB), so that is what ships as model.pt.

  python publish_model.py --ckpt X.pt --repo Xerneas3318/name --stage base [--public]
                          [--card-extra "..."] [--readme-only]
"""
import argparse
import json
import os
import tempfile

import torch
from huggingface_hub import HfApi

STAGES = {
    "base-stable": dict(
        title="modern-1024x24 — base (end of WSD stable phase)",
        blurb="Pretrained base captured at the **end of the WSD stable phase** (step 15000, "
              "just before LR decay begins at step 15259). Learning rate is still at maximum "
              "here, so this checkpoint is the right starting point for continued pretraining "
              "or your own annealing/decay schedule. For downstream use you probably want the "
              "decayed `modern-1024x24-base` instead."),
    "base": dict(
        title="modern-1024x24 — base (final, end of decay)",
        blurb="Pretrained base after the full WSD schedule including linear decay to zero "
              "(step 19072). This is the flagship base model."),
    "sft": dict(
        title="modern-1024x24 — SFT",
        blurb="Instruction-tuned from the final base on HuggingFaceTB/smol-smoltalk with ~15% "
              "synthetic arithmetic tool-call examples mixed in. Loss is masked to assistant "
              "tokens only (prompt and injected tool results are masked out)."),
    "sft-toolcall": dict(
        title="modern-1024x24 — SFT + tool-calling specialist",
        blurb="Continued from the SFT model with a 50% synthetic tool-call mix at a lower LR, "
              "specialising in emitting `<tool_call>expr</tool_call>` for arithmetic while "
              "retaining chat ability."),
    "sft-qa": dict(
        title="modern-1024x24 — SFT + question answering",
        blurb="Continued from the SFT model on closed-book QA (allenai/sciq + "
              "stanfordnlp/web_questions, 15.5k examples), specialising in answering questions "
              "directly and concisely. Being a ~481M model trained on ~10B tokens, its world "
              "knowledge is limited — for factual work prefer the RAG variant with retrieved "
              "context."),
    "sft-rag": dict(
        title="modern-1024x24 — SFT + RAG (grounded QA)",
        blurb="Continued from the SFT model on retrieval-augmented QA (rajpurkar/squad_v2, 40k "
              "examples). Answers strictly from a supplied context. **~34% of training examples "
              "are unanswerable**, so the model is explicitly taught to reply \"I don't know\" "
              "when the context lacks the answer rather than hallucinating. Prompt format:\n\n"
              "```\nContext:\n{your retrieved passages}\n\nQuestion: {question}\n```"),
}

CARD = """---
license: mit
library_name: pytorch
tags:
- causal-lm
- pretrained-from-scratch
- muon
- rope
- gqa
- swiglu
datasets:
- HuggingFaceFW/fineweb-edu
- codeparrot/codeparrot-clean
language:
- en
---

# {title}

{blurb}

~481M parameters, trained from scratch on a single H100 80GB.

## Architecture

| | |
| --- | --- |
| Parameters | 480.9M |
| Layers | 24 |
| Model dim | 1024 |
| Heads | 16 (GQA, `kv_group=4` → 4 KV heads) |
| Context | 2048 |
| Vocab | 50304 (GPT-2 BPE, padded from 50257) |

Modern components: RMSNorm (pre-norm), RoPE, SwiGLU FFN, grouped-query attention,
QK-norm, value embeddings, U-net style skip connections between layer halves, and a
final logit softcap (`15 * tanh(logits/15)`).

Optimizer: **Muon** (Newton-Schulz orthogonalisation) on hidden 2-D matrices, AdamW on
embeddings / lm_head / norms — embeddings are excluded from Muon by identity.

## Training

| | |
| --- | --- |
| Data | FineWeb-Edu `sample-10BT` (82%) + `codeparrot-clean` Python (18%) |
| Tokens | ~10.0B (19,073 steps x 524,288 tokens) |
| Schedule | WSD — 3% warmup, stable to step 15,259, linear decay to 0 |
| LR | Muon 0.02, AdamW 1.2e-3 |
| Precision | bf16 autocast |
| Hardware | 1x H100 80GB, ~132k tok/s, ~21h |

{extra}

## Usage

This is a raw PyTorch checkpoint, not a `transformers` model. You need `model.py`
from [xerneas3318/modern-lm](https://github.com/xerneas3318/modern-lm).

```python
import torch
from model import GPT, GPT2Config, build_enc

ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
model = GPT(GPT2Config(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()

enc = build_enc()   # GPT-2 BPE + reserved chat/tool special tokens
```

## Tokenizer

GPT-2 BPE with reserved special tokens in the padded vocab region (IDs 50257+):
`<|im_start|>`, `<|im_end|>`, `<pad>`, `<think>`, `</think>`, `<tool_call>`,
`</tool_call>`, `<tool_result>`, `</tool_result>`. These are untrained during
pretraining (they never occur in the corpus) and are learned at SFT.

## Limitations

Trained on ~10B tokens at 481M parameters — roughly compute-optimal for its size, but
small by modern standards. Expect factual unreliability and limited reasoning. No
safety tuning of any kind. The base models are next-token predictors and are not
instruction-following.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--card-extra", default="")
    ap.add_argument("--readme-only", action="store_true",
                    help="upload just the card (used to smoke-test write access)")
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)

    meta = STAGES[a.stage]
    card = CARD.format(title=meta["title"], blurb=meta["blurb"], extra=a.card_extra)

    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "README.md")
        with open(rp, "w") as f:
            f.write(card)
        api.upload_file(path_or_fileobj=rp, path_in_repo="README.md",
                        repo_id=a.repo, repo_type="model")
        print(f"uploaded README.md -> {a.repo}")

        if a.readme_only:
            return

        c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        clean = {"model": c["model"], "config": c["config"], "step": c.get("step")}
        mp = os.path.join(td, "model.pt")
        torch.save(clean, mp)
        raw_gb = os.path.getsize(a.ckpt) / 1e9
        new_gb = os.path.getsize(mp) / 1e9
        print(f"extracted weights: {raw_gb:.2f}GB raw -> {new_gb:.2f}GB clean")

        cp = os.path.join(td, "config.json")
        with open(cp, "w") as f:
            json.dump(c["config"], f, indent=2)
        api.upload_file(path_or_fileobj=cp, path_in_repo="config.json",
                        repo_id=a.repo, repo_type="model")

        api.upload_file(path_or_fileobj=mp, path_in_repo="model.pt",
                        repo_id=a.repo, repo_type="model")
        print(f"uploaded model.pt -> https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
