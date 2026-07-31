"""
build_sft_data.py — build QA and RAG SFT sets as ChatML `messages` JSONL.

sft.py consumes rows with a "messages" column ([{role, content}, ...]) and masks the
loss to assistant tokens only, so both tasks are expressed purely as data — no model
or training-loop changes.

  QA  : closed-book question answering. Teaches "answer the question directly".
  RAG : answer grounded in a supplied context, INCLUDING unanswerable cases so the
        model learns to abstain instead of hallucinating. That abstention behaviour is
        the whole point of squad_v2 over squad v1.

  python build_sft_data.py --task qa  --out /workspace/sft_data/qa.jsonl  --n 40000
  python build_sft_data.py --task rag --out /workspace/sft_data/rag.jsonl --n 40000
"""
import argparse
import json
import os
import random

from datasets import load_dataset

QA_SYS = "You are a helpful assistant. Answer the question accurately and concisely."
RAG_SYS = ("You are a helpful assistant. Answer the question using ONLY the provided context. "
           "If the context does not contain the answer, say that you don't know.")


def _row(msgs):
    return json.dumps({"messages": msgs}, ensure_ascii=False)


def build_qa(n, seed=1337):
    """sciq (science QA, with an explanation drawn from its support passage) +
    web_questions (short factoid). Mixed so the model sees both explanatory and
    terse answers rather than collapsing to one style."""
    out = []
    try:
        sciq = load_dataset("allenai/sciq", split="train")
        for r in sciq:
            q, a, sup = r.get("question"), r.get("correct_answer"), (r.get("support") or "").strip()
            if not q or not a:
                continue
            # keep the explanation short so the answer stays the dominant signal
            ans = f"{a}." if not sup else f"{a}.\n\n{sup[:400].rsplit('.', 1)[0]}."
            out.append([{"role": "system", "content": QA_SYS},
                        {"role": "user", "content": q.strip()},
                        {"role": "assistant", "content": ans}])
    except Exception as e:
        print("sciq unavailable:", e)

    try:
        wq = load_dataset("stanfordnlp/web_questions", split="train")
        for r in wq:
            q, answers = r.get("question"), r.get("answers") or []
            if not q or not answers:
                continue
            out.append([{"role": "system", "content": QA_SYS},
                        {"role": "user", "content": q.strip()},
                        {"role": "assistant", "content": ", ".join(answers)}])
    except Exception as e:
        print("web_questions unavailable:", e)

    random.Random(seed).shuffle(out)
    return out[:n]


ABSTAIN = "I don't know."


def build_rag(n, seed=1337, unans_frac=0.20):
    """squad_v2: context + question -> grounded answer, plus a MINORITY of unanswerable
    cases so abstention is available but not dominant.

    Two things are tuned deliberately, because the loss is token-level over assistant
    tokens and squad answers are very short (~3 tokens):

      1. ABSTAIN is short. A long fixed refusal ("I don't know — the context does not
         contain the answer.", ~12 tokens) contributes far more trained tokens per
         example than a real answer, so the model can cut its loss most cheaply by
         always refusing. That is exactly what happened on the first attempt.
      2. unans_frac is capped at 20% (squad_v2 is ~33% unanswerable). Combined with the
         short string this puts roughly 20%x2 vs 80%x3 token-units on refuse-vs-answer,
         so answering is the dominant signal.
    """
    ans_rows, unans_rows = [], []
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    for r in ds:
        ctx = (r.get("context") or "").strip()
        q = (r.get("question") or "").strip()
        texts = (r.get("answers") or {}).get("text") or []
        if not ctx or not q:
            continue
        if texts and texts[0].strip():
            row, bucket = texts[0].strip(), ans_rows
        else:
            row, bucket = ABSTAIN, unans_rows
        bucket.append([{"role": "system", "content": RAG_SYS},
                       {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
                       {"role": "assistant", "content": row}])

    rng = random.Random(seed)
    rng.shuffle(ans_rows); rng.shuffle(unans_rows)
    n_un = min(len(unans_rows), int(n * unans_frac))
    n_an = min(len(ans_rows), n - n_un)
    out = ans_rows[:n_an] + unans_rows[:n_un]
    rng.shuffle(out)
    print(f"rag: {len(out)} examples, {n_un} unanswerable ({n_un/max(1,len(out))*100:.1f}%), "
          f"abstain string = {ABSTAIN!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["qa", "rag"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = build_qa(a.n) if a.task == "qa" else build_rag(a.n)
    with open(a.out, "w") as f:
        for m in rows:
            f.write(_row(m) + "\n")
    print(f"wrote {len(rows)} examples -> {a.out}")
    if rows:
        print("--- sample ---")
        for m in rows[0]:
            print(f"  [{m['role']}] {m['content'][:180]}")


if __name__ == "__main__":
    main()
