"""
prepare_data.py -- pull a HuggingFace dataset, tokenize with the GPT-2 BPE, and
write uint16 .npy shards matching the existing edu_fineweb10B format so the
training data loader can read them.

Format (identical to Karpathy's fineweb.py, which produced edu_fineweb10B):
  - dtype uint16, one flat token array per shard
  - each document prefixed with <|endoftext|> (50256) as the separator
  - shard 0 -> val, shards 1.. -> train
  - filenames: <prefix>_val_000000.npy, <prefix>_train_000001.npy, ...

FineWeb-Edu 10B is already prepared at /mnt/datasets/edu_fineweb10B, so the main
use here is a CODE dataset (for the "write Python" goal). Run it into its own dir,
then the multi-source loader mixes edu_fineweb10B + code by ratio at train time.

Examples:
  # Python code (adjust dataset/config/text-col to one you have access to):
  python prepare_data.py --dataset codeparrot/github-code-clean --name Python-all \\
      --text-col code --out-dir /mnt/datasets/code_python --prefix code \\
      --total-tokens 2_000_000_000

  # tiny smoke test (writes one small shard and stops):
  python prepare_data.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \\
      --text-col text --out-dir /tmp/smoke --prefix smoke \\
      --total-tokens 2_000_000 --shard-size 1_000_000 --val-shards 0

HF auth: uses the HF_TOKEN env var / cached login automatically.
"""

import os
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]   # 50256, document separator


def tokenize_doc(doc, text_col):
    """One document -> uint16 token array, prefixed with EOT."""
    text = doc.get(text_col)
    if not text:
        return np.empty((0,), dtype=np.uint16)
    ids = [EOT]
    ids.extend(enc.encode_ordinary(text))
    arr = np.array(ids, dtype=np.uint32)
    assert (arr < 2**16).all(), "token id out of uint16 range"
    return arr.astype(np.uint16)


def write_shard(path, tokens):
    np.save(path, tokens)
    print(f"  wrote {path}  ({len(tokens):,} tokens)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. codeparrot/github-code-clean")
    ap.add_argument("--name", default=None, help="dataset config/name, e.g. Python-all or sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text", help="field holding the text (text / content / code)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True, help="shard filename prefix, e.g. code")
    ap.add_argument("--total-tokens", type=lambda s: int(float(s.replace('_', ''))), default=2_000_000_000)
    ap.add_argument("--shard-size", type=lambda s: int(float(s.replace('_', ''))), default=100_000_000)
    ap.add_argument("--val-shards", type=int, default=1, help="how many leading shards are 'val'")
    ap.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_shards = -(-args.total_tokens // args.shard_size)   # ceil
    print(f"dataset={args.dataset} name={args.name} split={args.split} text_col={args.text_col}")
    print(f"target {args.total_tokens:,} tokens -> up to {total_shards} shards of {args.shard_size:,} "
          f"({args.val_shards} val), out={args.out_dir}, nproc={args.nproc}", flush=True)

    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)

    buf = np.empty((args.shard_size,), dtype=np.uint16)
    count = 0
    shard_index = 0
    written_tokens = 0
    fn = partial(tokenize_doc, text_col=args.text_col)

    with mp.Pool(args.nproc) as pool:
        for tokens in pool.imap(fn, ds, chunksize=16):
            if len(tokens) == 0:
                continue
            if count + len(tokens) < args.shard_size:
                buf[count:count + len(tokens)] = tokens
                count += len(tokens)
            else:
                split = "val" if shard_index < args.val_shards else "train"
                path = os.path.join(args.out_dir, f"{args.prefix}_{split}_{shard_index:06d}.npy")
                take = args.shard_size - count
                buf[count:] = tokens[:take]
                write_shard(path, buf)
                written_tokens += args.shard_size
                shard_index += 1
                if shard_index >= total_shards:
                    break
                # carry the remainder into the next shard
                rest = tokens[take:]
                buf[:len(rest)] = rest
                count = len(rest)

        # trailing partial shard (only if we did not hit the shard cap)
        if shard_index < total_shards and count > 0:
            split = "val" if shard_index < args.val_shards else "train"
            path = os.path.join(args.out_dir, f"{args.prefix}_{split}_{shard_index:06d}.npy")
            write_shard(path, buf[:count])
            written_tokens += count

    print(f"done: {shard_index + (1 if count > 0 and shard_index < total_shards else 0)} shards, "
          f"~{written_tokens:,} tokens in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
