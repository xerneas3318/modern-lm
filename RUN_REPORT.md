# modern-lm — full run report (2026-07-30 → 2026-07-31)

Everything done to get a 481M-parameter LM pretrained from scratch, post-trained four
ways, and published — including every bug hit, what the symptom looked like, why it
happened, and how it was fixed. Written so the next run does not repeat any of it.

**Outcome:** 481M params, 10.0B tokens, final val loss **2.7618**, 21h13m on one H100 at
~131k tok/s sustained. Six models published. Zero crash-restarts during the run.

---

## 0. TL;DR of what actually bit us

| # | Problem | Symptom | Root cause | Fix |
| --- | --- | --- | --- | --- |
| 1 | Data build made no progress | 10 min, zero shards | `os.cpu_count()`=224 (host) vs 23.8-core cgroup quota → 222 workers thrashing | `--nproc 22` → whole build in 26 min |
| 2 | Runbook's `--batch-size 64` | CUDA OOM at 78.7 GiB | Sized for 96GB cards; also 32 and 16-uncompiled OOM | `--batch-size 16` **with** `torch.compile` |
| 3 | Training died instantly | `ModuleNotFoundError: tqdm` | bare `torchrun` = system py3.12/torch2.8, not the venv | `.venv/bin/torchrun` |
| 4 | ETA read `284h` then `~950h` | nonsense ETA | averaged compile time into every step; then fallback used compile-laden `dt` | EMA of per-step time + `(compiling)` sentinel |
| 5 | End-of-stable ckpt would vanish | — (caught before it happened) | `--keep-ckpts 3` deletes it at step 18000 | `preserve_ckpts.sh` daemon |
| 6 | RAG model always said "I don't know" | loss 0.459 (*lowest of any stage*) | long fixed refusal dominated token-level loss (~67% of trained tokens) | short `"I don't know."` + cap unanswerable at 20% → 19.6% |
| 7 | `datasets` missing | would break data build | absent from `pyproject.toml` despite `prepare_data.py` importing it | `uv add datasets huggingface_hub` |
| 8 | Bare dataset names rejected | `HfUriError` | `datasets 5.x` requires `namespace/name` | `rajpurkar/squad_v2` etc. |
| 9 | Local JSONL unusable for SFT | — | `sft.py` only accepted HF repo ids | added local-file support + epoch restart (PEP 479) |
| 10 | My own shell killed twice | exit 144 | `pkill -f "[p]attern"` also matched the *same* command line | kill by PID |
| 11 | A training run died silently mid-write | GPU idle, no traceback, checkpoint truncated at exactly 512 MiB | **volume quota exhausted** — and `df` reports MooseFS *cluster* free space (305T), not the provisioned quota | freed 37 GB; added `quota_watch.sh` that probes a 600 MB write every 2 min |
| 12 | Watchdog missed #11 entirely | no alert while the GPU sat idle | it exited on "pipeline COMPLETE and no trainer running" — true during the gap between two manual runs | only stop on an explicit `watchdog.stop` file |
| 13 | Both RAG evals crashed | `IsADirectoryError` | my eval script only expanded a run dir into a checkpoint if the path ended in `/` or had a `*` | use `os.path.isdir()` |
| 14 | Flagship model published private | — (caught on verification) | `create_repo(..., exist_ok=True)` does **not** change visibility of an existing repo | explicit `update_repo_settings(private=False)` |

---

## 1. Environment: what the pod actually was

The runbook assumed **2x RTX Pro 6000 (96GB)**. Reality, verified with `nvidia-smi -L`:

- **1x H100 80GB HBM3**, `RUNPOD_GPU_COUNT=1`, torch 2.13.0+cu130, bf16 supported
- 224 host CPU cores visible, but the container cgroup quota is **23.8 cores**
  (`/sys/fs/cgroup/cpu.max = 2380000 100000`)
- RunPod pod `8kvhsuqhpbjuu0`, $2.99/hr

Consequences: `--nproc_per_node=1` (not 2), roughly double the wall clock, and the
batch-size problem in §3.

### Persistence topology (the thing that decides where everything lives)

```
/workspace   mfs#…runpod.net  fuse   <- NETWORK VOLUME, persistent, survives stop/start
/root        overlay                 <- EPHEMERAL, wiped on stop/start
/usr /opt    overlay                 <- EPHEMERAL
```

Verified the volume is **exec-capable** (`nosuid,nodev` but not `noexec`) and fast
(542 MB/s), so the venv and toolchains can live there. Everything that must survive was
put on `/workspace`. One caveat found: **`chmod` does not stick** on this MooseFS mount —
the secrets file stays `rw-rw-rw-` regardless.

Corollary that mattered later: `gh`'s OAuth token lives in `/root/.config/gh/hosts.yml`,
i.e. on the **ephemeral** disk. A pod restart would silently destroy GitHub auth. This is
why artifacts were pushed as they became ready rather than all at the end.

---

## 2. Bug #1 — the data build was crippled by a 10x CPU oversubscription

**Symptom.** First `prepare_data.py` run produced *zero* shards in 10 minutes and looked
hung.

**Diagnosis.** The log header printed `nproc=222`. `prepare_data.py:79` defaults to
`os.cpu_count() - 2`. Inside a container, `os.cpu_count()` reports the **host's** 224
cores, not the cgroup quota. So it forked 222 tokenizer workers onto 23.8 usable cores —
each loading its own tiktoken encoder — and spent all its time context-switching.

This is a general containerised-Python trap: `os.cpu_count()` and even
`len(os.sched_getaffinity(0))` both returned 224 here. Only `cpu.max` tells the truth.

**Fix.** `--nproc 22`. The full build (10B FineWeb-Edu + 2B code) then took **25.7 min**
at ~4.7 shards/min, producing 23 GB:

```
99 x edufineweb_train_*.npy + 1 val   (10.0B tokens)
19 x code_train_*.npy + 1 val          (2.0B tokens)
```

Shards validated: `uint16`, all ids < 50257, EOT(50256) separators present.

**Aside — "can't we shard on the GPU?"** No. GPT-2 BPE is a sequential merge loop with
data-dependent branching over a 50k-rank table — poor SIMT fit, and no drop-in GPU
implementation produces bit-identical ids. RAPIDS' GPU tokenizer is WordPiece, a
different algorithm, and would silently corrupt the corpus. The real lever is CPU quota.

---

## 3. Bug #2 — batch size 64 does not fit, and neither does 32 or 16 (uncompiled)

The runbook says `--batch-size 64`. Measured on this 80GB card, real 480M config, seq 2048:

| B | grad_accum | compile | result | peak |
| --- | --- | --- | --- | --- |
| 64 | 4 | off | **OOM** | 78.7 GiB |
| 32 | 8 | off | **OOM** | 78.5 GiB |
| 16 | 16 | off | **OOM** | 74.4 GiB |
| **16** | **16** | **on** | **FITS** | **62.7 / 79.2 GiB** |

**Why it is so activation-heavy.** Two things compound:

1. `RMSNorm` (`model.py:128`) and `apply_rope` (`model.py:69`) both do `x.float()`,
   materialising full **fp32** `(B,T,C)` tensors that are kept for backward — ×24 layers.
2. The logit path is enormous: `lm_head` output is `(B,T,50304)`, then the softcap
   `cap*tanh(logits/cap)` (`model.py:280`) creates several more copies of it, then
   `cross_entropy` upcasts to fp32 again. At B=16 that alone is ~26 GB.

`torch.compile` fuses exactly these chains, saving ~10 GiB — which is the entire
difference between OOM and fitting. **Never pass `--no-compile` on the real run.**

Note `train.py:50`'s own default is already `32` ("dropped 64→32 for 2048 ctx"), so the
runbook's 64 was above even the code's own default.

Crucially `total_batch_size=524288` is unchanged at every B (grad accumulation
compensates), so **training dynamics are identical** to the validated recipe — only wall
clock changes. The `assert total_batch_size % (B*T*world) == 0` requires `B*world` to
divide 256, which 16 does.

---

## 4. Bug #3 — bare `torchrun` is the wrong Python

First real launch died in seconds:

```
File "train.py", line 29, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named 'tqdm'
```

`torchrun` resolved to `/usr/local/bin/torchrun` → system Python 3.12 with torch 2.8.0,
not the uv venv (Python 3.13.8, torch 2.13.0+cu130). Every smoke test had passed because
they all invoked `.venv/bin/python` directly and never went through `torchrun`.

**Fix:** `/workspace/modern-lm/.venv/bin/torchrun`. `launch_train.sh` hardcodes it with a
comment, because this is easy to reintroduce.

---

## 5. Logging: cadence, ETA, and two self-inflicted regressions

Originally `train.py` wrote a line **every step**, opening/closing the log each time.
Measured cost on the network volume: **2.23 ms/call** (vs 0.04 ms on local disk — 56x
slower) = **42.5 s across the whole 21h run**. So it was never a bottleneck; the change
was for readability.

Added `--log-every` (default 100, run at 10 → a line every ~40 s) and progress+ETA:

```
15200 train 2.931400 norm 0.4412 lr_muon 2.00e-02 lr_adam 1.20e-03 tok_s 132589 | 15201/19073 (79.7%) | elapsed 16h43m | eta 4h15m
```

**Regression A.** First ETA used `elapsed / steps_done`, which amortises the ~3 min
`torch.compile` across every remaining step → first line read **`eta 284h31m`**.
Fixed with an EMA of true per-step `dt`, seeded *after* the first step.

**Regression B.** That left `ema_dt` undefined on step 0, falling back to the
compile-laden `dt` → would have printed **~950h**. Fixed by printing `eta (compiling)`
instead of a fabricated number.

Result: stable and agreeing with tqdm's independent estimate:
`step 10: eta 20h52m`, `step 20: eta 20h54m`.

**Important coupling:** `tr_loss.append()` was deliberately left running **every step**.
It is in-memory (no I/O) and rides along inside each checkpoint, so `loss.png` keeps full
19,073-point resolution even though the log is now sparse. Thinning it would have
silently reduced the published loss curve to ~190 points. Consequently `loss.png` is
built from `torch.load(ckpt)["tr_loss"]`, **not** by parsing `train.log` as train.md §5b
suggests.

**Cost of applying these:** three deliberate restarts before the first checkpoint existed,
so ~400 steps (~26 min) were redone. Restarting *after* a checkpoint costs almost nothing;
restarting *before* the first one costs everything since step 0.

---

## 6. Checkpoint rotation would have destroyed a requested artifact

`--keep-ckpts 3` keeps only the newest three. The WSD phase boundaries:

```
warmup  0      -> 572
stable  572    -> 15259     (LR at max)
decay   15259  -> 19073     (LR -> 0)
```

The last checkpoint inside the stable phase is `ckpt_015000.pt` — and it gets **deleted**
when `ckpt_018000.pt` is written. That is a ~3.3 h window. `preserve_ckpts.sh` was written
to copy milestones to `/workspace/keep/` as soon as they appear (waiting for the file size
to settle first, since these are 4.3 GB writes).

Verified the rescued checkpoint is genuinely pre-decay:

```
step 15000, 480.9M params, 15001 tr_loss points
sched[0] last_lr: [0.02]              <- Muon still at MAX
sched[1] last_lr: [0.0012, 0.0012]    <- AdamW still at MAX
```

Rotation was later observed doing exactly what was predicted (`ckpt_001000`/`002000`
disappeared), confirming this was not hypothetical.

---

## 7. Pretraining results

```
run modern_1024x24 | val loss 2.7618 | 130869 tok/s | elapsed 21h13m
```

| phase | step | val |
| --- | --- | --- |
| init | 0 | 11.0320 |
| — | 1000 | ~3.66 |
| stable, mid | 9000 | 3.142 |
| **end of stable** | 15000 | **3.0997** |
| — | 17750 | 2.9147 |
| **end of decay (final)** | 19072 | **2.7618** |

Two observations worth recording:

- **The stable phase looks like a plateau and that is correct.** Between steps 4750 and
  6250 val moved only 3.182 → 3.180. At constant max LR, WSD progresses slowly by design.
- **The decay phase delivered −0.34** (3.0997 → 2.7618) in the final 3,814 steps — 4x the
  rate of the stable phase in a third of the steps. This is why `base-stable` and `base`
  are genuinely different models, not just early/late copies.

Train loss looks noisier than val because 18% of batches are **code**, which carries a
systematically different loss than web text. Val is FineWeb-only, hence the clean signal.

GPU: 99%+ util and ~693 W (card is rated ~700 W) for the entire run. Every sub-80%
sample in the history is attributable to either a deliberate restart or a 4.3 GB
checkpoint write (~1 sample each, memory stays held).

Generation quality progression (same prompt, greedy-ish sampling):

- step 4000: grammatical clauses, consistent proper nouns, still nonsense
- step 8000: document *scaffolding* — headed sections, lists, dates, prices; correctly
  emits `<|endoftext|>` as a document boundary

---

## 8. Post-training: four stages

All four ran automatically after pretraining, chained by `pipeline.sh`:

| stage | from | data | steps | loss | time |
| --- | --- | --- | --- | --- | --- |
| `sft1` | base | smol-smoltalk + 15% tool-call | 10,000 | 1.994 → 1.489 | 18.5 min |
| `sft2_toolcall` | sft1 | 50% tool-call | 4,000 | 1.753 → 1.392 | 6.9 min |
| `sft3_qa` | sft1 | sciq + web_questions (15,457) | 3,000 | 2.755 → 1.378 | 4.4 min |
| `sft4_rag` | sft1 | squad_v2 (40,000) | 4,000 | see §9 | 5.2 min |

QA and RAG branch from the **chat** model, not the tool-call one, so each specialist is an
independent head on the same instruction-tuned base.

Deliberately **not** `--toolcall-frac 1.0` for the tool-call stage: at 1.0 `sft.py:104`
skips the chat dataset entirely and trains on synthetic arithmetic only, producing a model
that calculates but cannot converse.

### Enabling changes to `sft.py` (data loading only — model/training loop untouched)

1. Accept a local `.jsonl` (`load_dataset("json", data_files=...)`) so custom QA/RAG mixes
   can be used without publishing them to the Hub.
2. **Restart the iterator on exhaustion.** A finite local file ends mid-generator, and
   letting `StopIteration` escape a generator is a `RuntimeError` under **PEP 479** — this
   would have killed each run at the end of epoch 1.

### Tool-calling: verified working, with an honest limit

```
Q: What is 47 * 89?
A: I'll compute 47 * 89.
   <tool_call>47*89</tool_call>
   <tool_result>4183</tool_result>     <- safe_calc ran, correct
   The answer is 4183.<|im_end|>       <- consumed result, stopped cleanly
```

The *mechanism* is reliable. Choosing the right *operation* is not: "23 boxes of 17
apples" produced `<tool_call>23+17</tool_call>` instead of `23*17`. A 481M reasoning
limit, not a formatting failure. `safe_calc` only evaluates arithmetic and rejects
names/calls/imports — a calculator, never code execution.

`sft1` (15% mix) fell into repetition loops on the same word problem while
`sft2_toolcall` (50%) stayed on task, so the specialisation demonstrably worked.

---

## 9. Bug #6 — the RAG model silently collapsed to always abstaining

**The most instructive failure of the run.**

**Symptom.** Trained "successfully": exit 0, and loss fell 2.086 → **0.459 — the lowest of
any stage**. By every number on the dashboard it was the best-trained model. But asked a
question whose answer is *verbatim in the context*:

```
Context: The Eiffel Tower ... stands 330 metres tall ...
Q: How tall is the Eiffel Tower?
A: "I don't know — the context does not contain the answer."
```

It abstained on everything.

**Root cause.** The SFT loss is token-level over assistant tokens
(`(loss_tok * m).sum() / m.sum()`). My abstention string was long and squad answers are
short:

```
abstention:  33.6% of examples x ~12 tokens = 4.03 token-units
answerable:  66.4% of examples x ~3  tokens = 1.99 token-units
→ ~67% of ALL trained tokens were the fixed refusal string
```

So the cheapest way to minimise loss was to emit that string unconditionally — and doing
so drove loss *lower* than actually learning the task. **The low loss was the symptom, not
the reassurance.**

**Fix (both levers).**

1. Shorten the refusal to `"I don't know."`
2. Cap unanswerable at 20% (squad_v2 is natively ~33.6%)

Verified empirically before retraining — abstention fell from ~67% to **19.6%** of trained
tokens.

**Lesson.** Loss going down is not evidence a model works. Any class with a long, fixed,
frequent target can hijack a token-level objective. The only thing that caught this was
querying the model with a case whose correct answer was known.

### 9b. Then tuning it properly — three variants on one fixed suite

All scored greedily on the *same* 10 cases (7 answerable, 3 unanswerable), reporting the
two halves separately because "always abstain" and "never abstain" are opposite failure
modes that a single aggregate hides:

| variant | config | answerable | abstain | total |
| --- | --- | --- | --- | --- |
| v0 | 4k, lr 5e-5, 33.6% unans, long refusal | — | — | **collapsed** (loss 0.459, useless) |
| v1 | 4k, lr 5e-5, 20% unans, short refusal | 4/7 | 2/3 | 6/10 |
| v2 | **12k, lr 1e-4** | — | — | worse, degenerate (`"ChI don't know."`) |
| **v3 (published)** | **12k, lr 5e-5** | **6/7** | 2/3 | **8/10** |

v2 is a clean negative result: at lr 1e-4 the *training* loss was also higher (1.657 vs
1.431), so the LR was the problem, not the step count. Holding lr at 5e-5 and tripling
steps to 2.4 epochs gave the win. Remaining failures are honest limits — a wrong span for
"How tall…" and answering "Peru" for a question the context does not cover.

### 9c. The run that died mid-tuning

v3's first attempt died at step 6000 with no traceback and a checkpoint truncated at
exactly 536,870,912 bytes (512 MiB). Not an OOM (1.9 TB RAM free) and not a Python error.
The cause was the **volume quota** (§0 #11): `torch.save` ran out of quota mid-write and
the process was killed. `df` said 305T free the entire time.

Three defences failed simultaneously: the quota was invisible, the write failed silently,
and the GPU watchdog had self-terminated minutes earlier. All three are now fixed.

---

## 10. Process mistakes (mine) worth not repeating

- **`pkill -f "[p]attern"` killed my own shell twice** (exit 144). The bracket trick stops
  the *pattern* self-matching, but the same command line also contained the literal target
  string elsewhere (e.g. `... bash /workspace/pipeline.sh`), which the regex matched.
  Kill by PID.
- **Edited `pipeline.sh` while bash was executing it.** Bash reads scripts lazily by byte
  offset; inserting ~30 lines mid-file can make the running shell resume at a corrupted
  offset. Restarted the supervisor so it re-read from the top.
- **Deliberate restarts consumed the supervisor's crash budget** (`MAX_RESTARTS=10`),
  leaving fewer slots for real failures. Restarted the supervisor to reset the counter.
- **Declared "two things did not happen"** when checking 84 s after completion — the
  pipeline had already advanced and the preserve daemon polls on a 60 s cycle. Both were
  fine.
- **Misattributed tool-call demos.** `tail -8` of the pipeline log showed the *QA and RAG*
  runs' demos (every `sft.py` run prints one) and I briefly concluded tool-calling was
  broken. It was not — checking the actual models showed a perfect tool loop.
- **`until grep -q ...` on an append-only log matched stale content** from a previous run
  twice, returning instantly. Wait on log *growth* or a run-unique marker instead.

---

## 11. Publishing

Six models, all public:

| repo | from | notes |
| --- | --- | --- |
| `modern-1024x24-base-stable` | step 15000 | end of WSD stable, LR still at max |
| `modern-1024x24-base` | step 19072 | end of decay, val **2.7618** |
| `modern-1024x24-sft` | sft1 | chat + 15% tool-call |
| `modern-1024x24-sft-toolcall` | sft1 → 50% tool-call | tool loop verified end-to-end |
| `modern-1024x24-sft-qa` | sft1 | format reliable, facts are not |
| `modern-1024x24-sft-rag` | sft1, 12k @ 5e-5 | **8/10** grounded QA |

**GitHub push gotcha.** The first push was rejected with
`push declined due to email privacy restrictions` — the account has "Block command line
pushes that expose my email" enabled, so commits authored from the real address are
refused. Fixed by authoring as `<id>+<login>@users.noreply.github.com`. Local
`git config user.email` was left as the user set it.

Uploads ship **extracted** weights, not raw checkpoints. A raw `ckpt_*.pt` is 4.31 GB
because it carries Muon momentum buffers, AdamW moments, scheduler and dataloader state.
Consumers need `state_dict + config`:

```
raw checkpoint : 4.31 GB
clean model.pt : 1.92 GB   (45%)
→ 9.6 GB less upload across the model set
```

The full training state stays on `/workspace` for resuming. Each repo gets a model card
with the architecture table, the Muon/AdamW split, data mix, WSD schedule, tokenizer
special-token notes, a load snippet, and **honest limitations** (the QA card states it
answers "the nucleus" for powerhouse-of-the-cell; the tool-call card states the
operation-selection failure).

---

## 12. Monitoring built for this run

| script | job |
| --- | --- |
| `pipeline.sh` | supervises pretraining (relaunch-on-death, resume verified), then chains SFT → tool-call → QA → RAG |
| `preserve_ckpts.sh` | rescues milestone checkpoints from `--keep-ckpts` rotation |
| `gpu_history.sh` | 60 s timestamped GPU samples, so saturation is auditable historically rather than by spot check |
| `gpu_watchdog.sh` | fires only on *real* anomalies: trainer alive but GPU under threshold for minutes, or nothing running while the pipeline is unfinished |
| `healthcheck.sh N` | waits up to N minutes, exiting **early** on NaN/death/completion, then prints a full report |
| `launch_train.sh` | correct venv `torchrun` + preflight on shard counts |
| `build_data.sh` | both corpora with the `--nproc` fix, resumable |

The watchdog threshold is deliberately **15%**, not 40%: pretraining runs at 99% util but
SFT runs at ~53% (small batches are launch-bound), so a higher bar would false-alarm on
healthy SFT. Real stalls read 0–4%. Checkpoint writes stall the GPU for ~1 sample, which
is why the stall counter requires several consecutive samples.

`launch_train.sh` preflights shard counts and **refuses to start on incomplete data** —
otherwise `DataLoaderLite` (`train.py:201-206`) logs "no train shards" and *renormalises
the mix weights to 1.0*, silently training with 0% code instead of 18%.

---

## 13. Reproducing this

```bash
source /workspace/.secrets/train-env.sh     # HF_TOKEN, HF_HOME, UV_* (chmod does not stick on the volume)
cd /workspace/modern-lm && uv sync          # torch 2.13.0+cu130, py3.13

bash /workspace/build_data.sh               # ~26 min, --nproc 22
bash /workspace/launch_train.sh             # B=16 + compile, --log-every 10, ~21h
nohup setsid bash /workspace/pipeline.sh &  # supervise + auto-chain all post-training
nohup setsid bash /workspace/preserve_ckpts.sh &
nohup setsid bash /workspace/gpu_watchdog.sh &

bash /workspace/healthcheck.sh 0            # instant status report
```

Everything above lives on `/workspace` and survives a pod stop/start. `/root` does not —
after a restart the shell environment, `gh` auth and Claude config are gone, though the
repo, venv, data, checkpoints and secrets are intact.

---

## 14. Final state and the /root migration

Only `/workspace` survives a pod stop on RunPod. `/root`, `/usr`, `/opt` and `/etc` are on
the container overlay and are wiped. Before shutdown the whole shell environment was moved
onto the volume and symlinked back, so a restarted pod can be restored in one command.

**Method — two classes of state, handled differently on purpose:**

- *Inert* (dotfiles, plugins, toolchains): **moved** to `/workspace/env/home/` and
  symlinked back. Zero drift afterwards — editing `~/.zshrc` edits the volume copy.
- *Live* (the Claude install and its state dir): **copied**, never moved. A running
  process was executing out of them; `bootstrap.sh` converts them to symlinks at next
  boot when nothing holds them open.

Verified after migration: all 13 symlinks resolve, and the tools still run *through* them
— `nvim 0.12.4`, `cargo 1.97.1`, `node v24.18.1`, and `gh auth status` still reports
logged in via the symlinked `~/.config/gh/hosts.yml`.

The restore path was **dry-run tested** into a throwaway `$HOME` before shutdown
(`ENV_HOME=/tmp/fakehome BOOTSTRAP_TEST=1 bash bootstrap.sh`), which linked every payload
item correctly. A migration you cannot restore is worthless, so this was tested rather
than assumed.

**To restore after starting the pod again:**

```bash
bash /workspace/env/bootstrap.sh
```

It is idempotent: reinstalls only the apt delta (from 36 MB of cached .debs, so it works
offline), relinks the payload, restores `/opt/nvim`, regenerates the locale, and sets zsh
as the login shell — in that order, because pointing root at zsh *before* zsh exists would
leave a container with no working login shell.

**Credentials note.** `~/.config/gh` (GitHub OAuth) and `~/.claude/.credentials.json` were
deliberately included so a restarted pod does not need re-authentication. Both now live on
the network volume, which outlives the pod and can be mounted by other pods — and `chmod`
does not stick on this MooseFS mount, so the files stay `rw-rw-rw-`. Rotate the tokens if
that volume is ever shared.

### Final artifact inventory (`/workspace`, 61 GB)

| what | size |
| --- | --- |
| `modern-lm/` repo + `.venv` | 7.2G |
| `data/` — 120 tokenized shards (10B + 2B tokens) | 23G |
| `keep/` — end-of-stable + end-of-decay checkpoints | 8.1G |
| `runs/` — every run's logs and final checkpoints | 19G |
| `env/` — the migrated `/root` environment | 4.9G |
| `sft_data/` — QA + RAG JSONL | 52M |
| `.secrets/train-env.sh` | — |

Nothing is single-homed on this pod: all six models are on HuggingFace and all code, logs
and this report are on GitHub.
