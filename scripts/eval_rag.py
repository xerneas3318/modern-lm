"""Greedy eval of a RAG checkpoint on a fixed 10-case suite (5 answerable, 5 abstain)."""
import sys, torch, glob
sys.path.insert(0, '/workspace/modern-lm')
from model import GPT, GPT2Config
from toolcall import build_enc, format_chat

import os
ck = sys.argv[1]
# accept a run directory OR a checkpoint path; isdir() is the reliable test
# (the old check only fired on a trailing slash or a glob, so passing a plain
# directory name tried to torch.load the directory itself)
if os.path.isdir(ck):
    cands = sorted(glob.glob(os.path.join(ck, 'checkpoints', 'sft_*.pt')))
    if not cands:
        sys.exit(f'no checkpoints in {ck}')
    ck = cands[-1]
enc = build_enc(); dev = 'cuda'
c = torch.load(ck, map_location=dev, weights_only=False)
m = GPT(GPT2Config(**c['config'])); m.load_state_dict(c['model']); m.to(dev).eval()
SYS = ('You are a helpful assistant. Answer the question using ONLY the provided context. '
       'If the context does not contain the answer, say that you dont know.')

def greedy(msgs, maxn=40):
    ids, _ = format_chat(enc, msgs, add_generation_prompt=True); ids = list(ids); s = len(ids)
    for _ in range(maxn):
        with torch.no_grad():
            lg, _ = m(torch.tensor([ids[-2048:]], device=dev))
        nx = int(lg[0, -1].argmax()); ids.append(nx)
        if nx in (50258, 50256): break
    return enc.decode(ids[s:])

A = 'The Eiffel Tower is located in Paris, France. It was completed in 1889 and stands 330 metres tall. It was designed by Gustave Eiffel.'
B = 'Photosynthesis occurs in the chloroplasts of plant cells. Chlorophyll absorbs light energy, which converts carbon dioxide and water into glucose and oxygen.'
C = 'The Amazon rainforest covers 5.5 million square kilometres across nine countries. Brazil contains about 60 percent of it.'
T = [(A,'How tall is the Eiffel Tower?','330'),(A,'Who designed it?','Gustave Eiffel'),
     (A,'In what year was it completed?','1889'),(A,'Who is the mayor of Paris?','ABSTAIN'),
     (B,'Where does photosynthesis occur?','chloroplast'),(B,'What absorbs light energy?','Chlorophyll'),
     (B,'What is the boiling point of water?','ABSTAIN'),(C,'How large is the Amazon rainforest?','5.5 million'),
     (C,'What percent is in Brazil?','60'),(C,'What is the capital of Peru?','ABSTAIN')]
ok = ab_ok = ans_ok = 0
for ctx, q, exp in T:
    a = greedy([{'role':'system','content':SYS},
                {'role':'user','content':f'Context:\n{ctx}\n\nQuestion: {q}'}]).replace('<|im_end|>','').strip()
    hit = ("don't know" in a.lower()) if exp == 'ABSTAIN' else (exp.lower() in a.lower())
    ok += hit
    if exp == 'ABSTAIN': ab_ok += hit
    else: ans_ok += hit
    print(f"  {'PASS' if hit else 'FAIL'}  expect={exp:15s} got={a!r}")
print(f"  ==== {ck.split('/')[-3]}: {ok}/10  (answerable {ans_ok}/7, abstain {ab_ok}/3) ====")
