"""Calculator tool-calling: a safe arithmetic evaluator, ChatML formatting with a
loss mask for SFT, and an inference loop that runs <tool_call> expressions."""

import ast
import operator

import torch
import torch.nn.functional as F

from model import SPECIAL_TOKENS, EOS_ID as EOT, build_enc

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_calc(expr):
    """Evaluate an arithmetic expression, rejecting anything else (names, calls, ...)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("not an arithmetic expression")
    return ev(ast.parse(expr.strip(), mode="eval").body)


def _fmt(x):
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def _ids(enc, s):
    return enc.encode(s, allowed_special="all")


def format_chat(enc, messages, add_generation_prompt=False):
    """ChatML token ids plus a mask that is 1 on the assistant's tokens only."""
    ids, mask = [], []

    def add(s, learn):
        t = _ids(enc, s)
        ids.extend(t)
        mask.extend([int(learn)] * len(t))

    for m in messages:
        add(f"<|im_start|>{m['role']}\n", False)
        learn = m["role"] == "assistant"
        add(m["content"], learn)
        add("<|im_end|>\n", learn)
    if add_generation_prompt:
        add("<|im_start|>assistant\n", False)
    return ids, mask


def format_toolcall_example(enc, question, reasoning, expr, answer, system=None):
    """One tool-calling SFT example. The injected tool result is masked out, so the
    model learns to write the call and use the result rather than produce it."""
    result = _fmt(safe_calc(expr))
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": question})
    ids, mask = format_chat(enc, msgs, add_generation_prompt=True)

    def add(s, learn):
        t = _ids(enc, s)
        ids.extend(t)
        mask.extend([int(learn)] * len(t))

    add(f"{reasoning}\n", True)
    add(f"<tool_call>{expr}</tool_call>\n", True)
    add(f"<tool_result>{result}</tool_result>\n", False)
    add(f"{answer}<|im_end|>\n", True)
    return ids, mask


@torch.no_grad()
def generate_with_tools(model, enc, prompt, device, system=None,
                        max_new=256, temperature=0.7, top_k=40, block_size=2048):
    """Generate an assistant reply, running <tool_call>...</tool_call> as it appears
    and feeding the result back. Stops at <|im_end|> or <|endoftext|>."""
    im_end = SPECIAL_TOKENS["<|im_end|>"]
    tc_open = SPECIAL_TOKENS["<tool_call>"]
    tc_close = SPECIAL_TOKENS["</tool_call>"]
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": prompt})
    ids, _ = format_chat(enc, msgs, add_generation_prompt=True)
    ids = list(ids)
    start = len(ids)

    steps = 0
    while steps < max_new:
        logits, _ = model(torch.tensor([ids[-block_size:]], device=device))
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
        steps += 1
        if nxt in (im_end, EOT):
            break
        if nxt == tc_close:
            try:
                open_at = len(ids) - 1 - ids[::-1].index(tc_open)
                expr = enc.decode(ids[open_at + 1:len(ids) - 1])
                result = _fmt(safe_calc(expr))
            except Exception as e:
                result = f"ERROR: {e}"
            ids.extend(_ids(enc, f"\n<tool_result>{result}</tool_result>\n"))
            steps = 0
    return enc.decode(ids[start:])


if __name__ == "__main__":
    enc = build_enc()
    assert safe_calc("47*89") == 4183
    assert safe_calc("2**10 + 3") == 1027
    try:
        safe_calc("__import__('os').system('ls')")
        raise SystemExit("expected a rejection")
    except ValueError:
        pass
    ids, mask = format_toolcall_example(enc, "What is 2 + 2?", "add them", "2+2", "It is 4.")
    print(f"ok: {len(ids)} tokens, {sum(mask)} trained")
