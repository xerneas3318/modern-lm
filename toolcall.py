"""
toolcall.py -- calculator tool-calling for the modern-lm model.

Three pieces, all model-agnostic (you pass in your loaded `model` + `enc`):
  1. safe_calc(expr)          : evaluate ONLY arithmetic (no names/calls/imports).
                                This is a calculator, not a code executor.
  2. SFT-data formatting      : format_chat() / format_toolcall_example() build
                                ChatML token sequences + a loss mask (train only on
                                assistant tokens; prompt + injected tool results masked).
  3. generate_with_tools()    : inference loop -- sample tokens, and when the model
                                emits <tool_call>expr</tool_call>, run safe_calc and
                                inject <tool_result>...</tool_result>, then continue.

Special-token IDs match train.py SPECIAL_TOKENS (reserved in the padded vocab, so no
embedding surgery). The model is TAUGHT to emit tool calls at SFT (see train.md); it
does not tool-call straight out of pretraining.

Wiring:
    from toolcall import build_enc, generate_with_tools
    enc = build_enc()
    # load your GPT + latest checkpoint's "model" state_dict, .to(device).eval()
    print(generate_with_tools(model, enc, "What is 47 * 89?", device))
"""

import ast
import operator

import torch
import torch.nn.functional as F

from model import SPECIAL_TOKENS, EOS_ID as EOT, build_enc  # shared tokenizer + specials


# ----------------------------------------------------------------------------
# 1. calculator (arithmetic only -- NOT code execution)
# ----------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_calc(expr: str):
    """Evaluate a plain arithmetic expression. Rejects names, calls, imports, etc.
    Returns a number, or raises ValueError. This never runs arbitrary code."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("not a plain arithmetic expression")
    return ev(ast.parse(expr.strip(), mode="eval").body)


def _fmt(x):
    # integer-ish floats print without the trailing .0
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


# ----------------------------------------------------------------------------
# 2. SFT-data formatting (ChatML + loss mask)
# ----------------------------------------------------------------------------
def _ids(enc, s):
    return enc.encode(s, allowed_special="all")


def format_chat(enc, messages, add_generation_prompt=False):
    """messages: list of {"role": "system|user|assistant", "content": str}.
    Returns (token_ids, loss_mask) where loss_mask[i]=1 only on ASSISTANT content
    tokens (+ the closing <|im_end|>), so SFT trains on the response only.
    ChatML: <|im_start|>role\n{content}<|im_end|>\n"""
    ids, mask = [], []
    def add(s, learn):
        t = _ids(enc, s)
        ids.extend(t)
        mask.extend([1 if learn else 0] * len(t))
    for m in messages:
        add(f"<|im_start|>{m['role']}\n", False)
        if m["role"] == "assistant":
            add(m["content"], True)
            add("<|im_end|>\n", True)      # learn to emit the stop token
        else:
            add(m["content"], False)
            add("<|im_end|>\n", False)
    if add_generation_prompt:
        add("<|im_start|>assistant\n", False)
    return ids, mask


def format_toolcall_example(enc, question, reasoning, expr, answer, system=None):
    """Build ONE tool-calling SFT example (tokens + loss mask).

    Layout of the assistant turn:
        {reasoning}
        <tool_call>{expr}</tool_call>
        <tool_result>{safe_calc(expr)}</tool_result>
        {answer}

    Loss is on the assistant's tokens EXCEPT the injected <tool_result>...</tool_result>
    (the model must learn to WRITE the call and USE the result, not to produce the result).
    """
    result = _fmt(safe_calc(expr))
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": question})
    ids, mask = format_chat(enc, msgs, add_generation_prompt=True)

    def add(s, learn):
        t = _ids(enc, s); ids.extend(t); mask.extend([1 if learn else 0] * len(t))

    add(f"{reasoning}\n", True)                                   # learn: reasoning
    add(f"<tool_call>{expr}</tool_call>\n", True)                 # learn: the call
    add(f"<tool_result>{result}</tool_result>\n", False)         # DON'T learn: injected result
    add(f"{answer}<|im_end|>\n", True)                            # learn: final answer + stop
    return ids, mask


# ----------------------------------------------------------------------------
# 3. tool-calling inference loop
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_with_tools(model, enc, prompt, device, system=None,
                        max_new=256, temperature=0.7, top_k=40, block_size=2048):
    """Generate an assistant reply, executing <tool_call>...</tool_call> as they appear
    and injecting <tool_result>...</tool_result> back into the context. Stops at
    <|im_end|> or <|endoftext|>. Returns the decoded assistant text."""
    im_end, tc_open, tc_close = SPECIAL_TOKENS["<|im_end|>"], SPECIAL_TOKENS["<tool_call>"], SPECIAL_TOKENS["</tool_call>"]
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
            # extract the expression between the last <tool_call> and this </tool_call>
            try:
                open_at = len(ids) - 1 - ids[::-1].index(tc_open)
                expr = enc.decode(ids[open_at + 1:len(ids) - 1])
                result = _fmt(safe_calc(expr))
            except Exception as e:
                result = f"ERROR: {e}"
            ids.extend(_ids(enc, f"\n<tool_result>{result}</tool_result>\n"))
            steps = 0   # give the model room to use the result

    return enc.decode(ids[start:])


if __name__ == "__main__":
    # quick self-test of the calculator + formatting (no model needed)
    enc = build_enc()
    assert safe_calc("47*89") == 4183 and safe_calc("2**10 + 3") == 1027
    try:
        safe_calc("__import__('os').system('ls')"); raise SystemExit("SECURITY FAIL")
    except ValueError:
        pass
    ids, mask = format_toolcall_example(
        enc,
        question="A box has 47 rows of 89 apples. How many apples?",
        reasoning="I need 47 times 89.",
        expr="47*89",
        answer="There are 4183 apples.",
    )
    print(f"safe_calc OK; rejects code; example = {len(ids)} tokens, "
          f"{sum(mask)} trained (result span masked = {mask[ids.index(SPECIAL_TOKENS['<tool_result>'])]==0})")
