#!/usr/bin/env python3
"""
score_progen2.py — ProGen2 (autoregressive) scorer for the nativeness study.

Computes, in a single forward pass per batch, for every pool sequence:
  * mean-pool embedding of the last hidden state over residue positions (-> .npz)
  * TRUE autoregressive perplexity = exp(mean per-residue NLL)            (-> .tsv)

ProGen2 is decoder-only; tokenization is per-residue (vocab 32, BOS='1', EOS='2').
We score N->C: input = '1' + seq + '2'; logits[i] predicts token[i+1]. The N residue
predictions (targets r_1..r_N) are kept for PPL; BOS, EOS, and pad are excluded from
both PPL and the embedding mean-pool — mirroring the ESM masked-recon / embedding
conventions so results are directly comparable.

Deterministic (no masking seeds). Output layout is defined entirely in _io.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nat_io as _io  # noqa: E402  (named nat_io: '_io' shadows a CPython builtin)

MODEL_MAP = {
    "progen2_small": "hugohrban/progen2-small",   # 151M
    "progen2_base":  "hugohrban/progen2-base",     # 764M  (matched-scale default)
    "progen2_large": "hugohrban/progen2-large",    # 2.7B
    "progen2_xlarge": "hugohrban/progen2-xlarge",  # 6.4B
}
BOS, EOS = "1", "2"


def load(model_key, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODEL_MAP[model_key]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # bf16 on GPU halves weight memory (2.7B/6.4B otherwise OOM on a 20GB MIG slice);
    # compute already autocasts to bf16 and logits are upcast to f32 for stable CE.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    return model, tok, name


def process_batch(seqs, model, tok, device, max_len, want_emb=True):
    """Return (embeddings [B,D] f32 or None, ppls [B] list) for raw AA strings.

    want_emb=False skips output_hidden_states (which otherwise materialises ALL
    layers) and the mean-pool — only PPL is computed, so big models fit a 20GB MIG.
    """
    toks = [tok(BOS + s[:max_len] + EOS, return_tensors=None)["input_ids"] for s in seqs]
    lens = [len(t) for t in toks]            # L_i = N_i + 2
    Lmax = max(lens)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    input_ids = torch.full((len(toks), Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((len(toks), Lmax), dtype=torch.long)
    for i, t in enumerate(toks):
        input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        attn[i, :len(t)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=want_emb)
    logits = out.logits.float()

    # --- embedding: keep residue positions 1..N_i (drop BOS idx0, EOS idx L_i-1, pad) ---
    emb = None
    if want_emb:
        hidden = out.hidden_states[-1]
        keep = attn.clone().float()
        keep[:, 0] = 0.0
        for i, L in enumerate(lens):
            keep[i, L - 1] = 0.0
        emb = _io.mean_pool(hidden, keep)

    # --- AR PPL: shifted CE; residue targets are shifted positions 0..N_i-1 ---
    logp = logits[:, :-1, :]
    tgt = input_ids[:, 1:]
    ce = F.cross_entropy(logp.reshape(-1, logp.size(-1)), tgt.reshape(-1),
                         reduction="none").view(tgt.shape)
    ppls = []
    for i, L in enumerate(lens):
        N = L - 2
        nll = ce[i, 0:N].mean().item()
        ppls.append(float(np.exp(nll)))
    return emb, ppls


def run_pool(task, model, tok, device, batch_size, max_len, limit, want_emb=True):
    recs = _io.parse_fasta(task["fasta"])
    if limit:
        recs = recs[:limit]
    accs = [a for a, _ in recs]
    seqs = [s for _, s in recs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))

    embs = [None] * len(seqs) if want_emb else None
    ppls = [None] * len(seqs)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        e, p = process_batch([seqs[i] for i in idx], model, tok, device, max_len, want_emb)
        for k, i in enumerate(idx):
            if want_emb:
                embs[i] = e[k]
            ppls[i] = p[k]
        done = min(b + batch_size, len(order))
        rate = done / max(time.time() - t0, 1e-6)
        print(f"\r  {task['emb_name']:>20s}: {done:,}/{len(order):,}  {rate:.0f} seq/s",
              end="", flush=True)
    print()
    if want_emb:
        _io.save_embeddings(task["model_key"], task["emb_name"], np.vstack(embs), np.array(accs))

    rows = []
    for acc, seq, ppl in zip(accs, seqs, ppls):
        if task["kind"] == "human":
            rows.append({"accession": acc, "label": task["label"], "split": task["split"],
                         "length": len(seq), "mean_perplexity": ppl})
        else:
            rows.append({"accession": acc, "label": task["label"],
                         "length": len(seq), "mean_perplexity": ppl})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_key", default="progen2_base", choices=list(MODEL_MAP))
    ap.add_argument("--pools", default="all", help="all|human|phage|controls")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N seqs/pool")
    ap.add_argument("--ppl_only", action="store_true",
                    help="skip embeddings/hidden-states (PPL only) — fits big models on a 20GB MIG")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"=== ProGen2 scorer | {args.model_key} | pools={args.pools} | device={device} ===")
    model, tok, name = load(args.model_key, device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {name}  ({nparams:.0f}M params)  emb_dim="
          f"{model.config.hidden_size if hasattr(model.config,'hidden_size') else '?'}")

    want_emb = not args.ppl_only
    human_rows, group_rows = [], {}
    for task in _io.all_tasks(args.pools):
        task["model_key"] = args.model_key
        rows = run_pool(task, model, tok, device, args.batch_size, args.max_len,
                        args.limit, want_emb)
        if task["kind"] == "human":
            human_rows.extend(rows)
        else:
            group_rows[task["group"]] = rows

    _io.write_ppl(args.model_key, human_rows, group_rows)
    print(f"\nDONE. embeddings -> {_io.EMB_ROOT/args.model_key}")
    print(f"      PPL        -> {_io.PPL_ROOT/args.model_key}")
    if human_rows:
        v = np.array([r["mean_perplexity"] for r in human_rows if r["label"] == "viral"])
        n = np.array([r["mean_perplexity"] for r in human_rows if r["label"] == "nonviral"])
        if len(v) and len(n):
            print(f"      human PPL: viral median {np.median(v):.2f} | "
                  f"nonviral median {np.median(n):.2f}  (expect viral > nonviral)")


if __name__ == "__main__":
    main()
