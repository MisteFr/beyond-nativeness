#!/usr/bin/env python3
"""
score_prott5.py — ProtT5 (T5 encoder-decoder, span denoising) scorer.

Per pool sequence:
  * embedding  = mean-pool of the ENCODER last hidden state over residues (clean seq)
  * perplexity = span-denoising pseudo-PPL = exp(mean NLL over masked residues)

PPL recipe (mirrors ProtT5's pretraining objective + the ESM masked-marginal idea):
mask k = min(max(1, round(0.15*N)), 90) residues (single pass; capped at 90 ≤ the 100
sentinels T5 provides). Each masked residue becomes a distinct sentinel <extra_id_j>
in the encoder input; the decoder target is the interleaved [<extra_id_j>, residue_j]
sequence; we score CE only on the residue tokens. Capping at 90 keeps it one forward
pass per sequence — the mean NLL over a random ≤90-residue subset is an unbiased
estimate of the per-residue reconstruction NLL, so no multi-pass chunking is needed.

Rostlab conventions: residues space-separated, U/Z/O/B -> X, </s> appended (no BOS).
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nat_io as _io  # noqa: E402

MODEL_MAP = {"prott5_xl": "Rostlab/prot_t5_xl_uniref50"}
RARE = re.compile(r"[UZOB]")


def load(model_key, device):
    from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer
    name = MODEL_MAP[model_key]
    tok = T5Tokenizer.from_pretrained(name, do_lower_case=False)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = T5ForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    sentinels = [tok.convert_tokens_to_ids(f"<extra_id_{j}>") for j in range(100)]
    return model, tok, name, sentinels


def _clean(seq, max_len):
    return RARE.sub("X", seq.upper())[:max_len]


@torch.no_grad()
def embed_batch(seqs, model, tok, device, max_len):
    proc = [" ".join(list(_clean(s, max_len))) for s in seqs]
    enc = tok(proc, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = model.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
    # drop the trailing </s> (last real token) from each row, plus pad
    keep = attn.clone().float()
    lens = attn.sum(dim=1)
    for i, L in enumerate(lens):
        keep[i, int(L) - 1] = 0.0
    return _io.mean_pool(hidden, keep)


def _res_ids(seq, tok, eos, max_len):
    r = tok(" ".join(list(_clean(seq, max_len))))["input_ids"]
    return r[:-1] if r and r[-1] == eos else r


@torch.no_grad()
def ppl_batch(seqs, model, tok, device, mask_id, max_len, mask_rate, n_seeds):
    """BART-style masked-reconstruction pseudo-PPL (ProtT5 is a full-sequence denoiser,
    confirmed by diagnostic): mask `mask_rate` of residues in the ENCODER, teacher-force
    the FULL true sequence in the decoder; logits[p] predicts residue r_p, so CE at the
    masked positions = reconstruction NLL. PPL = exp(mean NLL over masked positions).
    Mirrors the ESM masked-reconstruction recipe (15% mask, n_seeds), adapted to enc-dec.
    """
    eos, pad = tok.eos_token_id, tok.pad_token_id
    dec_start = model.config.decoder_start_token_id
    res = [_res_ids(s, tok, eos, max_len) for s in seqs]
    res = [r if r else [tok.unk_token_id] for r in res]
    B = len(res)
    Lmax = max(len(r) for r in res) + 1  # + </s> (enc) / + start (dec)

    base_enc = torch.full((B, Lmax), pad, dtype=torch.long)
    enc_attn = torch.zeros((B, Lmax), dtype=torch.long)
    dec = torch.full((B, Lmax), pad, dtype=torch.long)
    dec_attn = torch.zeros((B, Lmax), dtype=torch.long)
    for i, r in enumerate(res):
        N = len(r)
        base_enc[i, :N] = torch.tensor(r); base_enc[i, N] = eos
        enc_attn[i, :N + 1] = 1
        dec[i, 0] = dec_start; dec[i, 1:N + 1] = torch.tensor(r)  # logits[p] predicts r_p
        dec_attn[i, :N + 1] = 1
    base_enc, enc_attn = base_enc.to(device), enc_attn.to(device)
    dec, dec_attn = dec.to(device), dec_attn.to(device)

    sum_nll, cnt = np.zeros(B), np.zeros(B)
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        enc = base_enc.clone()
        masked = []
        for i, r in enumerate(res):
            N = len(r)
            k = max(1, int(N * mask_rate))
            pos = rng.choice(N, size=k, replace=False)
            enc[i, pos] = mask_id
            masked.append(pos)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=enc, attention_mask=enc_attn,
                           decoder_input_ids=dec, decoder_attention_mask=dec_attn).logits.float()
        for i, r in enumerate(res):
            pos = masked[i]
            lg = logits[i, pos, :]
            tr = torch.tensor([r[p] for p in pos], device=device)
            sum_nll[i] += F.cross_entropy(lg, tr, reduction="sum").item()
            cnt[i] += len(pos)
    return [float(np.exp(sum_nll[i] / cnt[i])) for i in range(B)]


def run_pool(task, model, tok, device, mask_id, batch_size, max_len, mask_rate, n_seeds, limit):
    recs = _io.parse_fasta(task["fasta"])
    if limit:
        recs = recs[:limit]
    accs = [a for a, _ in recs]
    seqs = [s for _, s in recs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    embs = [None] * len(seqs)
    ppls = [None] * len(seqs)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        bs = [seqs[i] for i in idx]
        e = embed_batch(bs, model, tok, device, max_len)
        p = ppl_batch(bs, model, tok, device, mask_id, max_len, mask_rate, n_seeds)
        for kk, i in enumerate(idx):
            embs[i] = e[kk]; ppls[i] = p[kk]
        done = min(b + batch_size, len(order))
        print(f"\r  {task['emb_name']:>20s}: {done:,}/{len(order):,}  "
              f"{done/max(time.time()-t0,1e-6):.0f} seq/s", end="", flush=True)
    print()
    _io.save_embeddings(task["model_key"], task["emb_name"], np.vstack(embs), np.array(accs))
    rows = []
    for acc, seq, ppl in zip(accs, seqs, ppls):
        base = {"accession": acc, "label": task["label"], "length": len(seq), "mean_perplexity": ppl}
        if task["kind"] == "human":
            base = {"accession": acc, "label": task["label"], "split": task["split"],
                    "length": len(seq), "mean_perplexity": ppl}
        rows.append(base)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_key", default="prott5_xl", choices=list(MODEL_MAP))
    ap.add_argument("--pools", default="all")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--mask_rate", type=float, default=0.15)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"=== ProtT5 scorer | {args.model_key} | pools={args.pools} | device={device} ===")
    model, tok, name, _ = load(args.model_key, device)
    mask_id = tok.unk_token_id  # diagnostic: unk == extra_id_0 for masked-recon
    print(f"loaded {name}  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)  mask_id={mask_id}")

    human_rows, group_rows = [], {}
    for task in _io.all_tasks(args.pools):
        task["model_key"] = args.model_key
        rows = run_pool(task, model, tok, device, mask_id,
                        args.batch_size, args.max_len, args.mask_rate, args.n_seeds, args.limit)
        if task["kind"] == "human":
            human_rows.extend(rows)
        else:
            group_rows[task["group"]] = rows
    _io.write_ppl(args.model_key, human_rows, group_rows)
    print(f"\nDONE. embeddings -> {_io.EMB_ROOT/args.model_key}\n      PPL -> {_io.PPL_ROOT/args.model_key}")
    if human_rows:
        v = np.array([r["mean_perplexity"] for r in human_rows if r["label"] == "viral"])
        n = np.array([r["mean_perplexity"] for r in human_rows if r["label"] == "nonviral"])
        if len(v) and len(n):
            print(f"      human PPL: viral median {np.median(v):.2f} | nonviral median {np.median(n):.2f}")


if __name__ == "__main__":
    main()
