#!/usr/bin/env python3
"""
score_evodiff.py — EvoDiff OA-DM-640M (discrete diffusion, OADM) scorer.

Per pool sequence:
  * embedding  = mean-pool of the pre-decoder hidden state (ByteNet trunk output)
                 of the CLEAN sequence (timestep = L, the fully-generated state).
  * perplexity = order-agnostic masked-reconstruction PPL from EvoDiff's own
                 OAMaskCollater (its exact OADM convention, so we never guess the
                 timestep). We emit TWO estimates from the SAME draws:
                   mean_perplexity      (pooled)  = exp( Σ CE / Σ masked tokens );
                       tokens pooled across draws -> heavily-masked (small-t) draws
                       dominate. This is the legacy estimate (over-weights the hard
                       regime, biased high).
                   mean_perplexity_elbo (ELBO)    = exp( mean_draws( mean_masked CE ) );
                       equal weight per draw. This is the Hoogeboom OA-ARDM per-residue
                       NLL bound that EvoDiff is TRAINED on: the reweight term
                       D/(D-t+1) cancels to a per-draw mean once normalised per residue
                       (see losses.OAMaskedCrossEntropyLoss, reweight=True), so this is
                       the model's actual (variational) perplexity, comparable in KIND
                       to ProGen2's exact causal exp(mean per-residue NLL).

Run in the dedicated venv (evodiff deps conflict with esm2_probe):
  source cross_architecture_nativeness/envs/evodiff_venv/bin/activate

API (from probe_evodiff diagnostic, 2026-06-01):
  model, collater, tokenizer, scheme = OA_DM_640M()        # scheme="mask"
  forward: model(x, timestep, input_mask=pad_mask[...,None]) -> logits [B,L,31]
  tokenizer.tokenize((seq,)) -> per-residue ids (no start/stop); pad_id=30, mask_id=28
  collater([(s,),...]) -> (src_masked[B,L], timestep[B], tgt[B,L], loss_mask[B,L])
  trunk: embedder -> decoder(PositionFeedForward, final proj) -> last_norm
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nat_io as _io  # noqa: E402

from evodiff.pretrained import OA_DM_38M, OA_DM_640M  # noqa: E402

# OADM scaling ladder (same ByteNetLMTime arch, different width/depth) — hook,
# embedding, and PPL code below are size-agnostic, so only the loader differs.
MODEL_MAP = {
    "evodiff_oadm_38m":  OA_DM_38M,   # 38M
    "evodiff_oadm_640m": OA_DM_640M,  # 640M (matched-scale default)
    # Same weights, separate output dir for the ELBO-vs-pooled scaling rerun (24
    # seeds, human pool) — keeps the main full-pool ρ(PC1,PPL) keys above untouched.
    "evodiff_oadm_38m_elbo":  OA_DM_38M,
    "evodiff_oadm_640m_elbo": OA_DM_640M,
}


def load(model_key, device):
    model, collater, tokenizer, scheme = MODEL_MAP[model_key]()
    model = model.to(device).eval()
    return model, collater, tokenizer


_HID = {}


def _register_hook(model):
    def hook(_mod, inp, _out):
        _HID["h"] = inp[0].detach()   # input to final decoder proj = trunk representation
    return model.decoder.register_forward_hook(hook)


@torch.no_grad()
def embed_batch(seqs, model, tokenizer, device, max_len):
    toks = [np.asarray(tokenizer.tokenize((s[:max_len],)), dtype=np.int64) for s in seqs]
    lens = [len(t) for t in toks]
    Lmax = max(lens)
    pad = tokenizer.pad_id
    x = torch.full((len(toks), Lmax), pad, dtype=torch.long)
    for i, t in enumerate(toks):
        x[i, :len(t)] = torch.tensor(t)
    x = x.to(device)
    pad_mask = (x != pad).float()
    timestep = torch.tensor(lens, dtype=torch.long, device=device)  # "fully generated" state
    _HID.clear()
    _ = model(x, timestep, input_mask=pad_mask.unsqueeze(-1))
    hidden = _HID["h"]                       # [B, Lmax, d_model]
    return _io.mean_pool(hidden, pad_mask)


@torch.no_grad()
def ppl_batch(seqs, model, collater, tokenizer, device, n_seeds, max_len):
    """Return (pooled_ppl[B], elbo_ppl[B]) computed from the SAME OADM draws.

    pooled = exp( Σ_draws Σ_masked CE / Σ_draws #masked )  -- tokens pooled across
             draws, so small-t (heavily masked) draws dominate; biased toward the hard
             regime (legacy estimate).
    elbo   = exp( (1/S) Σ_draws  mean_masked CE )          -- equal weight per draw.
             This is the per-residue OA-ARDM NLL bound: with the Hoogeboom reweight
             D/(D-t+1), one draw's joint-NLL estimate = D · mean_masked CE, so the
             per-residue contribution is just mean_masked CE and the D/(D-t+1) cancels.
             The ELBO is an expectation over t -> average per-draw means equally.
    """
    pad = tokenizer.pad_id
    sub = [s[:max_len] for s in seqs]
    sum_nll = np.zeros(len(sub))        # pooled numerator: Σ CE over all masked tokens
    cnt = np.zeros(len(sub))            # pooled denominator: # masked tokens
    sum_seed_mean = np.zeros(len(sub))  # elbo numerator: Σ over draws of per-draw mean CE
    n_draws = np.zeros(len(sub))        # elbo denominator: # draws with ≥1 masked pos
    for _ in range(n_seeds):
        src, timestep, tgt, _lossmask = collater([(s,) for s in sub])
        src, timestep, tgt = src.to(device), timestep.to(device), tgt.to(device)
        pad_mask = (src != pad).float()
        masked = (src == tokenizer.mask_id)              # positions OADM chose to predict
        logits = model(src, timestep, input_mask=pad_mask.unsqueeze(-1)).float()
        for i in range(len(sub)):
            pos = masked[i].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            ce = F.cross_entropy(logits[i, pos], tgt[i, pos], reduction="sum").item()
            sum_nll[i] += ce
            cnt[i] += len(pos)
            sum_seed_mean[i] += ce / len(pos)            # per-draw mean CE (equal weight)
            n_draws[i] += 1
    pooled = [float(np.exp(sum_nll[i] / cnt[i])) if cnt[i] > 0 else float("nan")
              for i in range(len(sub))]
    elbo = [float(np.exp(sum_seed_mean[i] / n_draws[i])) if n_draws[i] > 0 else float("nan")
            for i in range(len(sub))]
    return pooled, elbo


def run_pool(task, model, collater, tokenizer, device, batch_size, max_len, n_seeds, limit,
             want_emb=True):
    recs = _io.parse_fasta(task["fasta"])
    if limit:
        recs = recs[:limit]
    accs = [a for a, _ in recs]
    seqs = [s for _, s in recs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    embs = [None] * len(seqs) if want_emb else None
    ppls = [None] * len(seqs)
    ppls_elbo = [None] * len(seqs)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        bs = [seqs[i] for i in idx]
        if want_emb:
            e = embed_batch(bs, model, tokenizer, device, max_len)
        p, pe = ppl_batch(bs, model, collater, tokenizer, device, n_seeds, max_len)
        for kk, i in enumerate(idx):
            if want_emb:
                embs[i] = e[kk]
            ppls[i] = p[kk]; ppls_elbo[i] = pe[kk]
        done = min(b + batch_size, len(order))
        print(f"\r  {task['emb_name']:>20s}: {done:,}/{len(order):,}  "
              f"{done/max(time.time()-t0,1e-6):.0f} seq/s", end="", flush=True)
    print()
    if want_emb:
        _io.save_embeddings(task["model_key"], task["emb_name"], np.vstack(embs), np.array(accs))
    rows = []
    for acc, seq, ppl, ppl_e in zip(accs, seqs, ppls, ppls_elbo):
        base = {"accession": acc, "label": task["label"], "length": len(seq),
                "mean_perplexity": ppl, "mean_perplexity_elbo": ppl_e}
        if task["kind"] == "human":
            base = {"accession": acc, "label": task["label"], "split": task["split"],
                    "length": len(seq), "mean_perplexity": ppl, "mean_perplexity_elbo": ppl_e}
        rows.append(base)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_key", default="evodiff_oadm_640m", choices=list(MODEL_MAP))
    ap.add_argument("--pools", default="all")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="OADM masking draws per sequence (>=20 recommended for the ELBO estimate)")
    ap.add_argument("--seed", type=int, default=0,
                    help="np.random seed for the OAMaskCollater draws (reproducible masks)")
    ap.add_argument("--ppl_only", action="store_true",
                    help="skip pre-decoder embeddings (PPL only) — scaling needs PPL alone")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)            # OAMaskCollater draws via np.random -> reproducible
    device = torch.device(args.device)
    want_emb = not args.ppl_only
    print(f"=== EvoDiff scorer | {args.model_key} | pools={args.pools} | device={device} "
          f"| n_seeds={args.n_seeds} seed={args.seed} ppl_only={args.ppl_only} ===")
    model, collater, tokenizer = load(args.model_key, device)
    if want_emb:
        _register_hook(model)
    print(f"loaded {args.model_key} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    human_rows, group_rows = [], {}
    for task in _io.all_tasks(args.pools):
        task["model_key"] = args.model_key
        rows = run_pool(task, model, collater, tokenizer, device,
                        args.batch_size, args.max_len, args.n_seeds, args.limit, want_emb)
        if task["kind"] == "human":
            human_rows.extend(rows)
        else:
            group_rows[task["group"]] = rows
    _io.write_ppl(args.model_key, human_rows, group_rows)
    emb_msg = (_io.EMB_ROOT / args.model_key) if want_emb else "(skipped: --ppl_only)"
    print(f"\nDONE. embeddings -> {emb_msg}\n      PPL -> {_io.PPL_ROOT/args.model_key}")
    if human_rows:
        for col, name in [("mean_perplexity", "pooled"), ("mean_perplexity_elbo", "ELBO")]:
            v = np.array([r[col] for r in human_rows if r["label"] == "viral"])
            n = np.array([r[col] for r in human_rows if r["label"] == "nonviral"])
            if len(v) and len(n):
                print(f"      human PPL [{name:>6s}]: viral median {np.nanmedian(v):.2f} | "
                      f"nonviral median {np.nanmedian(n):.2f}")


if __name__ == "__main__":
    main()
