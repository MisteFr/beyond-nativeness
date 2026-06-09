#!/usr/bin/env python3
"""
run_masked_reconstruction_esm2.py
=================================
ESM2 masked token reconstruction experiment (local GPU).

Computes per-sequence perplexity (PPL) and log-likelihood (LL) as
zero-shot viral/nonviral classifiers.  No training step — pure inference.

Methodology identical to the ESMC/ESM3 masked reconstruction scripts:
  - Mask 15% of amino acid positions (excluding BOS/EOS/PAD)
  - 3 random seeds to reduce noise
  - PPL = exp(mean NLL over masked positions)           [length-normalized]
  - LL  = sum(log p(true AA) at masked positions)       [length-dependent]

Usage:
    python scripts/run_masked_reconstruction_esm2.py \\
        --model     esm2_650m \\
        --data_dir  data \\
        --out_dir   results/esm2_650m \\
        --cache_dir /path/to/hf_cache \\
        --batch_size 4 \\
        --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model name mapping
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "esm2_8m":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35m":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b":   "facebook/esm2_t36_3B_UR50D",
    "esm2_15b":  "facebook/esm2_t48_15B_UR50D",
}


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------
def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Return list of (accession, sequence) pairs."""
    records = []
    header, parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header.split()[0], "".join(parts)))
                header = line[1:]
                parts = []
            else:
                parts.append(line.upper())
    if header is not None:
        records.append((header.split()[0], "".join(parts)))
    return records


# ---------------------------------------------------------------------------
# Core masked reconstruction
# ---------------------------------------------------------------------------
def process_batch(
    batch_seqs: list[str],
    model,
    tokenizer,
    device: torch.device,
    mask_rate: float,
    n_seeds: int,
    max_len: int,
) -> list[dict]:
    """
    For each sequence in the batch, mask mask_rate fraction of amino acid
    positions (excluding CLS/EOS/PAD) for each of n_seeds random seeds, run
    EsmForMaskedLM forward pass, and compute per-seed NLL, recovery, and LL.
    """
    seqs_trunc = [s[:max_len] for s in batch_seqs]
    B = len(seqs_trunc)

    encoded = tokenizer(
        seqs_trunc,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2,
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)

    mask_token_id = tokenizer.mask_token_id
    special_tokens = {tokenizer.cls_token_id, tokenizer.eos_token_id,
                      tokenizer.pad_token_id}

    nlls = np.zeros((B, n_seeds))
    recoveries = np.zeros((B, n_seeds))
    log_likelihoods = np.zeros((B, n_seeds))
    n_masked_per_seq = np.zeros(B, dtype=int)

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        masked_ids = input_ids.clone()
        mask_positions = []

        for i in range(B):
            # Find maskable AA positions (exclude special tokens and padding)
            aa_pos = []
            for p in range(input_ids.shape[1]):
                if attn_mask[i, p] == 0:
                    continue
                if input_ids[i, p].item() in special_tokens:
                    continue
                aa_pos.append(p)
            n_mask = max(1, int(len(aa_pos) * mask_rate))
            chosen = rng.choice(len(aa_pos), size=n_mask, replace=False)
            pos_i = [aa_pos[j] for j in chosen]
            mask_positions.append(pos_i)
            for p in pos_i:
                masked_ids[i, p] = mask_token_id

            if seed == 0:
                n_masked_per_seq[i] = n_mask

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda"
        ):
            output = model(input_ids=masked_ids, attention_mask=attn_mask)
        logits = output.logits.float()

        for i in range(B):
            mp = torch.tensor(mask_positions[i], dtype=torch.long, device=device)
            lg = logits[i, mp]
            tr = input_ids[i, mp]

            # Sum NLL for both PPL and LL
            sum_nll = F.cross_entropy(lg, tr, reduction="sum").item()
            n_pos = len(mask_positions[i])
            mean_nll = sum_nll / n_pos

            nlls[i, seed] = mean_nll
            recoveries[i, seed] = (lg.argmax(dim=-1) == tr).float().mean().item()
            log_likelihoods[i, seed] = -sum_nll  # LL = -sum(NLL)

    results = []
    for i in range(B):
        results.append({
            "mean_perplexity":    float(np.exp(nlls[i].mean())),
            "mean_recovery_rate": float(recoveries[i].mean()),
            "mean_log_likelihood": float(log_likelihoods[i].mean()),
            "n_masked":           int(n_masked_per_seq[i]),
            "seed_ppls":          [float(np.exp(nlls[i, s])) for s in range(n_seeds)],
            "seed_recoveries":    [float(recoveries[i, s])   for s in range(n_seeds)],
            "seed_lls":           [float(log_likelihoods[i, s]) for s in range(n_seeds)],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()),
                        help="ESM2 model key")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing viral/nonviral {train,val,test}.faa")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory (TSV, JSON)")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1022)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                        help="Weight dtype for from_pretrained. bf16 halves memory for 3B/15B.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    hf_model_name = MODEL_MAP[args.model]

    print("=" * 60)
    print(f"ESM2 Masked Reconstruction — {args.model}")
    print(f"HF model:   {hf_model_name}")
    print(f"Device:     {device}")
    if device.type == "cuda":
        try:
            print(f"GPU:        {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_mem', None) or props.total_memory
            print(f"VRAM:       {vram / 1e9:.1f} GB")
        except Exception:
            pass
    print(f"Mask rate:  {args.mask_rate}")
    print(f"N seeds:    {args.n_seeds}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    # ---- Load model ----
    from transformers import EsmForMaskedLM, AutoTokenizer

    print(f"\nLoading tokenizer: {hf_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    print(f"Loading model: {hf_model_name}  (dtype={args.dtype}) ...")
    model = EsmForMaskedLM.from_pretrained(hf_model_name, torch_dtype=torch_dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")

    # Sanity check
    print("  Sanity check (forward pass)...")
    with torch.no_grad():
        _tok = tokenizer(["MKTAYIAKQR"], return_tensors="pt", padding=True,
                         truncation=True, max_length=15)
        _out = model(input_ids=_tok["input_ids"].to(device),
                     attention_mask=_tok["attention_mask"].to(device))
    assert hasattr(_out, "logits"), \
        "ERROR: EsmForMaskedLM output missing .logits"
    print(f"  logits shape: {list(_out.logits.shape)}  OK")
    del _tok, _out

    # ---- Load sequences ----
    print("\nLoading sequences...")
    all_meta = []
    all_seqs = []

    for label in ["viral", "nonviral"]:
        for split in ["train", "val", "test"]:
            fasta_path = Path(args.data_dir) / f"{label}_{split}.faa"
            if not fasta_path.exists():
                print(f"  WARNING: {fasta_path} not found — skipping")
                continue
            records = parse_fasta(str(fasta_path))
            for acc, seq in records:
                all_meta.append((acc, label, split))
                all_seqs.append(seq)
            print(f"  {label}/{split}: {len(records):,}")

    n_total = len(all_seqs)
    print(f"  Total: {n_total:,} sequences")

    # Sort by length for efficient batching
    order = sorted(range(n_total), key=lambda i: len(all_seqs[i]))
    all_meta = [all_meta[i] for i in order]
    all_seqs = [all_seqs[i] for i in order]

    # ---- Process in batches ----
    print(
        f"\nRunning masked reconstruction "
        f"({args.mask_rate*100:.0f}% mask, {args.n_seeds} seeds) ..."
    )
    result_rows = []
    t0 = time.time()

    for b_start in range(0, n_total, args.batch_size):
        b_meta = all_meta[b_start : b_start + args.batch_size]
        b_seqs = all_seqs[b_start : b_start + args.batch_size]

        try:
            batch_results = process_batch(
                b_seqs, model, tokenizer, device,
                args.mask_rate, args.n_seeds, args.max_len,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(
                    f"\n  [OOM] batch starting at {b_start}: "
                    f"retrying at batch_size=1 on CPU ..."
                )
                model.cpu()
                cpu = torch.device("cpu")
                batch_results = []
                for seq in b_seqs:
                    r = process_batch(
                        [seq], model, tokenizer, cpu,
                        args.mask_rate, args.n_seeds, args.max_len,
                    )
                    batch_results.extend(r)
                model.to(device)
            else:
                raise

        for i, (acc, label, split) in enumerate(b_meta):
            br = batch_results[i]
            row = {
                "accession":          acc,
                "label":              label,
                "split":              split,
                "length":             len(b_seqs[i]),
                "n_masked":           br["n_masked"],
                "mean_perplexity":    br["mean_perplexity"],
                "mean_recovery_rate": br["mean_recovery_rate"],
                "mean_log_likelihood": br["mean_log_likelihood"],
            }
            for s in range(args.n_seeds):
                row[f"seed{s}_ppl"]      = br["seed_ppls"][s]
                row[f"seed{s}_recovery"] = br["seed_recoveries"][s]
                row[f"seed{s}_ll"]       = br["seed_lls"][s]
            result_rows.append(row)

        done = min(b_start + args.batch_size, n_total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 1
        eta_min = (n_total - done) / rate / 60
        print(
            f"\r  {done:,}/{n_total:,} seqs | "
            f"{rate:.1f} seq/s | ETA {eta_min:.1f} min",
            end="", flush=True,
        )

    total_min = (time.time() - t0) / 60
    print(f"\nTotal time: {total_min:.1f} min")

    # ---- Save TSV ----
    tsv_path = out_dir / "per_sequence_results.tsv"
    if result_rows:
        header = list(result_rows[0].keys())
        with open(tsv_path, "w") as fh:
            fh.write("\t".join(header) + "\n")
            for row in result_rows:
                fh.write("\t".join(str(row[k]) for k in header) + "\n")
        print(f"\nSaved: {tsv_path}  ({len(result_rows):,} rows)")

    # ---- Summary statistics ----
    viral_ppl    = [r["mean_perplexity"]      for r in result_rows if r["label"] == "viral"]
    nonviral_ppl = [r["mean_perplexity"]      for r in result_rows if r["label"] == "nonviral"]
    viral_rec    = [r["mean_recovery_rate"]   for r in result_rows if r["label"] == "viral"]
    nonviral_rec = [r["mean_recovery_rate"]   for r in result_rows if r["label"] == "nonviral"]
    viral_ll     = [r["mean_log_likelihood"]  for r in result_rows if r["label"] == "viral"]
    nonviral_ll  = [r["mean_log_likelihood"]  for r in result_rows if r["label"] == "nonviral"]

    mwu_ppl = stats.mannwhitneyu(viral_ppl, nonviral_ppl, alternative="two-sided")
    mwu_rec = stats.mannwhitneyu(viral_rec, nonviral_rec, alternative="two-sided")
    mwu_ll  = stats.mannwhitneyu(viral_ll, nonviral_ll, alternative="two-sided")

    binary_labels = [1 if r["label"] == "viral" else 0 for r in result_rows]
    # PPL AUC: higher PPL → viral, so use perplexity directly as score
    auroc_ppl = roc_auc_score(binary_labels, [r["mean_perplexity"] for r in result_rows])
    # Recovery AUC: lower recovery → viral, so use -recovery
    auroc_rec = roc_auc_score(binary_labels, [-r["mean_recovery_rate"] for r in result_rows])
    # LL AUC: lower (more negative) LL → viral, so use -LL
    auroc_ll = roc_auc_score(binary_labels, [-r["mean_log_likelihood"] for r in result_rows])

    summary = {
        "model": args.model,
        "hf_model_name": hf_model_name,
        "n_viral":   len(viral_ppl),
        "n_nonviral": len(nonviral_ppl),
        "mask_rate": args.mask_rate,
        "n_seeds":   args.n_seeds,
        "viral_mean_perplexity":    float(np.mean(viral_ppl)),
        "viral_std_perplexity":     float(np.std(viral_ppl)),
        "nonviral_mean_perplexity": float(np.mean(nonviral_ppl)),
        "nonviral_std_perplexity":  float(np.std(nonviral_ppl)),
        "viral_mean_recovery":    float(np.mean(viral_rec)),
        "viral_std_recovery":     float(np.std(viral_rec)),
        "nonviral_mean_recovery": float(np.mean(nonviral_rec)),
        "nonviral_std_recovery":  float(np.std(nonviral_rec)),
        "viral_mean_log_likelihood":    float(np.mean(viral_ll)),
        "viral_std_log_likelihood":     float(np.std(viral_ll)),
        "nonviral_mean_log_likelihood": float(np.mean(nonviral_ll)),
        "nonviral_std_log_likelihood":  float(np.std(nonviral_ll)),
        "mannwhitneyu_perplexity": {
            "statistic": float(mwu_ppl.statistic),
            "pvalue":    float(mwu_ppl.pvalue),
        },
        "mannwhitneyu_recovery": {
            "statistic": float(mwu_rec.statistic),
            "pvalue":    float(mwu_rec.pvalue),
        },
        "mannwhitneyu_log_likelihood": {
            "statistic": float(mwu_ll.statistic),
            "pvalue":    float(mwu_ll.pvalue),
        },
        "auroc_perplexity":     float(auroc_ppl),
        "auroc_recovery":       float(auroc_rec),
        "auroc_log_likelihood": float(auroc_ll),
        "note": "auroc_perplexity: higher PPL = viral (AUC>0.5 means viral has higher PPL). "
                "auroc_log_likelihood: lower LL = viral (AUC>0.5 means viral has lower LL).",
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== RESULTS ===")
    print(f"Model:   {args.model} ({hf_model_name})")
    print(f"Viral    perplexity: {summary['viral_mean_perplexity']:.3f}"
          f" +/- {summary['viral_std_perplexity']:.3f}")
    print(f"Nonviral perplexity: {summary['nonviral_mean_perplexity']:.3f}"
          f" +/- {summary['nonviral_std_perplexity']:.3f}")
    print(f"Viral    log-likelihood: {summary['viral_mean_log_likelihood']:.3f}"
          f" +/- {summary['viral_std_log_likelihood']:.3f}")
    print(f"Nonviral log-likelihood: {summary['nonviral_mean_log_likelihood']:.3f}"
          f" +/- {summary['nonviral_std_log_likelihood']:.3f}")
    print(f"Viral    recovery:   {summary['viral_mean_recovery']:.4f}"
          f" +/- {summary['viral_std_recovery']:.4f}")
    print(f"Nonviral recovery:   {summary['nonviral_mean_recovery']:.4f}"
          f" +/- {summary['nonviral_std_recovery']:.4f}")
    print(f"\nAUC-ROC (PPL):       {auroc_ppl:.4f}")
    print(f"AUC-ROC (LL):        {auroc_ll:.4f}")
    print(f"AUC-ROC (Recovery):  {auroc_rec:.4f}")
    print(f"Mann-Whitney PPL p:  {mwu_ppl.pvalue:.2e}")
    print(f"Mann-Whitney LL  p:  {mwu_ll.pvalue:.2e}")

    # ---- Quick KDE figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, title in [
        (axes[0], (viral_ppl, nonviral_ppl), "Perplexity"),
        (axes[1], (viral_ll, nonviral_ll), "Log-Likelihood"),
    ]:
        ax.hist(vals[0], bins=80, density=True, alpha=0.5, label="viral", color="tab:red")
        ax.hist(vals[1], bins=80, density=True, alpha=0.5, label="nonviral", color="tab:blue")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"{args.model} — {title}")
    fig.tight_layout()
    fig_path = out_dir / "distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
