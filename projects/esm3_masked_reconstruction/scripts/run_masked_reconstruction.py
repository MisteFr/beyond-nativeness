#!/usr/bin/env python3
"""
run_masked_reconstruction.py
============================
ESM3-open masked token reconstruction experiment.

Masks MASK_RATE fraction of amino acid positions in each sequence and
measures how well ESM3-open reconstructs them. Compares reconstruction
quality between viral and non-viral sequences from the human-virus dataset.

Metrics (per sequence, averaged over N_SEEDS masking seeds):
  - perplexity:     exp(mean cross-entropy at masked positions)
  - recovery_rate:  fraction of masked tokens predicted correctly (top-1)

Hypothesis: ESM3 pretraining may produce systematically different
perplexity on viral vs non-viral sequences, providing a zero-shot
signal about what the model "knows".

Usage:
    python scripts/run_masked_reconstruction.py \\
        --data_dir  /path/to/esm_viral_probe/datasets/human_virus/data/processed \\
        --out_dir   /path/to/esm3_masked_reconstruction/results \\
        --cache_dir /path/to/esm_viral_probe/data/hf_cache \\
        --batch_size 4 \\
        --mask_rate 0.15 \\
        --n_seeds 3 \\
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
# ESM package imports — with compatibility patch for transformers 4.44+
# ---------------------------------------------------------------------------
try:
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer as _EsmTok

    _ESM_SPECIAL = {
        "cls_token": "<cls>", "eos_token": "<eos>", "pad_token": "<pad>",
        "unk_token": "<unk>", "mask_token": "<mask>", "bos_token": "<cls>",
        "sep_token": "<eos>",
    }
    for _attr in list(_ESM_SPECIAL):
        _prop = _EsmTok.__dict__.get(_attr)
        if isinstance(_prop, property) and _prop.fset is None:
            setattr(_EsmTok, _attr, property(_prop.fget, lambda self, v: None))
    _EsmTok._get_token = lambda self, name: _ESM_SPECIAL.get(name, "<unk>")
    EsmSequenceTokenizer = _EsmTok

    from esm.models.esm3 import ESM3
except ImportError:
    sys.exit(
        "ERROR: EvolutionaryScale ESM package not found.\n"
        "Install with:  pip install esm httpx"
    )

try:
    from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN as MASK_TOKEN_ID
except ImportError:
    MASK_TOKEN_ID = 32  # fallback — verified value for esm 3.x


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
    positions (excluding BOS/EOS) for each of n_seeds random seeds, run
    ESM3 forward pass, and compute per-seed NLL and recovery rate.

    Returns a list of dicts (one per sequence) with mean and per-seed metrics.
    """
    seqs_trunc = [s[:max_len] for s in batch_seqs]
    B = len(seqs_trunc)

    encoded = tokenizer(
        seqs_trunc,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2,  # +2 for BOS and EOS tokens
    )
    original_ids = encoded["input_ids"].to(device)   # [B, L_max]
    seq_lens = encoded["attention_mask"].sum(dim=1)  # [B] — includes BOS + EOS

    nlls = np.zeros((B, n_seeds))
    recoveries = np.zeros((B, n_seeds))

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        masked_ids = original_ids.clone()
        mask_positions = []  # list of lists — one per sequence in batch

        for i in range(B):
            length = seq_lens[i].item()
            # Amino acid positions: 1 .. length-2 (0 = BOS, length-1 = EOS)
            res_pos = list(range(1, length - 1))
            n_mask = max(1, int(len(res_pos) * mask_rate))
            chosen = rng.choice(len(res_pos), size=n_mask, replace=False)
            pos_i = [res_pos[j] for j in chosen]
            mask_positions.append(pos_i)
            for p in pos_i:
                masked_ids[i, p] = MASK_TOKEN_ID

        with torch.no_grad():
            output = model(sequence_tokens=masked_ids)

        logits = output.sequence_logits.float()  # [B, L_max, vocab_size]

        for i in range(B):
            mp = torch.tensor(mask_positions[i], dtype=torch.long, device=device)
            lg = logits[i, mp]            # [n_masked, vocab_size]
            tr = original_ids[i, mp]     # [n_masked]

            nlls[i, seed] = F.cross_entropy(lg, tr).item()
            recoveries[i, seed] = (lg.argmax(dim=-1) == tr).float().mean().item()

    results = []
    for i in range(B):
        results.append({
            "mean_perplexity":    float(np.exp(nlls[i].mean())),
            "mean_recovery_rate": float(recoveries[i].mean()),
            "seed_ppls":          [float(np.exp(nlls[i, s])) for s in range(n_seeds)],
            "seed_recoveries":    [float(recoveries[i, s])    for s in range(n_seeds)],
        })
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(
    viral_ppl, nonviral_ppl,
    viral_rec, nonviral_rec,
    auroc_ppl, p_ppl,
    auroc_rec, p_rec,
    out_path: str,
) -> None:
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    viral_color    = "#E36B45"
    nonviral_color = "#4A90D9"

    def _panel(ax, viral_vals, nonviral_vals, xlabel, auroc, pval, log_x=False):
        for vals, label, color in [
            (np.array(viral_vals),    "Viral",     viral_color),
            (np.array(nonviral_vals), "Non-viral", nonviral_color),
        ]:
            x_vals = np.log10(vals) if log_x else vals
            kde = gaussian_kde(x_vals, bw_method="scott")
            x = np.linspace(x_vals.min(), x_vals.max(), 400)
            ax.fill_between(x, kde(x), alpha=0.25, color=color)
            ax.plot(x, kde(x), color=color, lw=2, label=label)

        pval_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        annotation = f"AUC = {auroc:.3f}\np = {pval_str}"
        ax.text(
            0.97, 0.95, annotation,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="0.75", boxstyle="round,pad=0.3"),
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    _panel(
        axes[0], viral_ppl, nonviral_ppl,
        "Perplexity (log₁₀ scale)", auroc_ppl, p_ppl, log_x=True,
    )
    axes[0].set_title("Reconstruction Perplexity", fontsize=12, pad=8)

    _panel(
        axes[1], viral_rec, nonviral_rec,
        "Recovery rate", auroc_rec, p_rec, log_x=False,
    )
    axes[1].set_title("Token Recovery Rate", fontsize=12, pad=8)

    fig.suptitle(
        "ESM3-open Masked Reconstruction: Viral vs Non-viral",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",   required=True,
                        help="Directory containing viral/nonviral {train,val,test}.faa files")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory (TSV, JSON, figure)")
    parser.add_argument("--cache_dir",  default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mask_rate",  type=float, default=0.15)
    parser.add_argument("--n_seeds",    type=int, default=3)
    parser.add_argument("--max_len",    type=int, default=1022)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("ESM3 Masked Reconstruction Experiment")
    print(f"Device:     {device}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
        print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Mask rate:  {args.mask_rate}")
    print(f"N seeds:    {args.n_seeds}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    # ---- Load model ----
    print("\nLoading EsmSequenceTokenizer...")
    tokenizer = EsmSequenceTokenizer()

    print("Loading ESM3-open (~1.4B params) ...")
    model = ESM3.from_pretrained("esm3-open").float().eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {n_params:.2f}B")

    # Sanity check — verify sequence_logits are available
    print("  Sanity check (forward pass)...")
    with torch.no_grad():
        _tok = tokenizer(
            ["MKTAYIAKQR"], return_tensors="pt",
            padding=True, truncation=True, max_length=15,
        )
        _out = model(sequence_tokens=_tok["input_ids"].to(device))
    assert hasattr(_out, "sequence_logits"), \
        "ERROR: ESM3 output missing sequence_logits — check esm package version"
    print(f"  sequence_logits shape: {list(_out.sequence_logits.shape)}  ✓")
    del _tok, _out

    # ---- Load sequences ----
    print("\nLoading sequences...")
    all_meta = []   # (accession, label, split)
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

    # Sort by length for efficient batching (less wasted padding)
    order = sorted(range(n_total), key=lambda i: len(all_seqs[i]))
    all_meta = [all_meta[i] for i in order]
    all_seqs = [all_seqs[i] for i in order]

    # ---- Process in batches ----
    print(
        f"\nRunning masked reconstruction "
        f"({args.mask_rate*100:.0f}% mask, {args.n_seeds} seeds) ..."
    )
    result_rows = []
    n_batches = (n_total + args.batch_size - 1) // args.batch_size
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
                "accession":         acc,
                "label":             label,
                "split":             split,
                "length":            len(b_seqs[i]),
                "mean_perplexity":   br["mean_perplexity"],
                "mean_recovery_rate": br["mean_recovery_rate"],
            }
            for s in range(args.n_seeds):
                row[f"seed{s}_ppl"]      = br["seed_ppls"][s]
                row[f"seed{s}_recovery"] = br["seed_recoveries"][s]
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
    viral_ppl    = [r["mean_perplexity"]    for r in result_rows if r["label"] == "viral"]
    nonviral_ppl = [r["mean_perplexity"]    for r in result_rows if r["label"] == "nonviral"]
    viral_rec    = [r["mean_recovery_rate"] for r in result_rows if r["label"] == "viral"]
    nonviral_rec = [r["mean_recovery_rate"] for r in result_rows if r["label"] == "nonviral"]

    mwu_ppl = stats.mannwhitneyu(viral_ppl, nonviral_ppl, alternative="two-sided")
    mwu_rec = stats.mannwhitneyu(viral_rec, nonviral_rec, alternative="two-sided")

    # AUC: treat -perplexity as viral score (lower ppl → higher viral score)
    binary_labels = [1 if r["label"] == "viral" else 0 for r in result_rows]
    auroc_ppl = roc_auc_score(binary_labels, [-r["mean_perplexity"]    for r in result_rows])
    auroc_rec = roc_auc_score(binary_labels, [ r["mean_recovery_rate"] for r in result_rows])

    summary = {
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
        "mannwhitneyu_perplexity": {
            "statistic": float(mwu_ppl.statistic),
            "pvalue":    float(mwu_ppl.pvalue),
        },
        "mannwhitneyu_recovery": {
            "statistic": float(mwu_rec.statistic),
            "pvalue":    float(mwu_rec.pvalue),
        },
        "auroc_perplexity": float(auroc_ppl),
        "auroc_recovery":   float(auroc_rec),
        "note": (
            "auroc_perplexity uses -perplexity as viral score "
            "(>0.5 = viral has lower perplexity than non-viral). "
            "auroc_recovery uses recovery_rate as viral score "
            "(>0.5 = viral has higher recovery rate)."
        ),
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== RESULTS ===")
    print(f"Viral    perplexity: {summary['viral_mean_perplexity']:.3f}"
          f" ± {summary['viral_std_perplexity']:.3f}")
    print(f"Nonviral perplexity: {summary['nonviral_mean_perplexity']:.3f}"
          f" ± {summary['nonviral_std_perplexity']:.3f}")
    print(f"Viral    recovery:   {summary['viral_mean_recovery']:.4f}"
          f" ± {summary['viral_std_recovery']:.4f}")
    print(f"Nonviral recovery:   {summary['nonviral_mean_recovery']:.4f}"
          f" ± {summary['nonviral_std_recovery']:.4f}")
    print(f"Mann-Whitney U (ppl): p = {mwu_ppl.pvalue:.4g}")
    print(f"Mann-Whitney U (rec): p = {mwu_rec.pvalue:.4g}")
    print(f"AUC-ROC (ppl):        {auroc_ppl:.4f}")
    print(f"AUC-ROC (rec):        {auroc_rec:.4f}")

    # ---- Figure ----
    fig_path = str(out_dir / "perplexity_distribution.png")
    try:
        make_figure(
            viral_ppl, nonviral_ppl,
            viral_rec, nonviral_rec,
            auroc_ppl, float(mwu_ppl.pvalue),
            auroc_rec, float(mwu_rec.pvalue),
            fig_path,
        )
    except Exception as exc:
        print(f"WARNING: figure generation failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
