#!/usr/bin/env python3
"""
run_masked_reconstruction.py — Post-cutoff non-viral proteins
=============================================================
ESMC-600M masked token reconstruction on post-cutoff non-viral proteins.

Reuses the same methodology as esm3_masked_reconstruction but only runs
on the new post-cutoff sequences (existing pre-cutoff results are merged
at analysis time in 03_plot_comparison.py).

Usage:
    python scripts/02_run_masked_reconstruction.py \
        --fasta data/postcutoff_nonviral_filtered.faa \
        --out_dir results/esmc_600m \
        --model esmc_600m \
        --batch_size 4 \
        --mask_rate 0.15 \
        --n_seeds 3
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

except ImportError:
    sys.exit(
        "ERROR: EvolutionaryScale ESM package not found.\n"
        "Install with:  pip install esm httpx"
    )

try:
    from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN as MASK_TOKEN_ID
except ImportError:
    MASK_TOKEN_ID = 32


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
# Core masked reconstruction (identical to esm3_masked_reconstruction)
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
    """Mask, forward pass, compute NLL and recovery for each seed."""
    seqs_trunc = [s[:max_len] for s in batch_seqs]
    B = len(seqs_trunc)

    encoded = tokenizer(
        seqs_trunc,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2,
    )
    original_ids = encoded["input_ids"].to(device)
    seq_lens = encoded["attention_mask"].sum(dim=1)

    nlls = np.zeros((B, n_seeds))
    recoveries = np.zeros((B, n_seeds))

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        masked_ids = original_ids.clone()
        mask_positions = []

        for i in range(B):
            length = seq_lens[i].item()
            res_pos = list(range(1, length - 1))
            n_mask = max(1, int(len(res_pos) * mask_rate))
            chosen = rng.choice(len(res_pos), size=n_mask, replace=False)
            pos_i = [res_pos[j] for j in chosen]
            mask_positions.append(pos_i)
            for p in pos_i:
                masked_ids[i, p] = MASK_TOKEN_ID

        with torch.no_grad():
            output = model(sequence_tokens=masked_ids)

        logits = output.sequence_logits.float()

        for i in range(B):
            mp = torch.tensor(mask_positions[i], dtype=torch.long, device=device)
            lg = logits[i, mp]
            tr = original_ids[i, mp]

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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", required=True,
                        help="Input FASTA of post-cutoff non-viral proteins")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory (TSV, JSON)")
    parser.add_argument("--model", default="esmc_600m",
                        choices=["esmc_300m", "esmc_600m"],
                        help="ESMC model name (default: esmc_600m)")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1022)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"Post-Cutoff Masked Reconstruction — {args.model}")
    print(f"Device:     {device}")
    if device.type == "cuda":
        try:
            print(f"GPU:        {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', None)
            if mem:
                print(f"VRAM:       {mem / 1e9:.1f} GB")
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
    print("\nLoading EsmSequenceTokenizer...")
    tokenizer = EsmSequenceTokenizer()

    print(f"Loading {args.model}...")
    from esm.models.esmc import ESMC
    model = ESMC.from_pretrained(args.model, device=device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")

    # Sanity check
    print("  Sanity check (forward pass)...")
    with torch.no_grad():
        _tok = tokenizer(
            ["MKTAYIAKQR"], return_tensors="pt",
            padding=True, truncation=True, max_length=15,
        )
        _out = model(sequence_tokens=_tok["input_ids"].to(device))
    assert hasattr(_out, "sequence_logits"), \
        "ERROR: ESMC output missing sequence_logits"
    print(f"  sequence_logits shape: {list(_out.sequence_logits.shape)}  OK")
    del _tok, _out

    # ---- Load sequences ----
    print(f"\nLoading sequences from {args.fasta}...")
    records = parse_fasta(args.fasta)
    n_total = len(records)
    print(f"  Total: {n_total:,} sequences")

    all_acc = [r[0] for r in records]
    all_seq = [r[1] for r in records]

    # Sort by length for efficient batching
    order = sorted(range(n_total), key=lambda i: len(all_seq[i]))
    all_acc = [all_acc[i] for i in order]
    all_seq = [all_seq[i] for i in order]

    # ---- Process in batches ----
    print(
        f"\nRunning masked reconstruction "
        f"({args.mask_rate*100:.0f}% mask, {args.n_seeds} seeds) ..."
    )
    result_rows = []
    t0 = time.time()

    for b_start in range(0, n_total, args.batch_size):
        b_acc = all_acc[b_start : b_start + args.batch_size]
        b_seq = all_seq[b_start : b_start + args.batch_size]

        try:
            batch_results = process_batch(
                b_seq, model, tokenizer, device,
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
                for seq in b_seq:
                    r = process_batch(
                        [seq], model, tokenizer, cpu,
                        args.mask_rate, args.n_seeds, args.max_len,
                    )
                    batch_results.extend(r)
                model.to(device)
            else:
                raise

        for i, acc in enumerate(b_acc):
            br = batch_results[i]
            row = {
                "accession":          acc,
                "label":              "postcutoff_nonviral",
                "split":              "all",
                "length":             len(b_seq[i]),
                "mean_perplexity":    br["mean_perplexity"],
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

    # ---- Quick summary ----
    ppls = [r["mean_perplexity"] for r in result_rows]
    recs = [r["mean_recovery_rate"] for r in result_rows]

    summary = {
        "model": args.model,
        "n_sequences": len(result_rows),
        "label": "postcutoff_nonviral",
        "mask_rate": args.mask_rate,
        "n_seeds": args.n_seeds,
        "mean_perplexity": float(np.mean(ppls)),
        "std_perplexity": float(np.std(ppls)),
        "median_perplexity": float(np.median(ppls)),
        "mean_recovery_rate": float(np.mean(recs)),
        "std_recovery_rate": float(np.std(recs)),
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== RESULTS ===")
    print(f"Model:       {args.model}")
    print(f"N sequences: {summary['n_sequences']:,}")
    print(f"Perplexity:  {summary['mean_perplexity']:.3f} +/- {summary['std_perplexity']:.3f}"
          f"  (median {summary['median_perplexity']:.3f})")
    print(f"Recovery:    {summary['mean_recovery_rate']:.4f} +/- {summary['std_recovery_rate']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
