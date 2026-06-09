#!/usr/bin/env python3
"""
run_masked_reconstruction_api.py
=================================
ESM3-small (Forge API) masked token reconstruction experiment.

Same methodology as run_masked_reconstruction.py but using the
EvolutionaryScale Forge API (esm3-small-2024-08) instead of ESM3 Open
loaded locally. Allows comparison between the open-source model and
the API-hosted version.

Resumable: if interrupted, re-run with the same --out_dir and it will
skip already-completed sequences and continue from where it left off.

Usage:
    export FORGE_TOKEN=<your_token>
    python scripts/run_masked_reconstruction_api.py \\
        --data_dir  /path/to/esm_viral_probe/datasets/human_virus/data/processed \\
        --out_dir   /path/to/esm3_masked_reconstruction/results/esm3_small_api \\
        --model     esm3-small-2024-08 \\
        --mask_rate 0.15 \\
        --n_seeds   3
"""

import argparse
import csv
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

# ---------------------------------------------------------------------------
# ESM package imports — with same compatibility patch as local model script
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
# Resume helpers
# ---------------------------------------------------------------------------
def load_completed(tsv_path: Path) -> dict[str, dict]:
    """Load already-completed rows from TSV. Returns {accession: row_dict}."""
    completed = {}
    if not tsv_path.exists():
        return completed
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            completed[row["accession"]] = {
                k: (float(v) if k not in ("accession", "label", "split") else v)
                for k, v in row.items()
            }
    return completed


# ---------------------------------------------------------------------------
# Per-sequence API inference
# ---------------------------------------------------------------------------
def process_sequence_api(
    seq: str,
    tokenizer,
    client,
    mask_rate: float,
    n_seeds: int,
    max_len: int,
) -> dict:
    """
    Run masked reconstruction for a single sequence via Forge API.

    Tokenizes locally (no API call), applies mask tokens to the tensor,
    then calls client.logits() once per seed (1 API call per seed).

    Returns dict with mean_perplexity, mean_recovery_rate, and per-seed values.
    """
    from esm.sdk.api import ESMProteinTensor, ESMProteinError, LogitsConfig

    seq_trunc = seq[:max_len]

    # Tokenize locally — same as local model script
    encoded = tokenizer(
        [seq_trunc],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len + 2,  # +2 for BOS and EOS
    )
    original_ids = encoded["input_ids"][0]  # [L+2], 1D int tensor
    token_len = original_ids.shape[0]       # L + 2

    # Amino acid positions in token tensor: 1 .. token_len-2
    # (excludes BOS at 0 and EOS at token_len-1)
    res_pos = list(range(1, token_len - 1))
    n_mask = max(1, int(len(res_pos) * mask_rate))

    config = LogitsConfig(sequence=True)

    nlls = []
    seed_recoveries = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(res_pos), size=n_mask, replace=False)
        mask_positions = [res_pos[j] for j in chosen_idx]  # token-level positions

        # Apply masks to a copy of the token tensor
        masked_ids = original_ids.clone()
        for p in mask_positions:
            masked_ids[p] = MASK_TOKEN_ID

        protein_tensor = ESMProteinTensor(sequence=masked_ids)

        for attempt in range(3):
            try:
                output = client.logits(protein_tensor, config)
                if isinstance(output, ESMProteinError):
                    raise RuntimeError(f"logits API error: {output.error_msg}")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 5 * (2 ** attempt)
                print(f"\n  Retry {attempt + 1}/3: {e}. Waiting {wait}s...", flush=True)
                time.sleep(wait)

        # Extract sequence logits — shape may be [1, L+2, V] or [L+2, V]
        logits = output.logits.sequence
        if isinstance(logits, torch.Tensor):
            logits = logits.float().cpu()
        else:
            logits = torch.tensor(np.array(logits, dtype=np.float32))
        if logits.ndim == 3:
            logits = logits[0]  # remove batch dim → [L+2, vocab_size]

        # Index at masked positions (already include BOS offset)
        mp = torch.tensor(mask_positions, dtype=torch.long)
        lg = logits[mp]               # [n_masked, vocab_size]
        tr = original_ids[mp]         # [n_masked], true token IDs

        nll = F.cross_entropy(lg, tr).item()
        recovery = (lg.argmax(dim=-1) == tr).float().mean().item()

        nlls.append(nll)
        seed_recoveries.append(float(recovery))

    mean_perplexity = float(np.exp(np.mean(nlls)))
    seed_ppls = [float(np.exp(n)) for n in nlls]

    return {
        "mean_perplexity":    mean_perplexity,
        "mean_recovery_rate": float(np.mean(seed_recoveries)),
        "seed_ppls":          seed_ppls,
        "seed_recoveries":    seed_recoveries,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def compute_and_save_summary(
    result_rows: list[dict],
    out_dir: Path,
    mask_rate: float,
    n_seeds: int,
) -> None:
    viral_ppl    = [r["mean_perplexity"]    for r in result_rows if r["label"] == "viral"]
    nonviral_ppl = [r["mean_perplexity"]    for r in result_rows if r["label"] == "nonviral"]
    viral_rec    = [r["mean_recovery_rate"] for r in result_rows if r["label"] == "viral"]
    nonviral_rec = [r["mean_recovery_rate"] for r in result_rows if r["label"] == "nonviral"]

    mwu_ppl = stats.mannwhitneyu(viral_ppl, nonviral_ppl, alternative="two-sided")
    mwu_rec = stats.mannwhitneyu(viral_rec, nonviral_rec, alternative="two-sided")

    binary_labels = [1 if r["label"] == "viral" else 0 for r in result_rows]
    auroc_ppl = roc_auc_score(binary_labels, [-r["mean_perplexity"]    for r in result_rows])
    auroc_rec = roc_auc_score(binary_labels, [ r["mean_recovery_rate"] for r in result_rows])

    summary = {
        "n_viral":   len(viral_ppl),
        "n_nonviral": len(nonviral_ppl),
        "mask_rate": mask_rate,
        "n_seeds":   n_seeds,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",  required=True,
                        help="Directory with viral/nonviral {train,val,test}.faa files")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory (TSV + JSON)")
    parser.add_argument("--model",     default="esm3-small-2024-08",
                        help="Forge model name (default: esm3-small-2024-08)")
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--n_seeds",   type=int,   default=3)
    parser.add_argument("--max_len",   type=int,   default=1022)
    parser.add_argument("--timeout",   type=int,   default=120,
                        help="Per-request timeout in seconds (default: 120)")
    args = parser.parse_args()

    forge_token = os.environ.get("FORGE_TOKEN", "")
    if not forge_token:
        sys.exit("ERROR: FORGE_TOKEN environment variable not set.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "per_sequence_results.tsv"

    print("=" * 60)
    print("ESM3 Small API — Masked Reconstruction Experiment")
    print(f"Model:      {args.model}")
    print(f"Mask rate:  {args.mask_rate}")
    print(f"N seeds:    {args.n_seeds}")
    print(f"Output:     {out_dir}")
    print("=" * 60)

    # ---- Load tokenizer locally (no API call needed) ----
    print("\nLoading EsmSequenceTokenizer (local)...")
    tokenizer = EsmSequenceTokenizer()

    # ---- Connect to Forge API ----
    try:
        from esm.sdk.forge import ESM3ForgeInferenceClient
        from esm.sdk.api import ESMProteinTensor, ESMProteinError, LogitsConfig
    except ImportError:
        sys.exit("ERROR: esm.sdk.forge not found. Install with: pip install esm httpx")

    print(f"Connecting to Forge API ({args.model})...")
    client = ESM3ForgeInferenceClient(
        model=args.model,
        url="https://forge.evolutionaryscale.ai",
        token=forge_token,
        request_timeout=args.timeout,
    )

    # Sanity check — short sequence
    print("  Sanity check (1 sequence)...")
    _enc = tokenizer(["MKTAYIAKQR"], return_tensors="pt", padding=False,
                     truncation=True, max_length=14)
    _ids = _enc["input_ids"][0]
    _pt = ESMProteinTensor(sequence=_ids)
    _out = client.logits(_pt, LogitsConfig(sequence=True))
    if isinstance(_out, ESMProteinError):
        sys.exit(f"ERROR: Forge API sanity check failed: {_out.error_msg}")
    _logits = _out.logits.sequence
    _shape = list(_logits.shape) if isinstance(_logits, torch.Tensor) else f"type={type(_logits)}"
    print(f"  logits shape: {_shape}  ✓")

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

    # ---- Resume: load already-completed rows ----
    completed = load_completed(tsv_path)
    print(f"\nAlready completed: {len(completed):,} sequences")

    # TSV header — matches original script's column layout
    header = (
        ["accession", "label", "split", "length",
         "mean_perplexity", "mean_recovery_rate"]
        + [f"seed{s}_ppl"      for s in range(args.n_seeds)]
        + [f"seed{s}_recovery" for s in range(args.n_seeds)]
    )

    # Open TSV for appending; write header only if new file
    write_header = not tsv_path.exists()
    tsv_fh = open(tsv_path, "a", buffering=1)  # line-buffered for safety
    if write_header:
        tsv_fh.write("\t".join(header) + "\n")
        tsv_fh.flush()

    result_rows = list(completed.values())

    # ---- Process sequences ----
    n_todo = sum(1 for acc, _, _ in all_meta if acc not in completed)
    print(f"\nRunning masked reconstruction "
          f"({args.mask_rate * 100:.0f}% mask, {args.n_seeds} seeds)...")
    print(f"  To process: {n_todo:,} sequences")

    t0 = time.time()
    n_done = 0

    for (acc, label, split), seq in zip(all_meta, all_seqs):
        if acc in completed:
            continue

        try:
            br = process_sequence_api(
                seq, tokenizer, client,
                args.mask_rate, args.n_seeds, args.max_len,
            )
        except Exception as e:
            print(f"\nFATAL: {acc} failed after retries: {e}")
            tsv_fh.close()
            raise

        row_dict = {
            "accession":          acc,
            "label":              label,
            "split":              split,
            "length":             len(seq),
            "mean_perplexity":    br["mean_perplexity"],
            "mean_recovery_rate": br["mean_recovery_rate"],
        }
        for s in range(args.n_seeds):
            row_dict[f"seed{s}_ppl"]      = br["seed_ppls"][s]
            row_dict[f"seed{s}_recovery"] = br["seed_recoveries"][s]

        tsv_fh.write("\t".join(str(row_dict[k]) for k in header) + "\n")
        result_rows.append(row_dict)
        n_done += 1

        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 1
        eta_h = (n_todo - n_done) / rate / 3600
        print(
            f"\r  {n_done}/{n_todo} new | "
            f"{len(result_rows)}/{n_total} total | "
            f"{rate:.2f} seq/s | ETA {eta_h:.1f}h",
            end="", flush=True,
        )

    tsv_fh.close()
    elapsed_total = (time.time() - t0) / 3600
    print(f"\nNew sequences processed: {n_done}")
    print(f"Total elapsed: {elapsed_total:.2f}h")
    print(f"Saved: {tsv_path}  ({len(result_rows):,} rows)")

    if len(result_rows) < 10:
        print("Not enough results for summary statistics — exiting.")
        return

    # ---- Summary statistics ----
    compute_and_save_summary(result_rows, out_dir, args.mask_rate, args.n_seeds)
    print("\nDone.")


if __name__ == "__main__":
    main()
