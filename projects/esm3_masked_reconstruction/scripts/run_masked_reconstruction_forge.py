#!/usr/bin/env python3
"""
run_masked_reconstruction_forge.py
===================================
Masked token reconstruction via EvolutionaryScale Forge API.

Supports both ESM3 (medium, large) and ESMC (6B) models.
Resumable: re-run with the same --out_dir to continue from where it left off.

Usage:
    export FORGE_TOKEN=<your_token>

    # ESM3 Medium
    python scripts/run_masked_reconstruction_forge.py \
        --data_dir  data \
        --out_dir   results/esm3_medium \
        --model_name esm3-medium-2024-08 \
        --client_type esm3

    # ESM3 Large
    python scripts/run_masked_reconstruction_forge.py \
        --data_dir  data \
        --out_dir   results/esm3_large \
        --model_name esm3-large-2024-03 \
        --client_type esm3

    # ESMC 6B
    python scripts/run_masked_reconstruction_forge.py \
        --data_dir  data \
        --out_dir   results/esmc_6b \
        --model_name esmc-6b-2024-12 \
        --client_type esmc
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
# ESM package imports
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
    sys.exit("ERROR: EvolutionaryScale ESM package not found.")

try:
    from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN as MASK_TOKEN_ID
except ImportError:
    MASK_TOKEN_ID = 32

from esm.sdk.api import ESMProteinTensor, ESMProteinError, LogitsConfig


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------
def parse_fasta(path: str) -> list[tuple[str, str]]:
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
# Forge client factory
# ---------------------------------------------------------------------------
def make_client(client_type: str, model_name: str, token: str, timeout: int):
    """Create either ESM3 or ESMC Forge inference client."""
    if client_type == "esm3":
        from esm.sdk.forge import ESM3ForgeInferenceClient
        return ESM3ForgeInferenceClient(
            model=model_name,
            url="https://forge.evolutionaryscale.ai",
            token=token,
            request_timeout=timeout,
        )
    elif client_type == "esmc":
        from esm.sdk.forge import ESMCForgeInferenceClient
        return ESMCForgeInferenceClient(
            model=model_name,
            url="https://forge.evolutionaryscale.ai",
            token=token,
            request_timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown client_type: {client_type}")


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
    seq_trunc = seq[:max_len]

    encoded = tokenizer(
        [seq_trunc],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len + 2,
    )
    original_ids = encoded["input_ids"][0]
    token_len = original_ids.shape[0]
    res_pos = list(range(1, token_len - 1))
    n_mask = max(1, int(len(res_pos) * mask_rate))

    config = LogitsConfig(sequence=True)
    nlls = []
    seed_recoveries = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(res_pos), size=n_mask, replace=False)
        mask_positions = [res_pos[j] for j in chosen_idx]

        masked_ids = original_ids.clone()
        for p in mask_positions:
            masked_ids[p] = MASK_TOKEN_ID

        protein_tensor = ESMProteinTensor(sequence=masked_ids)

        for attempt in range(3):
            try:
                output = client.logits(protein_tensor, config)
                if isinstance(output, ESMProteinError):
                    raise RuntimeError(
                        f"API error {output.error_code}: {output.error_msg}"
                    )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 5 * (2 ** attempt)
                print(f"\n  Retry {attempt+1}/3: {e}. Waiting {wait}s...", flush=True)
                time.sleep(wait)

        logits = output.logits.sequence
        if isinstance(logits, torch.Tensor):
            logits = logits.float().cpu()
        else:
            logits = torch.tensor(np.array(logits, dtype=np.float32))
        if logits.ndim == 3:
            logits = logits[0]

        mp = torch.tensor(mask_positions, dtype=torch.long)
        lg = logits[mp]
        tr = original_ids[mp]

        nll = F.cross_entropy(lg, tr).item()
        recovery = (lg.argmax(dim=-1) == tr).float().mean().item()

        nlls.append(nll)
        seed_recoveries.append(float(recovery))

    return {
        "mean_perplexity":    float(np.exp(np.mean(nlls))),
        "mean_recovery_rate": float(np.mean(seed_recoveries)),
        "seed_ppls":          [float(np.exp(n)) for n in nlls],
        "seed_recoveries":    seed_recoveries,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def compute_and_save_summary(
    result_rows: list[dict],
    out_dir: Path,
    model_name: str,
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
        "model": model_name,
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
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    print(f"\n=== RESULTS ({model_name}) ===")
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
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--model_name",  required=True,
                        help="Forge model name (e.g. esm3-medium-2024-08, esmc-6b-2024-12)")
    parser.add_argument("--client_type", required=True, choices=["esm3", "esmc"],
                        help="Client type: esm3 or esmc")
    parser.add_argument("--mask_rate",   type=float, default=0.15)
    parser.add_argument("--n_seeds",     type=int,   default=3)
    parser.add_argument("--max_len",     type=int,   default=1022)
    parser.add_argument("--timeout",     type=int,   default=120)
    args = parser.parse_args()

    forge_token = os.environ.get("FORGE_TOKEN", "")
    if not forge_token:
        sys.exit("ERROR: FORGE_TOKEN environment variable not set.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "per_sequence_results.tsv"

    print("=" * 60)
    print(f"Forge API Masked Reconstruction — {args.model_name}")
    print(f"Client type: {args.client_type}")
    print(f"Mask rate:   {args.mask_rate}")
    print(f"N seeds:     {args.n_seeds}")
    print(f"Output:      {out_dir}")
    print("=" * 60)

    # ---- Load tokenizer locally ----
    print("\nLoading EsmSequenceTokenizer (local)...")
    tokenizer = EsmSequenceTokenizer()

    # ---- Connect to Forge API ----
    print(f"Connecting to Forge API ({args.model_name})...")
    client = make_client(args.client_type, args.model_name, forge_token, args.timeout)

    # ---- Sanity check ----
    print("  Sanity check (1 test sequence)...")
    _enc = tokenizer(["MKTAYIAKQR"], return_tensors="pt", padding=False,
                     truncation=True, max_length=14)
    _ids = _enc["input_ids"][0]
    _pt = ESMProteinTensor(sequence=_ids)
    try:
        _out = client.logits(_pt, LogitsConfig(sequence=True))
    except Exception as e:
        print(f"\nERROR: Sanity check API call failed for model '{args.model_name}':")
        print(f"  {e}")
        print(f"\nCandidate model names to try:")
        if args.client_type == "esm3":
            print("  esm3-small-2024-08")
            print("  esm3-medium-2024-08")
            print("  esm3-medium-2024-03")
            print("  esm3-large-2024-03")
            print("  esm3-large-2024-11")
        else:
            print("  esmc-300m-2024-12")
            print("  esmc-600m-2024-12")
            print("  esmc-6b-2024-12")
        sys.exit(1)

    if isinstance(_out, ESMProteinError):
        print(f"\nERROR: API returned error: {_out.error_code} {_out.error_msg}")
        sys.exit(1)

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

    # ---- Resume ----
    completed = load_completed(tsv_path)
    print(f"\nAlready completed: {len(completed):,} sequences")

    header = (
        ["accession", "label", "split", "length",
         "mean_perplexity", "mean_recovery_rate"]
        + [f"seed{s}_ppl"      for s in range(args.n_seeds)]
        + [f"seed{s}_recovery" for s in range(args.n_seeds)]
    )

    write_header = not tsv_path.exists()
    tsv_fh = open(tsv_path, "a", buffering=1)
    if write_header:
        tsv_fh.write("\t".join(header) + "\n")
        tsv_fh.flush()

    result_rows = list(completed.values())

    # ---- Process sequences ----
    n_todo = sum(1 for acc, _, _ in all_meta if acc not in completed)
    print(f"\nRunning masked reconstruction "
          f"({args.mask_rate*100:.0f}% mask, {args.n_seeds} seeds)...")
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

    compute_and_save_summary(
        result_rows, out_dir, args.model_name,
        args.mask_rate, args.n_seeds,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
