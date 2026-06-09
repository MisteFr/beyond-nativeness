#!/usr/bin/env python3
"""ESMC-600M masked reconstruction on a single FASTA file.

Adapted from esm3_masked_reconstruction/scripts/run_masked_reconstruction_esmc.py.
Takes a single FASTA + group label instead of the viral/nonviral split structure.

Usage:
    python scripts/04_masked_reconstruction.py \
        --fasta data/processed/bacteria_clean.faa \
        --label bacteria \
        --out_dir results/masked_reconstruction \
        --model esmc_600m \
        --batch_size 4 \
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

# ---------------------------------------------------------------------------
# ESM package imports with compatibility patch
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
    sys.exit("ERROR: esm package not found. pip install esm httpx")

try:
    from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN as MASK_TOKEN_ID
except ImportError:
    MASK_TOKEN_ID = 32


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


def process_batch(
    batch_seqs: list[str],
    model, tokenizer, device: torch.device,
    mask_rate: float, n_seeds: int, max_len: int,
) -> list[dict]:
    seqs_trunc = [s[:max_len] for s in batch_seqs]
    B = len(seqs_trunc)

    encoded = tokenizer(
        seqs_trunc, return_tensors="pt", padding=True,
        truncation=True, max_length=max_len + 2,
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
            "mean_perplexity": float(np.exp(nlls[i].mean())),
            "mean_recovery_rate": float(recoveries[i].mean()),
            "seed_ppls": [float(np.exp(nlls[i, s])) for s in range(n_seeds)],
            "seed_recoveries": [float(recoveries[i, s]) for s in range(n_seeds)],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--label", required=True, help="Group label (bacteria/archaea/phage)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="esmc_600m", choices=["esmc_300m", "esmc_600m"])
    parser.add_argument("--cache_dir", default=None)
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
    print(f"Masked Reconstruction — {args.label} ({args.model})")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mask rate: {args.mask_rate}, Seeds: {args.n_seeds}")
    print("=" * 60)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    tokenizer = EsmSequenceTokenizer()
    from esm.models.esmc import ESMC
    model = ESMC.from_pretrained(args.model, device=device).eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Sanity check
    with torch.no_grad():
        _tok = tokenizer(["MKTAYIAKQR"], return_tensors="pt", padding=True,
                         truncation=True, max_length=15)
        _out = model(sequence_tokens=_tok["input_ids"].to(device))
    assert hasattr(_out, "sequence_logits")
    del _tok, _out

    # Load sequences
    records = parse_fasta(args.fasta)
    print(f"Loaded {len(records):,} sequences from {args.fasta}")

    # Sort by length for efficient batching
    order = sorted(range(len(records)), key=lambda i: len(records[i][1]))
    records = [records[i] for i in order]

    # Process
    result_rows = []
    t0 = time.time()
    n_total = len(records)

    for b_start in range(0, n_total, args.batch_size):
        batch = records[b_start:b_start + args.batch_size]
        b_accs = [r[0] for r in batch]
        b_seqs = [r[1] for r in batch]

        try:
            batch_results = process_batch(
                b_seqs, model, tokenizer, device,
                args.mask_rate, args.n_seeds, args.max_len,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                model.cpu()
                cpu = torch.device("cpu")
                batch_results = []
                for seq in b_seqs:
                    r = process_batch([seq], model, tokenizer, cpu,
                                      args.mask_rate, args.n_seeds, args.max_len)
                    batch_results.extend(r)
                model.to(device)
            else:
                raise

        for i, acc in enumerate(b_accs):
            br = batch_results[i]
            row = {
                "accession": acc,
                "label": args.label,
                "length": len(b_seqs[i]),
                "mean_perplexity": br["mean_perplexity"],
                "mean_recovery_rate": br["mean_recovery_rate"],
            }
            for s in range(args.n_seeds):
                row[f"seed{s}_ppl"] = br["seed_ppls"][s]
                row[f"seed{s}_recovery"] = br["seed_recoveries"][s]
            result_rows.append(row)

        done = min(b_start + args.batch_size, n_total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 1
        eta_min = (n_total - done) / rate / 60
        print(f"\r  {done:,}/{n_total:,} | {rate:.1f} seq/s | ETA {eta_min:.1f} min",
              end="", flush=True)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # Save TSV
    tsv_path = out_dir / f"{args.label}_ppl.tsv"
    if result_rows:
        header = list(result_rows[0].keys())
        with open(tsv_path, "w") as fh:
            fh.write("\t".join(header) + "\n")
            for row in result_rows:
                fh.write("\t".join(str(row[k]) for k in header) + "\n")
        print(f"Saved: {tsv_path} ({len(result_rows):,} rows)")

    # Summary
    ppls = [r["mean_perplexity"] for r in result_rows]
    recs = [r["mean_recovery_rate"] for r in result_rows]
    summary = {
        "label": args.label,
        "model": args.model,
        "n_sequences": len(ppls),
        "mean_ppl": float(np.mean(ppls)),
        "std_ppl": float(np.std(ppls)),
        "median_ppl": float(np.median(ppls)),
        "mean_recovery": float(np.mean(recs)),
        "std_recovery": float(np.std(recs)),
    }
    json_path = out_dir / f"{args.label}_ppl_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    print(f"\n{args.label}: mean PPL = {summary['mean_ppl']:.3f} +/- {summary['std_ppl']:.3f}")
    print(f"{args.label}: mean recovery = {summary['mean_recovery']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
