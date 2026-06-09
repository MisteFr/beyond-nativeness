#!/usr/bin/env python3
"""Score a single arbitrary FASTA with an ESM2 model using the masked
reconstruction pipeline in `run_masked_reconstruction_esm2.py`. Writes a TSV
with the same schema as the main per_sequence_results.tsv.

Used to compute ESM2-650M masked-reconstruction PPL on the shuffled_viral,
shuffled_nonviral, and random_uniform control pools for the appendix
nativeness figure (viral/nonviral across the ESM family).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse `process_batch` and `parse_fasta` from the main ESM2 script.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_masked_reconstruction_esm2 import MODEL_MAP, parse_fasta, process_batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    p.add_argument("--fasta", required=True)
    p.add_argument("--label", required=True,
                   help="Group label written into the `label` column")
    p.add_argument("--out_tsv", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--mask_rate", type=float, default=0.15)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--max_len", type=int, default=1022)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    args = p.parse_args()

    device = torch.device(args.device)
    if args.cache_dir:
        import os
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    from transformers import EsmForMaskedLM, AutoTokenizer
    hf_model_name = MODEL_MAP[args.model]
    print(f"Loading {hf_model_name} (dtype={args.dtype}) …")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    model = EsmForMaskedLM.from_pretrained(
        hf_model_name, torch_dtype=dtype_map[args.dtype]
    ).to(device).eval()

    records = parse_fasta(args.fasta)
    print(f"Loaded {len(records):,} sequences from {args.fasta}")

    # Sort by length for padding efficiency; preserve accession order at write time.
    indexed = list(enumerate(records))
    indexed.sort(key=lambda kv: len(kv[1][1]))
    order_ix = [i for i, _ in indexed]
    seqs = [kv[1][1] for kv in indexed]
    accs = [kv[1][0] for kv in indexed]

    per_seq: list[dict] = []
    t0 = time.time()
    for start in range(0, len(seqs), args.batch_size):
        batch = seqs[start : start + args.batch_size]
        out = process_batch(
            batch, model, tokenizer, device,
            args.mask_rate, args.n_seeds, args.max_len,
        )
        per_seq.extend(out)
        if start % (args.batch_size * 25) == 0:
            print(f"  {start+len(batch):5d}/{len(seqs)}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    # Restore input order for the output TSV.
    by_input_ix = [None] * len(seqs)
    for pos, orig_ix in enumerate(order_ix):
        by_input_ix[orig_ix] = (accs[pos], per_seq[pos], len(seqs[pos]))

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "accession", "label", "split", "length", "n_masked",
        "mean_perplexity", "mean_recovery_rate", "mean_log_likelihood",
        "seed0_ppl", "seed0_recovery", "seed0_ll",
        "seed1_ppl", "seed1_recovery", "seed1_ll",
        "seed2_ppl", "seed2_recovery", "seed2_ll",
    ]
    with out_tsv.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for entry in by_input_ix:
            if entry is None:
                continue
            acc, d, length = entry
            row = [
                acc, args.label, "all", str(length), str(d["n_masked"]),
                f"{d['mean_perplexity']:.6f}",
                f"{d['mean_recovery_rate']:.6f}",
                f"{d['mean_log_likelihood']:.6f}",
            ]
            for s in range(3):
                row.append(f"{d['seed_ppls'][s]:.6f}")
                row.append(f"{d['seed_recoveries'][s]:.6f}")
                row.append(f"{d['seed_lls'][s]:.6f}")
            fh.write("\t".join(row) + "\n")
    print(f"Wrote {out_tsv}  ({len(seqs)} rows)")


if __name__ == "__main__":
    main()
