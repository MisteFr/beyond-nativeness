#!/usr/bin/env python3
"""
extract_embeddings.py
=====================
Extract per-sequence mean-pool embeddings from a pretrained ESM2 model.

The script processes sequences in batches, extracts the final-layer
representations, averages over residues (mean pooling), and saves the
result as a compressed NumPy archive (.npz).

ESM2 model choices (HuggingFace):
  - facebook/esm2_t6_8M_UR50D        (8M  params, fast)
  - facebook/esm2_t12_35M_UR50D      (35M params)
  - facebook/esm2_t30_150M_UR50D     (150M params)
  - facebook/esm2_t33_650M_UR50D     (650M params, default)
  - facebook/esm2_t36_3B_UR50D       (3B  params, GPU required)

Usage:
  python extract_embeddings.py \
      --fasta   ../data/processed/viral_train.faa \
      --outfile ../data/embeddings/viral_train.npz \
      --model   facebook/esm2_t33_650M_UR50D \
      --batch_size 16 \
      --device  cuda   # or cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def parse_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Yield (accession, sequence) pairs from a FASTA file."""
    header, seq_parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0], "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header is not None:
        yield header.split()[0], "".join(seq_parts)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    sequences: list[str],
    model: EsmModel,
    tokenizer,
    device: torch.device,
    max_len: int = 1022,
) -> np.ndarray:
    """
    Tokenize a batch of sequences, run ESM2 forward pass, and return
    mean-pooled last-layer embeddings [B, D].

    Sequences are truncated to max_len (ESM2 positional encoding limit).
    The padding token is excluded from the mean pool via attention mask.
    """
    # Truncate sequences to model max length
    seqs_truncated = [s[:max_len] for s in sequences]

    encoded = tokenizer(
        seqs_truncated,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2,  # +2 for [CLS] / [EOS] tokens
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    # last_hidden_state: [B, L, D]
    hidden = outputs.last_hidden_state
    # attention_mask: [B, L], 1 for real tokens (including CLS/EOS)
    mask = encoded["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]

    # Exclude CLS (index 0) and EOS (last real token) from pooling:
    # create a mask that is 0 for position 0 and the last non-pad position
    seq_lens = encoded["attention_mask"].sum(dim=1)  # [B]
    residue_mask = mask.clone()
    residue_mask[:, 0, :] = 0  # zero out CLS
    for i, length in enumerate(seq_lens):
        residue_mask[i, length - 1, :] = 0  # zero out EOS

    # Mean pool over residue positions
    pooled = (hidden * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1)
    return pooled.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta",      required=True, help="Input FASTA file")
    parser.add_argument("--outfile",    required=True, help="Output .npz file")
    parser.add_argument("--model",      default="facebook/esm2_t33_650M_UR50D",
                        help="HuggingFace ESM2 model ID")
    parser.add_argument("--cache_dir",  default=None,
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len",    type=int, default=1022,
                        help="Max sequence length (truncate beyond this)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Load model ----
    print(f"\nLoading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = EsmModel.from_pretrained(args.model, cache_dir=args.cache_dir)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")

    # ---- Load sequences ----
    print(f"\nParsing FASTA: {args.fasta} ...")
    records = list(parse_fasta(args.fasta))
    print(f"  Sequences: {len(records):,}")

    # ---- Extract embeddings in batches ----
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)

    accessions = []
    embeddings = []
    n_batches = (len(records) + args.batch_size - 1) // args.batch_size

    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        batch_accs, batch_seqs = zip(*batch)

        try:
            emb = extract_embeddings(batch_seqs, model, tokenizer, device, args.max_len)
        except RuntimeError as e:
            # OOM: fall back to CPU for this batch
            print(f"\n  [WARNING] OOM on batch {i//args.batch_size + 1}: {e}")
            print("  Retrying on CPU...")
            model_cpu = model.cpu()
            emb = extract_embeddings(batch_seqs, model_cpu, tokenizer, torch.device("cpu"), args.max_len)
            model.to(device)

        accessions.extend(batch_accs)
        embeddings.append(emb)

        # Progress reporting
        done = min(i + args.batch_size, len(records))
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(records) - done) / rate if rate > 0 else 0
        print(f"\r  Batch {i//args.batch_size + 1}/{n_batches} | "
              f"{done:,}/{len(records):,} seqs | "
              f"{rate:.1f} seq/s | ETA {eta:.0f}s", end="", flush=True)

    print()  # newline after progress

    # ---- Save ----
    embeddings_arr = np.vstack(embeddings)  # [N, D]
    accessions_arr = np.array(accessions)

    print(f"\nEmbedding matrix shape: {embeddings_arr.shape}")
    print(f"Saving to: {args.outfile} ...")
    np.savez_compressed(
        args.outfile,
        embeddings=embeddings_arr,
        accessions=accessions_arr,
    )
    size_mb = os.path.getsize(args.outfile) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
