#!/usr/bin/env python3
"""
extract_embeddings_esmc.py
==========================
Extract per-sequence mean-pool embeddings from EvolutionaryScale ESMC (ESM3 open).

Requires the `esm` package from EvolutionaryScale:
    pip install esm

Supported models (pass short key to --model):
    esmc_300m   →  EvolutionaryScale/esmc-300m-2024-12  (960-dim)
    esmc_600m   →  EvolutionaryScale/esmc-600m-2024-12  (1152-dim)

Usage:
    python extract_embeddings_esmc.py \
        --fasta   ../data/processed/viral_train.faa \
        --outfile ../data/embeddings/esm3_300m/viral_train.npz \
        --model   esmc_300m \
        --batch_size 16 \
        --device  cuda
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ESM package import — helpful error if not installed
# ---------------------------------------------------------------------------

try:
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer as _EsmTok
    # Patch: transformers 4.44+ removed __getattr__ and made setattr strict for
    # special tokens that EsmSequenceTokenizer defines as read-only properties.
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
    from esm.models.esmc import ESMC
except ImportError:
    sys.exit(
        "ERROR: EvolutionaryScale ESM package not found.\n"
        "Install with:  pip install esm httpx\n"
        "Then re-run this script."
    )


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

def extract_embeddings_batch(
    sequences: list[str],
    model: ESMC,
    tokenizer: EsmSequenceTokenizer,
    device: torch.device,
    max_len: int = 1022,
) -> np.ndarray:
    """
    Tokenize a batch of sequences, run ESMC forward pass, and return
    mean-pooled last-layer embeddings [B, D].

    BOS (position 0) and EOS (last real token) are excluded from mean pooling,
    matching the convention used for ESM2.
    """
    seqs_truncated = [s[:max_len] for s in sequences]

    encoded = tokenizer(
        seqs_truncated,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2,  # +2 for BOS/EOS
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)  # [B, L]

    with torch.no_grad():
        output = model(sequence_tokens=input_ids)

    # output.embeddings: [B, L, D]
    hidden = output.embeddings.float()

    # Build residue mask: exclude BOS (position 0) and EOS (last real token)
    seq_lens = attn_mask.sum(dim=1)  # [B]
    residue_mask = attn_mask.clone().float().unsqueeze(-1)  # [B, L, 1]
    residue_mask[:, 0, :] = 0.0  # zero out BOS
    for i, length in enumerate(seq_lens):
        residue_mask[i, length - 1, :] = 0.0  # zero out EOS

    # Mean pool
    pooled = (hidden * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta",      required=True, help="Input FASTA file")
    parser.add_argument("--outfile",    required=True, help="Output .npz file")
    parser.add_argument("--model",      default="esmc_300m",
                        help="ESMC model key: esmc_300m or esmc_600m (default: esmc_300m)")
    parser.add_argument("--cache_dir",  default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len",    type=int, default=1022,
                        help="Max sequence length (truncate beyond this)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Set HF cache if provided ----
    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    # ---- Load model ----
    print(f"\nLoading ESMC model: {args.model} ...")
    model = ESMC.from_pretrained(args.model)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")

    # ---- Load tokenizer ----
    tokenizer = EsmSequenceTokenizer()
    print(f"  Tokenizer: EsmSequenceTokenizer")

    # ---- Probe hidden dim ----
    with torch.no_grad():
        _test = tokenizer(["MKTAYIAKQRQISFVKSHFSRQ"], return_tensors="pt",
                          padding=True, truncation=True, max_length=24)
        _out = model(sequence_tokens=_test["input_ids"].to(device))
    hidden_dim = _out.embeddings.shape[-1]
    print(f"  Embedding dim: {hidden_dim}")

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
        batch = records[i : i + args.batch_size]
        batch_accs, batch_seqs = zip(*batch)

        try:
            emb = extract_embeddings_batch(batch_seqs, model, tokenizer, device, args.max_len)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  [WARNING] OOM on batch {i // args.batch_size + 1}: retrying on CPU ...")
                torch.cuda.empty_cache()
                model_cpu = model.cpu()
                emb = extract_embeddings_batch(
                    batch_seqs, model_cpu, tokenizer, torch.device("cpu"), args.max_len
                )
                model.to(device)
            else:
                raise

        accessions.extend(batch_accs)
        embeddings.append(emb)

        done = min(i + args.batch_size, len(records))
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(records) - done) / rate if rate > 0 else 0
        print(
            f"\r  Batch {i // args.batch_size + 1}/{n_batches} | "
            f"{done:,}/{len(records):,} seqs | "
            f"{rate:.1f} seq/s | ETA {eta:.0f}s",
            end="",
            flush=True,
        )

    print()

    # ---- Save ----
    embeddings_arr = np.vstack(embeddings)
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
