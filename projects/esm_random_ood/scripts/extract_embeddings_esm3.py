#!/usr/bin/env python3
"""
extract_embeddings_esm3.py
==========================
Extract per-sequence mean-pool embeddings from ESM3 open
(EvolutionaryScale/esm3-sm-open-v1 / "esm3-open").

Requires the `esm` package from EvolutionaryScale:
    pip install esm

The ESM3-open model requires accepting EvolutionaryScale's non-commercial
license on HuggingFace before first download (one-time, no token needed
after acceptance).  Set HF_TOKEN if prompted:
    export HF_TOKEN="hf_..."

Usage:
    python extract_embeddings_esm3.py \
        --fasta   ../data/processed/viral_train.faa \
        --outfile ../data/embeddings/esm3_open/viral_train.npz \
        --model   esm3-open \
        --batch_size 8 \
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
# ESM package imports — helpful error if not installed
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
    from esm.models.esm3 import ESM3
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
    model: ESM3,
    tokenizer: EsmSequenceTokenizer,
    device: torch.device,
    max_len: int = 1022,
) -> np.ndarray:
    """
    Tokenize a batch of sequences, run ESM3 forward pass (sequence-only),
    and return mean-pooled last-layer embeddings [B, D].

    BOS (position 0) and EOS (last real token) are excluded from mean pooling.
    """
    seqs_truncated = [s[:max_len] for s in sequences]

    encoded = tokenizer(
        seqs_truncated,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2,  # +2 for BOS/EOS tokens
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
    residue_mask[:, 0, :] = 0.0
    for i, length in enumerate(seq_lens):
        residue_mask[i, length - 1, :] = 0.0

    # Mean pool over residue positions
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
    parser.add_argument("--model",      default="esm3-open",
                        help="ESM3 model key (default: esm3-open = esm3_sm_open_v1)")
    parser.add_argument("--cache_dir",  default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default 8; ESM3 is large ~1.4B params)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len",    type=int, default=1022,
                        help="Max sequence length (truncate beyond this)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    # ---- Load tokenizer ----
    print(f"\nLoading EsmSequenceTokenizer...")
    tokenizer = EsmSequenceTokenizer()

    # ---- Load model ----
    print(f"Loading ESM3 model: {args.model} ...")
    model = ESM3.from_pretrained(args.model)
    # ESM3 checkpoint loads in bfloat16 but some internal scalars (e.g. average_plddt)
    # remain float32, causing a dtype mismatch in plddt_projection. Cast to float32.
    model.float().eval().to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")

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
