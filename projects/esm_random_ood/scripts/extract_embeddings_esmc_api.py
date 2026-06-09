#!/usr/bin/env python3
"""
extract_embeddings_esmc_api.py
==============================
Extract per-sequence mean-pooled embeddings from the EvolutionaryScale Forge API
for ESMC models (esmc-600m-2024-12, esmc-6b-2024-12).

Uses ESMCForgeInferenceClient (not ESM3ForgeInferenceClient).

Supports checkpoint/resume: partial results are saved every 100 sequences to a
.cache.npy file alongside the output. If the job is interrupted, simply re-run
with the same --outfile and it will pick up where it left off.

Requires the `esm` package from EvolutionaryScale:
    pip install esm httpx

Usage:
    python extract_embeddings_esmc_api.py \\
        --fasta   ../data/processed/viral_train.faa \\
        --outfile ../data/embeddings/esmc_600m/viral_train.npz \\
        --token   <YOUR_FORGE_TOKEN> \\
        --model   esmc-600m-2024-12
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np


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
# Forge API embedding extraction (ESMC)
# ---------------------------------------------------------------------------

def extract_esmc_forge_embeddings(
    sequences: list[str],
    token: str,
    model_name: str = "esmc-600m-2024-12",
    cache_path: str | None = None,
    request_timeout: int = 120,
    save_interval: int = 100,
) -> np.ndarray:
    """Extract mean-pooled embeddings via the EvolutionaryScale Forge API (ESMC models).

    Args:
        sequences: List of amino acid sequences.
        token: Forge API token.
        model_name: Forge ESMC model ID (e.g. "esmc-600m-2024-12", "esmc-6b-2024-12").
        cache_path: Path to .npy file for incremental checkpointing.
        request_timeout: Per-request timeout in seconds.
        save_interval: Save checkpoint every N sequences.

    Returns:
        numpy array of shape (n_sequences, hidden_dim).
    """
    try:
        from esm.sdk.forge import ESMCForgeInferenceClient
        from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
    except ImportError:
        sys.exit(
            "ERROR: EvolutionaryScale ESM package not found.\n"
            "Install with:  pip install esm httpx\n"
        )

    import torch

    # Resume from checkpoint if exists
    start_idx = 0
    embeddings = []
    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path)
        start_idx = len(cached)
        embeddings = list(cached)
        print(f"Resuming from checkpoint: {start_idx}/{len(sequences)} done")
        if start_idx >= len(sequences):
            return np.stack(embeddings)

    print(f"Connecting to Forge API ({model_name})...")
    client = ESMCForgeInferenceClient(
        model=model_name,
        url="https://forge.evolutionaryscale.ai",
        token=token,
        request_timeout=request_timeout,
    )

    config = LogitsConfig(return_embeddings=True)

    t0 = time.time()
    for i in range(start_idx, len(sequences)):
        seq = sequences[i]

        for attempt in range(3):
            try:
                protein = ESMProtein(sequence=seq)
                tokens = client.encode(protein)
                if isinstance(tokens, ESMProteinError):
                    raise RuntimeError(f"encode error: {tokens.error_msg}")
                output = client.logits(tokens, config)
                if isinstance(output, ESMProteinError):
                    raise RuntimeError(f"logits error: {output.error_msg}")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\nFATAL: seq {i} failed after 3 attempts: {e}")
                    if cache_path and embeddings:
                        np.save(cache_path, np.stack(embeddings))
                        print(f"Checkpoint saved at {len(embeddings)} sequences")
                    raise
                wait = 2 ** attempt * 5
                print(f"\n  Retry {attempt+1}/3 for seq {i}: {e}. Waiting {wait}s...")
                time.sleep(wait)

        # Mean-pool over sequence positions, excluding BOS/EOS special tokens
        # output.embeddings shape: (1, seq_len+2, hidden_dim)
        emb = output.embeddings
        if isinstance(emb, torch.Tensor):
            emb = emb.float().cpu()
            emb = emb[0, 1:-1, :].mean(dim=0).numpy()
        else:
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim == 3:
                emb = emb[0, 1:-1, :].mean(axis=0)
            else:
                emb = emb.flatten()
        embeddings.append(emb)

        # Progress reporting
        done = i - start_idx + 1
        total_done = len(embeddings)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(sequences) - total_done
        eta = remaining / rate if rate > 0 else 0
        print(
            f"\r  {total_done}/{len(sequences)} seqs | "
            f"{rate:.2f} seq/s | ETA {eta/3600:.1f}h",
            end="",
            flush=True,
        )

        # Incremental checkpoint
        if cache_path and (len(embeddings) % save_interval == 0):
            np.save(cache_path, np.stack(embeddings))

    print()
    result = np.stack(embeddings)

    # Final checkpoint save
    if cache_path:
        np.save(cache_path, result)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta",   required=True,  help="Input FASTA file")
    parser.add_argument("--outfile", required=True,  help="Output .npz file")
    parser.add_argument("--token",   required=True,  help="Forge API token")
    parser.add_argument("--model",   default="esmc-600m-2024-12",
                        help="Forge ESMC model name (default: esmc-600m-2024-12)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-request timeout in seconds (default: 120)")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable checkpoint/resume")
    args = parser.parse_args()

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cache_path = None if args.no_checkpoint else str(outfile) + ".cache.npy"

    # ---- Load sequences ----
    print(f"Parsing FASTA: {args.fasta} ...")
    records = list(parse_fasta(args.fasta))
    accessions = [acc for acc, _ in records]
    sequences  = [seq for _, seq in records]
    print(f"  Sequences: {len(sequences):,}")

    # ---- Extract embeddings ----
    t0 = time.time()
    embeddings_arr = extract_esmc_forge_embeddings(
        sequences=sequences,
        token=args.token,
        model_name=args.model,
        cache_path=cache_path,
        request_timeout=args.timeout,
    )

    # ---- Save ----
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
    print(f"Total time: {(time.time() - t0) / 3600:.2f}h")

    # Clean up cache file on success
    if cache_path and Path(cache_path).exists():
        Path(cache_path).unlink()
        print(f"Cache file removed: {cache_path}")


if __name__ == "__main__":
    main()
