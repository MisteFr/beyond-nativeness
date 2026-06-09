#!/usr/bin/env python3
"""
run_masked_reconstruction_single_fasta.py
==========================================
Run masked reconstruction on a single FASTA file with a custom label.
Supports all ESM3/ESMC backends (local GPU and Forge API).

Usage (local ESM3 Open):
    python scripts/run_masked_reconstruction_single_fasta.py \
        --fasta /path/to/random_uniform.faa \
        --label random --out_dir results/random_uniform \
        --backend esm3_open --cache_dir /path/to/hf_cache

Usage (local ESMC 600M):
    python scripts/run_masked_reconstruction_single_fasta.py \
        --fasta /path/to/random_uniform.faa \
        --label random --out_dir results/esmc_600m/random_uniform \
        --backend esmc_local --model esmc_600m --cache_dir /path/to/hf_cache

Usage (Forge API):
    export FORGE_TOKEN=<token>
    python scripts/run_masked_reconstruction_single_fasta.py \
        --fasta /path/to/random_uniform.faa \
        --label random --out_dir results/esm3_medium/random_uniform \
        --backend forge --model_name esm3-medium-2024-08 --client_type esm3
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
    sys.exit("ERROR: EvolutionaryScale ESM package not found.\nInstall with:  pip install esm httpx")

try:
    from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN as MASK_TOKEN_ID
except ImportError:
    MASK_TOKEN_ID = 32


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
# Resume helpers (for API backends)
# ---------------------------------------------------------------------------
def load_completed(tsv_path: Path) -> set[str]:
    completed = set()
    if not tsv_path.exists():
        return completed
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            completed.add(row["accession"])
    return completed


# ---------------------------------------------------------------------------
# Core: batched local inference (ESM3 Open / ESMC)
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
# Core: per-sequence API inference (Forge)
# ---------------------------------------------------------------------------
def process_sequence_api(seq, tokenizer, client, mask_rate, n_seeds, max_len):
    from esm.sdk.api import ESMProteinTensor, ESMProteinError, LogitsConfig

    seq_trunc = seq[:max_len]
    encoded = tokenizer(
        [seq_trunc], return_tensors="pt", padding=False,
        truncation=True, max_length=max_len + 2,
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
                    raise RuntimeError(f"API error {output.error_code}: {output.error_msg}")
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
# Forge client factory
# ---------------------------------------------------------------------------
def make_client(client_type, model_name, token, timeout):
    if client_type == "esm3":
        from esm.sdk.forge import ESM3ForgeInferenceClient
        return ESM3ForgeInferenceClient(
            model=model_name, url="https://forge.evolutionaryscale.ai",
            token=token, request_timeout=timeout,
        )
    elif client_type == "esmc":
        from esm.sdk.forge import ESMCForgeInferenceClient
        return ESMCForgeInferenceClient(
            model=model_name, url="https://forge.evolutionaryscale.ai",
            token=token, request_timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown client_type: {client_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta",       required=True, help="Input FASTA file")
    parser.add_argument("--label",       default="random", help="Label for all sequences")
    parser.add_argument("--split",       default="all", help="Split value in TSV")
    parser.add_argument("--out_dir",     required=True, help="Output directory")
    parser.add_argument("--backend",     required=True,
                        choices=["esm3_open", "esmc_local", "forge"],
                        help="Model backend")
    parser.add_argument("--model",       default="esmc_600m",
                        help="ESMC model name for esmc_local backend")
    parser.add_argument("--model_name",  default=None,
                        help="Forge model name (e.g. esm3-medium-2024-08)")
    parser.add_argument("--client_type", default=None, choices=["esm3", "esmc"],
                        help="Forge client type")
    parser.add_argument("--cache_dir",   default=None, help="HF model cache dir")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--mask_rate",   type=float, default=0.15)
    parser.add_argument("--n_seeds",     type=int, default=3)
    parser.add_argument("--max_len",     type=int, default=1022)
    parser.add_argument("--timeout",     type=int, default=120, help="API timeout (s)")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "per_sequence_results.tsv"

    print("=" * 60)
    print(f"Masked Reconstruction — Single FASTA")
    print(f"Backend:  {args.backend}")
    print(f"FASTA:    {args.fasta}")
    print(f"Label:    {args.label}")
    print(f"Mask:     {args.mask_rate*100:.0f}%  Seeds: {args.n_seeds}")
    print(f"Output:   {out_dir}")
    print("=" * 60)

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    # ---- Load tokenizer ----
    print("\nLoading EsmSequenceTokenizer...")
    tokenizer = EsmSequenceTokenizer()

    # ---- Load model / client ----
    if args.backend == "esm3_open":
        from esm.models.esm3 import ESM3
        device = torch.device(args.device)
        print("Loading ESM3-open (~1.4B params) ...")
        model = ESM3.from_pretrained("esm3-open").float().eval().to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    elif args.backend == "esmc_local":
        from esm.models.esmc import ESMC
        device = torch.device(args.device)
        print(f"Loading ESMC ({args.model}) ...")
        model = ESMC.from_pretrained(args.model, device=device).eval()
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    elif args.backend == "forge":
        if not args.model_name or not args.client_type:
            sys.exit("ERROR: --model_name and --client_type required for forge backend")
        forge_token = os.environ.get("FORGE_TOKEN", "")
        if not forge_token:
            sys.exit("ERROR: FORGE_TOKEN environment variable not set.")
        print(f"Connecting to Forge API ({args.model_name})...")
        client = make_client(args.client_type, args.model_name, forge_token, args.timeout)

    # ---- Sanity check ----
    print("  Sanity check...")
    if args.backend in ("esm3_open", "esmc_local"):
        with torch.no_grad():
            _tok = tokenizer(["MKTAYIAKQR"], return_tensors="pt",
                             padding=True, truncation=True, max_length=15)
            _out = model(sequence_tokens=_tok["input_ids"].to(device))
        assert hasattr(_out, "sequence_logits"), "Missing sequence_logits"
        print(f"  sequence_logits shape: {list(_out.sequence_logits.shape)}  OK")
        del _tok, _out
    else:
        from esm.sdk.api import ESMProteinTensor, ESMProteinError, LogitsConfig
        _enc = tokenizer(["MKTAYIAKQR"], return_tensors="pt", padding=False,
                         truncation=True, max_length=14)
        _pt = ESMProteinTensor(sequence=_enc["input_ids"][0])
        _out = client.logits(_pt, LogitsConfig(sequence=True))
        if isinstance(_out, ESMProteinError):
            sys.exit(f"ERROR: API sanity check failed: {_out.error_code} {_out.error_msg}")
        print(f"  logits OK")

    # ---- Load FASTA ----
    print(f"\nLoading {args.fasta} ...")
    records = parse_fasta(args.fasta)
    n_total = len(records)
    print(f"  {n_total:,} sequences")

    # ---- TSV header ----
    header = (
        ["accession", "label", "split", "length",
         "mean_perplexity", "mean_recovery_rate"]
        + [f"seed{s}_ppl"      for s in range(args.n_seeds)]
        + [f"seed{s}_recovery" for s in range(args.n_seeds)]
    )

    # ---- Run inference ----
    t0 = time.time()

    if args.backend in ("esm3_open", "esmc_local"):
        # Sort by length for efficient batching
        order = sorted(range(n_total), key=lambda i: len(records[i][1]))
        records_sorted = [records[i] for i in order]

        result_rows = []
        n_batches = (n_total + args.batch_size - 1) // args.batch_size

        for b_start in range(0, n_total, args.batch_size):
            b_records = records_sorted[b_start : b_start + args.batch_size]
            b_seqs = [r[1] for r in b_records]

            try:
                batch_results = process_batch(
                    b_seqs, model, tokenizer, device,
                    args.mask_rate, args.n_seeds, args.max_len,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    print(f"\n  [OOM] batch at {b_start}: retrying batch_size=1 on CPU ...")
                    model.cpu()
                    device_cpu = torch.device("cpu")
                    batch_results = []
                    for seq in b_seqs:
                        r = process_batch(
                            [seq], model, tokenizer, device_cpu,
                            args.mask_rate, args.n_seeds, args.max_len,
                        )
                        batch_results.extend(r)
                    model.to(device)
                else:
                    raise

            for (acc, seq), res in zip(b_records, batch_results):
                row = {
                    "accession": acc, "label": args.label, "split": args.split,
                    "length": len(seq),
                    "mean_perplexity": res["mean_perplexity"],
                    "mean_recovery_rate": res["mean_recovery_rate"],
                }
                for s in range(args.n_seeds):
                    row[f"seed{s}_ppl"] = res["seed_ppls"][s]
                    row[f"seed{s}_recovery"] = res["seed_recoveries"][s]
                result_rows.append(row)

            bi = b_start // args.batch_size + 1
            if bi % 50 == 0 or bi == n_batches:
                elapsed = time.time() - t0
                print(f"  batch {bi}/{n_batches}  "
                      f"elapsed {elapsed/60:.1f}m  "
                      f"last_ppl={batch_results[-1]['mean_perplexity']:.2f}",
                      flush=True)

        # Write TSV
        with open(tsv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
            writer.writeheader()
            for row in result_rows:
                writer.writerow(row)

    else:
        # API backend — per-sequence, resumable
        completed = load_completed(tsv_path)
        print(f"Already completed: {len(completed):,}")

        write_header = not tsv_path.exists()
        tsv_fh = open(tsv_path, "a", buffering=1)
        if write_header:
            tsv_fh.write("\t".join(header) + "\n")

        n_done = len(completed)
        for idx, (acc, seq) in enumerate(records):
            if acc in completed:
                continue

            res = process_sequence_api(seq, tokenizer, client,
                                       args.mask_rate, args.n_seeds, args.max_len)

            vals = [acc, args.label, args.split, str(len(seq)),
                    f"{res['mean_perplexity']:.6f}",
                    f"{res['mean_recovery_rate']:.6f}"]
            for s in range(args.n_seeds):
                vals.append(f"{res['seed_ppls'][s]:.6f}")
                vals.append(f"{res['seed_recoveries'][s]:.6f}")
            tsv_fh.write("\t".join(vals) + "\n")

            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate / 60 if rate > 0 else 0
                print(f"  {n_done}/{n_total}  "
                      f"elapsed {elapsed/60:.1f}m  "
                      f"ETA {eta:.0f}m  "
                      f"ppl={res['mean_perplexity']:.2f}", flush=True)

        tsv_fh.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # ---- Print summary stats ----
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"\n=== Summary ({args.label}, n={len(df)}) ===")
    print(f"  Perplexity:    {df['mean_perplexity'].mean():.3f} "
          f"± {df['mean_perplexity'].std():.3f}")
    print(f"  Recovery rate: {df['mean_recovery_rate'].mean():.4f} "
          f"± {df['mean_recovery_rate'].std():.4f}")
    print(f"  Output: {tsv_path}")


if __name__ == "__main__":
    main()
