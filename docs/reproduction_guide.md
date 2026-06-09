# Reproduction guide

There are two ways to reproduce this paper's results, in increasing order of cost.

1. **Render the figures** from the committed summary data: CPU only, ~1 minute.
2. **Regenerate the summary data** from the raw FASTAs: GPUs + an
   EvolutionaryScale Forge API token, several GPU-days.


## 1. Render the figures (CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./render_figures.sh
```

## 2. Regenerate the summary data (GPU + Forge)

### Prerequisites

1. **Conda + Python 3.11**:
   ```bash
   conda env create -f environment.yml -n beyond-nativeness
   conda activate beyond-nativeness
   ```
2. **A CUDA GPU.** ESM2 (8M–650M), ESMC (300M, 600M), and ESM3-open run locally;
   ESM2-3B/15B want a larger-memory GPU (we used H200-class cards, bf16).
3. **Forge API token** for the closed checkpoints (ESM3-SMALL/MEDIUM/LARGE,
   ESMC-6B) — see [`forge_api_setup.md`](forge_api_setup.md):
   ```bash
   export FORGE_TOKEN="<your token>"
   ```
4. **HuggingFace login** for the gated ESM3-open checkpoint:
   ```bash
   huggingface-cli login
   ```
5. **Repo root**:
   ```bash
   export BEYOND_NATIVENESS_ROOT="$PWD"
   ```
6. **MMseqs2** (provided by the conda env; verify with `which mmseqs`).


## Caveats

- **NCBI Virus FASTA** for `human_virus_dataset` comes from a manual UI export.