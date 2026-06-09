# Forge API setup

Four ESM checkpoints in this paper (ESMC-6B, ESM3-SMALL, ESM3-MEDIUM, ESM3-LARGE) are not openly released. They are accessed via EvolutionaryScale's [Forge API](https://forge.evolutionaryscale.ai/), which returns the same final-layer activations and logits as the open-weight checkpoints would.

## Obtaining a token

Register at [forge.evolutionaryscale.ai](https://forge.evolutionaryscale.ai/) and create an API key.


Every Forge-dependent job in this repo reads `FORGE_TOKEN` from the environment and fails loudly (with a pointer to this file) if it isn't set.

## Models accessed via Forge

| Model | Forge ID | Used for |
|---|---|---|
| ESMC-6B | `esmc-6b-2024-12` | Largest ESMC PCA / probe / PPL |
| ESM3-SMALL | `esm3-small-2024-08` | ESM3 family scaling (Fig 4) |
| ESM3-MEDIUM | `esm3-medium-2024-08` | ESM3 family scaling (Fig 4) |
| ESM3-LARGE | `esm3-large-2024-08` | Largest ESM3 PCA / probe / PPL |

## Embedding extraction via Forge

Forge returns final-layer hidden states identical to the open-weight forward pass:

```python
from esm.api import ESM3, ESMC, ESMProtein, LogitsConfig

client = ESM3(forge_token=FORGE_TOKEN, model_name="esm3-large-2024-08")
tokens = client.encode(ESMProtein(sequence=seq))
result = client.logits(tokens, LogitsConfig(return_embeddings=True))
emb = result.embeddings.mean(axis=1)   # mean over residues, drop BOS/EOS
```

The wrapper used in this repo is `projects/esm_viral_probe/scripts/extract_embeddings_forge.py`. It checkpoints to `<out>/.cache.npy`.
