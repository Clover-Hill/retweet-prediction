"""
Fuse tweet features into a single 2 048-D embedding
--------------------------------------------------

• Text   block (1 024 D): layer-normalised copy of the original 1 024-D text embedding.
• Scalar block (1 024 D): 7 numeric features projected with a *frozen* random matrix.

The whole vector is finally L2-normalised so text and scalar parts have comparable
magnitudes in cosine space.
"""

import numpy as np
from datasets import load_from_disk, DatasetDict

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DATA_DIR = "/home/jqcao/projects/retweet-prediction/data/preprocessed_dataset"
OUTPUT_DIR = "/home/jqcao/projects/retweet-prediction/data/feature_dataset"
SCALAR_FEATS = [
    "has_url",
    "has_hashtags",
    "user_verified",
    "text_length",
    "user_followers_count",
    "low_time_interval",
    "high_hour_interval",
]
PROJ_DIM = 1_024               # size of the scalar projection
RNG_SEED = 42                  # change for a different (but fixed) projection

# --------------------------------------------------------------------
# Build a frozen random projection matrix  R ∈ ℝ^(7 × 1 024)
# Entries ~ N(0, 1/√d) so each projected dimension has unit variance.
# --------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
R = rng.normal(0, 1 / np.sqrt(PROJ_DIM), size=(len(SCALAR_FEATS), PROJ_DIM)).astype(
    np.float32
)  # (7, 1024)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-sample layer normalisation (mean-0, var-1 along last dim)."""
    mu = x.mean(-1, keepdims=True)
    sigma = x.std(-1, keepdims=True)
    return (x - mu) / (sigma + eps)


def build_feature_embedding(batch):
    """
    Args:
        batch: dict of columns, each key → list or np.ndarray with shape (B, …)

    Returns:
        batch with new key 'feature_embedding' (shape: B × 2 048)
    """
    # -------- 1.  Text block -------------------------------------------------
    text = np.array(batch["text_embedding"], dtype=np.float32)          # (B, 1024)
    text = layer_norm(text)                                             # layer-norm

    # -------- 2.  Scalar features  ------------------------------------------
    scalars = []
    for key in SCALAR_FEATS:
        col = np.array(batch[key])
        if key in {"text_length", "user_followers_count"}:              # heavy-tailed
            col = np.log1p(col)
        scalars.append(col.astype(np.float32))
    scalars = np.stack(scalars, axis=-1)                                # (B, 7)

    # -------- 3.  Random projection to 1 024 D ------------------------------
    scalar_vec = scalars @ R                                            # (B, 1024)

    # -------- 4.  Concatenate & L2-normalise -------------------------------
    feat = np.concatenate([text, scalar_vec], axis=-1)                  # (B, 2048)
    norm = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
    feat /= norm

    batch["feature_embedding"] = feat
    return batch


def main():
    # ----------------------------------------------------------------
    # Load dataset (Dataset or DatasetDict) and rename the column
    # ----------------------------------------------------------------
    ds = load_from_disk(DATA_DIR)

    if isinstance(ds, DatasetDict):
        for split in ds.keys():
            ds[split] = ds[split].rename_column("embedding", "text_embedding")
            ds[split] = ds[split].map(
                build_feature_embedding, 
                batched=True, 
                num_proc=32,
                load_from_cache_file=False,
                )
    else:  # a single Dataset
        ds = ds.rename_column("embedding", "text_embedding")
        ds = ds.map(build_feature_embedding, batched=True)

    # ----------------------------------------------------------------
    # Save back to the same directory (over-write)
    # ----------------------------------------------------------------
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Done. Dataset with 'feature_embedding' saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()