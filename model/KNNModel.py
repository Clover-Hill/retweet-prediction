"""
K-NN‐based retweet predictor
----------------------------
Predicts the retweet count of each tweet sample by inspecting its nearest
neighbours in an already-built FAISS index.

Rule:
• Take the k NNs of the query’s `feature_embedding`.
• Keep only neighbours with similarity > sim_threshold.
• If the share of viral neighbours ≥ viral_threshold:
      prediction ← similarity-weighted mean(retweet_count of viral neighbours)
  else:
      prediction ← 0
Returns a dict: {"pred_retweet_count": Tensor[B]}
"""

from typing import Dict

import torch
import datasets
import numpy as np

# import your existing searcher
from knn.searcher import KNNSearcher   # make sure this file is on PYTHONPATH

class KNNModel(torch.nn.Module):
    def __init__(
        self,
        index_name: str = "feature_all",          # which FAISS index to use
        dim: int = 2_048,                         # dimension of feature_embedding
        k: int = 10,                              # number of neighbours
        use_gpu: bool = False,                    # search on GPU?
        dataset_dir: str = "/home/jqcao/projects/retweet-prediction/data/feature_dataset",
        sim_threshold: float = 0.7,              # IP similarity cut-off
        viral_threshold: float = 0.20,            # ratio that triggers viral weighting
    ):
        super().__init__()
        self.sim_threshold = sim_threshold
        self.viral_threshold = viral_threshold

        # ---------------------------  KNN searcher  --------------------------
        self.searcher = KNNSearcher(
            index_name=index_name,
            dim=dim,
            k=k,
            use_gpu=use_gpu,
            dataset_dir=dataset_dir,
        )
        self.device = self.searcher.device        # keep search/device consistent

        # ----------------------  neighbour meta-data  ------------------------
        # We cache the per-row meta data as tensors for O(1) gather.
        ds = datasets.load_from_disk(dataset_dir)
        train_ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds

        self.register_buffer(
            "_if_viral", torch.tensor(train_ds["if_viral"], dtype=torch.bool)
        )
        self.register_buffer(
            "_retweet_cnt",
            torch.tensor(train_ds["retweet_count"], dtype=torch.float32),
        )

    # --------------------------------------------------------------------- #
    #  forward                                                              #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, feature_embedding):
        """
        Args
        ----
        batch : dict coming from your DataLoader.
                Must contain key **'feature_embedding'**
                shape (B, dim), float32, L2-normalised.

        Returns
        -------
        {"pred_retweet_count": Tensor[B] (same device as input)}
        """
        feats = feature_embedding.to(torch.float32).to(self.device)

        # 1. retrieve k NNs (similarities because IndexFlatIP)
        sims, knns = self.searcher.get_knns(feats)          # (B,k) each

        B, k = sims.shape
        preds = torch.zeros(B, dtype=torch.float32)         # default 0

        # 2. loop over batch (vectorisation is possible but clarity first)
        for i in range(B):
            sim_i = sims[i]              # (k,)
            idx_i = knns[i]              # (k,)

            # keep neighbours above similarity threshold
            keep = sim_i > self.sim_threshold
            if not keep.any():
                continue                 # no valid neighbour → pred = 0

            sim_i = sim_i[keep]
            idx_i = idx_i[keep]

            viral_flags = self._if_viral[idx_i]
            n_viral = viral_flags.sum().item()
            ratio_viral = n_viral / len(idx_i)
            
            if ratio_viral >= self.viral_threshold and n_viral > 0:
                viral_idx = idx_i[viral_flags]
                viral_sims = sim_i[viral_flags]

                counts = self._retweet_cnt[viral_idx]

                # weighted average of viral tweets
                weights = torch.softmax(viral_sims, dim=0)
                preds[i] = torch.dot(weights, counts)
            # else leave preds[i] at 0

        preds = torch.zeros(B, dtype=torch.float32)         # default 0
        return {"pred_retweet_count": preds.to(feats.device)}