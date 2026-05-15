"""
CBraMod Foundation Model Provider
===================================
Implementation for the CBraMod EEG foundation model.
"""

from typing import Optional, Type

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .base import BaseFoundationModel, BaseTransformerModule, EmbeddingInfo
from .cbramod_src.cbramod import CBraMod as RawCBraMod


class CBraModModule(BaseTransformerModule):
    """
    CBraMod wrapper module.
    """

    def __init__(
        self,
        model_name: str = "braindecode/cbramod-pretrained",
        output_dim: int = 2,
        train_mode: str = "frozen",
        pooling: str = "mean",
        token: Optional[str] = None,
        patch_size: int = 200,
        seq_len: int = 4,
        **kwargs,
    ):
        super().__init__()

        self.pooling = pooling
        self.train_mode = train_mode
        self.patch_size = patch_size
        self.seq_len = seq_len

        # 1. Instantiate raw CBraMod
        self.backbone = RawCBraMod(
            in_dim=self.patch_size,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=self.seq_len,
            n_layer=12,
            nhead=8,
        )

        # 2. Download and load weights from HuggingFace Hub
        weights_path = hf_hub_download(
            repo_id=model_name, filename="pytorch_model.bin", token=token
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=False)

        # 3. Handle Parameter Freezing (Linear Probing)
        if self.train_mode == "frozen":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 4. Task-Specific Head
        self.backbone.proj_out = nn.Identity()
        feat_dim = 200  # CBraMod's d_model dimension
        self.head = nn.Linear(feat_dim, output_dim)

    def forward(
        self, X: torch.Tensor, return_embeddings: bool = False, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for CBraMod.

        Parameters
        ----------
        X : torch.Tensor
            Input EEG data of shape (batch, channels, time).
        return_embeddings : bool
            If True, returns the pooled representation instead of logits.
        """
        bz, ch_num, time = X.shape

        # Reshape to (batch, channels, patch_num, patch_size)
        patch_num = time // self.patch_size
        X_reshaped = X.contiguous().view(bz, ch_num, patch_num, self.patch_size)

        # Transform backbone pass
        # hidden shape: (batch, channels, patch_num, d_model)
        hidden = self.backbone(X_reshaped)

        # Spatial and Temporal Pooling
        if self.pooling == "mean":
            # Pool over channels (dim 1) and patches (dim 2)
            pooled = hidden.mean(dim=(1, 2))
        else:
            # First token pooling (mean over channels, first patch)
            pooled = hidden.mean(dim=1)[:, 0, :]

        if return_embeddings:
            return pooled

        return self.head(pooled)


class CBraModModel(BaseFoundationModel):
    """
    Unified CBraMod wrapper for the coco-pipe decoding engine.
    """

    def __init__(self, **kwargs):
        model_name = kwargs.pop("model_name", "braindecode/cbramod-pretrained")
        super().__init__(model_name=model_name, **kwargs)
        self.provider = "cbramod"

    def get_module_cls(self) -> Type[nn.Module]:
        """Return the CBraModModule class."""
        return CBraModModule

    def _get_net_params(self) -> dict:
        """Add CBraMod-specific parameters for skorch initialization."""
        params = super()._get_net_params()
        params.update(
            {
                "module__pooling": self.kwargs.get("pooling", "mean"),
                "module__token": self.kwargs.get("token"),
                "module__patch_size": self.kwargs.get("patch_size", 200),
                "module__seq_len": self.kwargs.get("seq_len", 4),
            }
        )
        return params

    def get_embedding_info(self) -> EmbeddingInfo:
        """Return metadata about CBraMod embeddings."""
        return EmbeddingInfo(
            n_embeddings=200,
            embedding_name=f"CBraMod ({self.kwargs.get('pooling', 'mean')})",
            provider="cbramod",
            model_name=self.model_name,
            sfreq=self.kwargs.get("sfreq", 200.0),
        )
