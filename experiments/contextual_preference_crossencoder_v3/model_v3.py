"""V3: V2 architecture with LeakyReLU in the two prediction heads only."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "contextual_preference_crossencoder_v2"))
import model_v2 as v2

SEED = v2.SEED
SIDE_TOKENS = v2.SIDE_TOKENS
PAIR_MAX_LENGTH = v2.PAIR_MAX_LENGTH
QUERY_BATCH = v2.QUERY_BATCH
N_REFERENCE = v2.N_REFERENCE
EPOCHS = v2.EPOCHS
NEGATIVE_SLOPE = 0.01

prepare = v2.prepare
references = v2.references
pair_batch = v2.pair_batch
query_batch = v2.query_batch
query_loss = v2.query_loss
contextual_loss = v2.contextual_loss
adapter = v2.adapter
normalized_cls = v2.normalized_cls


class QueryHead(nn.Module):
    """Parameter layout deliberately matches V2; only activation differs."""
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Linear(768, 96)
        self.activation = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.drop = nn.Dropout(.1)
        self.out = nn.Linear(96, 5)

    def forward(self, cls, return_hidden=False):
        hidden = self.activation(self.in_layer(10 * cls))
        logits = self.out(self.drop(hidden))
        return (logits, hidden) if return_hidden else logits


class ContextHead(nn.Module):
    """Parameter layout deliberately matches V2; only activation differs."""
    def __init__(self):
        super().__init__()
        self.ref_label = nn.Embedding(5, 8)
        self.in_layer = nn.Linear(776, 96)
        self.activation = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
        self.drop = nn.Dropout(.1)
        self.out = nn.Linear(96, 5)
        self.compare = nn.Linear(776, 1)

    def forward(self, cls, ref_labels, return_hidden=False):
        features = torch.cat([10 * cls, self.ref_label(ref_labels - 1)], dim=1)
        hidden = self.activation(self.in_layer(features))
        pair_logits = self.out(self.drop(hidden))
        logits = pair_logits.reshape(-1, N_REFERENCE, 5).mean(1)
        compare = self.compare(features).reshape(-1, N_REFERENCE)
        return (logits, compare, hidden) if return_hidden else (logits, compare)


def head(arm):
    return QueryHead().to("cuda") if arm == "query_only" else ContextHead().to("cuda")


def forward(ctx, encoder, head_model, arm, queries, refs=None, monitor=False):
    if arm == "query_only":
        cls = normalized_cls(encoder(**query_batch(ctx, queries)))
        return head_model(cls, return_hidden=monitor)
    cls = normalized_cls(encoder(**pair_batch(ctx, queries, refs)))
    labels = torch.tensor(ctx["labels"][refs], device="cuda", dtype=torch.long)
    return head_model(cls, labels.reshape(-1), return_hidden=monitor)
