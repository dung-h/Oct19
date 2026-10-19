"""Normalized, monitored version of the contextual preference heads."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "contextual_preference_crossencoder"))
import model as v1

SEED = 20261030
SIDE_TOKENS = v1.SIDE_TOKENS
PAIR_MAX_LENGTH = v1.PAIR_MAX_LENGTH
QUERY_BATCH = v1.QUERY_BATCH
N_REFERENCE = v1.N_REFERENCE
EPOCHS = 2

prepare = v1.prepare
references = v1.references
pair_batch = v1.pair_batch
query_batch = v1.query_batch
query_loss = v1.query_loss
contextual_loss = v1.contextual_loss
adapter = v1._adapter


def normalized_cls(output):
    return F.normalize(output.last_hidden_state[:, 0].float(), dim=1)


class QueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Linear(768, 96)
        self.drop = nn.Dropout(.1)
        self.out = nn.Linear(96, 5)

    def forward(self, cls, return_hidden=False):
        hidden = F.relu(self.in_layer(10 * cls))
        logits = self.out(self.drop(hidden))
        return (logits, hidden) if return_hidden else logits


class ContextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ref_label = nn.Embedding(5, 8)
        self.in_layer = nn.Linear(776, 96)
        self.drop = nn.Dropout(.1)
        self.out = nn.Linear(96, 5)
        self.compare = nn.Linear(776, 1)

    def forward(self, cls, ref_labels, return_hidden=False):
        features = torch.cat([10 * cls, self.ref_label(ref_labels - 1)], dim=1)
        hidden = F.relu(self.in_layer(features))
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
