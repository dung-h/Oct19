"""Group-safe contextual SPECTER2 pair model. No query label enters model input."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "specter2_preference"))
import modeling as old

SEED = 20261029
SIDE_TOKENS = 126
PAIR_MAX_LENGTH = 2 * SIDE_TOKENS + 3
QUERY_BATCH = 4
N_REFERENCE = 6
EPOCHS = 2
THRESHOLDS = np.arange(0.0, .901, .05)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def paper_text(ctx, index):
    row = ctx["data"].iloc[int(index)]
    title = str(row.title or "").strip()
    abstract = str(row.abstract or "").strip()
    return (title + " " + abstract).strip() or title or "[no text]"


def prepare(ctx):
    """Pre-tokenize each side separately: deterministic 126+126 allocation."""
    _, _, tokenizer = old.resources()
    texts = [paper_text(ctx, i) for i in range(len(ctx["data"]))]
    raw = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    sides = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=SIDE_TOKENS)["input_ids"]
    ctx.update(tokenizer=tokenizer, raw_token_lengths=np.asarray([len(x) for x in raw]), side_ids=sides)
    return dict(n=len(texts), side_tokens=SIDE_TOKENS, pair_max_length=PAIR_MAX_LENGTH,
                max_raw_tokens=int(max(map(len, raw))), mean_raw_tokens=float(np.mean(list(map(len, raw)))),
                papers_side_truncated=int(sum(len(x) > SIDE_TOKENS for x in raw)),
                text_sha256=sha(texts), side_token_sha256=sha(sides))


def _nearest(ctx, query, candidates, count):
    if not len(candidates):
        raise ValueError(f"No eligible reference for query {ctx['ids'][query]}")
    candidates = np.asarray(candidates, dtype=int)
    venue = ctx["venues"][query]
    same = candidates[ctx["venues"][candidates] == venue]
    pool = same if len(same) >= count else candidates
    space = ctx["spaces"]["retrieval_margin"]
    scores = space[pool] @ space[query]
    order = sorted(range(len(pool)), key=lambda k: (-float(scores[k]), str(ctx["ids"][pool[k]])))
    selected = pool[np.asarray(order[:count], dtype=int)].tolist()
    repeated = 0
    while len(selected) < count:
        selected.append(selected[-1]); repeated += 1
    return selected, bool(len(same) >= count), repeated


def references(ctx, fit, query_indices):
    """Return (n_query, 6) refs: 3 label1, then 3 label>1, all from current fit."""
    fit = np.asarray(fit, dtype=int)
    labels = ctx["labels"]
    result = np.empty((len(query_indices), N_REFERENCE), dtype=int)
    audit = []
    for row, q in enumerate(np.asarray(query_indices, dtype=int)):
        allowed = fit[ctx["groups"][fit] != ctx["groups"][q]]
        neg = allowed[labels[allowed] == 1]
        pos = allowed[labels[allowed] > 1]
        nref, nvenue, nrepeat = _nearest(ctx, q, neg, 3)
        pref, pvenue, prepeat = _nearest(ctx, q, pos, 3)
        result[row] = nref + pref
        audit.append(dict(query=int(q), id=str(ctx["ids"][q]), negative_same_venue=nvenue,
                          positive_same_venue=pvenue, negative_repeated=nrepeat,
                          positive_repeated=prepeat, references=result[row].tolist()))
    groups = ctx["groups"]
    assert all(groups[q] != groups[r] for q, refs in zip(query_indices, result) for r in refs)
    assert np.all(labels[result[:, :3]] == 1) and np.all(labels[result[:, 3:]] > 1)
    return result, audit


def pair_batch(ctx, queries, ref_matrix):
    """Build padded B*6 token pairs from independently truncated token IDs."""
    records = []
    for query, refs in zip(queries, ref_matrix):
        qids = ctx["side_ids"][int(query)]
        for ref in refs:
            records.append(ctx["tokenizer"].prepare_for_model(qids, ctx["side_ids"][int(ref)],
                                                               add_special_tokens=True, truncation=False,
                                                               return_token_type_ids=False))
    batch = ctx["tokenizer"].pad(records, padding=True, return_tensors="pt")
    assert int(batch["input_ids"].shape[1]) <= PAIR_MAX_LENGTH
    return {key: value.to("cuda") for key, value in batch.items()}


def query_batch(ctx, queries):
    records = [ctx["tokenizer"].prepare_for_model(ctx["side_ids"][int(q)], add_special_tokens=True,
                                                    truncation=False, return_token_type_ids=False)
               for q in queries]
    batch = ctx["tokenizer"].pad(records, padding=True, return_tensors="pt")
    return {key: value.to("cuda") for key, value in batch.items()}


class ContextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ref_label = nn.Embedding(5, 8)
        self.pair = nn.Sequential(nn.Linear(776, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 5))
        self.compare = nn.Linear(776, 1)

    def forward(self, cls, ref_labels):
        z = torch.cat([10 * cls, self.ref_label(ref_labels - 1)], dim=1)
        pair_logits = self.pair(z)
        return pair_logits.reshape(-1, N_REFERENCE, 5).mean(1), self.compare(z).reshape(-1, N_REFERENCE)


class QueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 5))

    def forward(self, cls):
        return self.net(10 * cls)


def read_logit(logits):
    return torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]


def contextual_loss(logits, compare, labels, ref_labels):
    main = F.cross_entropy(logits, labels - 1) + .5 * F.binary_cross_entropy_with_logits(read_logit(logits), (labels > 1).float())
    target = (labels[:, None] > ref_labels).float()
    mask = labels[:, None] != ref_labels
    ranking = F.binary_cross_entropy_with_logits(compare[mask], target[mask]) if mask.any() else compare.sum() * 0
    return main + .25 * ranking


def query_loss(logits, labels):
    return F.cross_entropy(logits, labels - 1) + .5 * F.binary_cross_entropy_with_logits(read_logit(logits), (labels > 1).float())


def _adapter():
    return old.encoder(True)


def preflight(ctx, fit, queries):
    """Single real optimization step; training rows only and no score calculation."""
    refs, audit = references(ctx, fit, queries)
    model = _adapter()
    head = ContextHead().to("cuda")
    params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{ "params": list(head.parameters()), "lr": .001},
                             { "params": [p for p in model.parameters() if p.requires_grad], "lr": .00005}], weight_decay=.01)
    labels = torch.tensor(ctx["labels"][queries], device="cuda", dtype=torch.long)
    ref_labels = torch.tensor(ctx["labels"][refs], device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        cls = model(**pair_batch(ctx, queries, refs)).last_hidden_state[:, 0].float()
        logits, compare = head(cls, ref_labels.reshape(-1))
        value = contextual_loss(logits, compare, labels, ref_labels)
    assert torch.isfinite(value)
    value.backward()
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in adapter_params)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
    lengths = [len(ctx["side_ids"][int(q)]) + len(ctx["side_ids"][int(r)]) + 3 for q, rr in zip(queries, refs) for r in rr]
    result = dict(status="PASSED", queries=[int(x) for x in queries], fit_size=int(len(fit)), loss=float(value.detach()),
                  elapsed_seconds=time.monotonic()-started, peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  trainable_adapter_parameters=int(sum(p.numel() for p in adapter_params)),
                  pairs=int(len(queries) * N_REFERENCE), pair_tokens=dict(min=min(lengths), mean=float(np.mean(lengths)), max=max(lengths), cap=PAIR_MAX_LENGTH),
                  references=audit, same_group_violations=0)
    del model, head, opt; torch.cuda.empty_cache()
    return result
