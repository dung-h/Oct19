"""Matched SPECTER2 frozen/adapted heads; labels never enter tokenizer input."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = 'allenai/specter2_base'
BASE_REV = '3447645e1def9117997203454fa4495937bfbd83'
ADAPTER = 'allenai/specter2_regression'
ADAPTER_REV = 'a9d105def7a081c8f7c5a88ab4145646901c6003'
SEED = 20261021
CHECKPOINTS = [1, 2, 4, 8]
THRESHOLDS = np.arange(0, .901, .05)
BATCH = 32
MAX_LENGTH = 128


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    base = snapshot_download(BASE, revision=BASE_REV, local_files_only=True)
    adapter = snapshot_download(ADAPTER, revision=ADAPTER_REV, local_files_only=True)
    return base, adapter, AutoTokenizer.from_pretrained(base)


def encoder(adapted):
    from adapters import AutoAdapterModel
    base, adapter, _ = resources()
    torch.manual_seed(SEED)
    model = AutoAdapterModel.from_pretrained(base)
    model.load_adapter(adapter, load_as='preference', set_active=True)
    if adapted:
        model.train_adapter('preference')
    else:
        model.requires_grad_(False)
    model.to('cuda').eval()
    enabled = [name for name, p in model.named_parameters() if p.requires_grad]
    assert (len(enabled) > 0) == adapted
    assert all('.adapters.preference.' in name for name in enabled)
    return model


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 5))

    def forward(self, x):
        return self.net(10 * x)


def loss(logits, labels):
    read_logit = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
    return F.cross_entropy(logits, labels - 1) + .5 * F.binary_cross_entropy_with_logits(read_logit, (labels > 1).float())


def priority(probabilities, cutoff):
    gain = probabilities @ (np.arange(5) / 4)
    return np.where(1 - probabilities[:, 0] > cutoff, 10 * gain, 0.)


def batch_tokens(ctx, indices):
    tokens = ctx['tokens']
    batch = ctx['tokenizer'].pad([{k: tokens[k][i] for k in tokens} for i in indices], return_tensors='pt')
    return {k: v.to('cuda') for k, v in batch.items()}


def embed(model, ctx, indices):
    if model is None:
        return torch.from_numpy(ctx['vectors'][indices]).to('cuda')
    hidden = model(**batch_tokens(ctx, indices)).last_hidden_state[:, 0].float()
    return F.normalize(hidden, dim=1)


def predict(model, head, ctx, indices):
    result = np.zeros((len(indices), 5), dtype=np.float32)
    order = np.argsort([len(ctx['tokens']['input_ids'][i]) for i in indices], kind='stable')
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH):
            offsets = order[start:start+BATCH]
            with torch.autocast('cuda', dtype=torch.float16):
                logits = head(embed(model, ctx, indices[offsets])).float()
            result[offsets] = torch.softmax(logits, dim=1).cpu().numpy()
    assert np.isfinite(result).all() and np.allclose(result.sum(axis=1), 1, atol=1e-6)
    return result


def prepare(ctx, out):
    _, _, tokenizer = resources()
    texts = (ctx['data'].title + tokenizer.sep_token).tolist()
    raw = tokenizer(texts, truncation=False, return_token_type_ids=False)
    tokens = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, return_token_type_ids=False)
    ctx.update(tokenizer=tokenizer, tokens=tokens)
    keys = (ctx['data'].split + ':' + ctx['data'].id).to_numpy(dtype=str)
    path = out / 'frozen_title_embeddings.npz'
    if path.exists():
        z = np.load(path)
        assert np.array_equal(z['keys'], keys)
        ctx['vectors'] = z['vectors']
    else:
        model = encoder(False)
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        order = np.argsort([len(x) for x in tokens['input_ids']], kind='stable')
        with torch.inference_mode():
            for start in range(0, len(order), BATCH):
                ix = order[start:start+BATCH]
                with torch.autocast('cuda', dtype=torch.float16):
                    vectors[ix] = embed(model, ctx, ix).cpu().numpy()
        np.savez_compressed(path, keys=keys, vectors=vectors)
        ctx['vectors'] = vectors
        del model
        torch.cuda.empty_cache()
    stats = dict(max_length=MAX_LENGTH, n=len(texts),
                 truncated=int(sum(len(x) > MAX_LENGTH for x in raw['input_ids'])),
                 max_tokens=max(map(len, raw['input_ids'])), mean_tokens=float(np.mean(list(map(len, raw['input_ids'])))),
                 token_sha256=hashlib.sha256(json.dumps(tokens.data, sort_keys=True).encode()).hexdigest())
    (out/'input_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    return ctx


def fit(ctx, fit_indices, pred_indices, arm, epochs, prefix, save_dir=None, include_train=False):
    path = Path(str(prefix)+'.npz')
    meta_path = Path(str(prefix)+'.json')
    if path.exists() and (save_dir is None or (save_dir/'manifest.json').exists()):
        z = np.load(path)
        assert np.array_equal(z['fit'], fit_indices) and np.array_equal(z['pred'], pred_indices)
        assert np.array_equal(z['epochs'], epochs)
        return {e:z[f'epoch{e}'] for e in epochs}, json.loads(meta_path.read_text()), z['train'] if include_train else None
    started = time.monotonic()
    assert len(set(fit_indices) & set(pred_indices)) == 0
    assert set(fit_indices) <= set(range(2494))
    assert set(ctx['groups'][fit_indices]).isdisjoint(ctx['groups'][pred_indices]) or save_dir is not None
    model = encoder(True) if arm == 'adapted' else None
    torch.manual_seed(SEED)
    head = Head().to('cuda')
    initial_head_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()
    adapter_params = [] if model is None else [p for p in model.parameters() if p.requires_grad]
    initial_adapter = [p.detach().clone() for p in adapter_params]
    parameters = [{'params': list(head.parameters()), 'lr': .001}]
    if adapter_params:
        parameters.append({'params': adapter_params, 'lr': .00005})
    optimizer = torch.optim.AdamW(parameters, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    rng = np.random.default_rng(SEED)
    predictions = {}
    trace = []
    last_train = None
    nonzero_gradient_steps = 0
    overflow_steps = 0
    for epoch in range(1, max(epochs)+1):
        head.train()
        losses = []
        permutation = rng.permutation(fit_indices)
        for start in range(0, len(permutation), BATCH):
            ix = permutation[start:start+BATCH]
            target = torch.tensor(ctx['y'][ix], dtype=torch.long, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                value = loss(head(embed(model, ctx, ix)).float(), target)
            assert torch.isfinite(value), 'Nonfinite loss'
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 1.)
            finite_gradient = bool(torch.isfinite(norm))
            if not finite_gradient:
                overflow_steps += 1
            if adapter_params and finite_gradient:
                assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in adapter_params)
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                nonzero_gradient_steps += 1
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(value.detach()))
        trace.append(dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.monotonic()-started))
        if epoch in epochs:
            predictions[epoch] = predict(model, head, ctx, pred_indices)
        print(Path(prefix).name, 'epoch', epoch, 'loss', round(trace[-1]['loss'], 4),
              'seconds', round(trace[-1]['seconds'], 1), flush=True)
    if include_train:
        last_train = predict(model, head, ctx, fit_indices)
    drift = float(sum((p.detach()-old).square().sum().item() for p, old in zip(adapter_params, initial_adapter)) ** .5)
    assert (drift > 0) == (arm == 'adapted')
    meta = dict(arm=arm, seed=SEED, epochs=epochs, trace=trace, seconds=time.monotonic()-started,
                trained_encoder_parameters=sum(p.numel() for p in adapter_params), adapter_l2_drift=drift,
                nonzero_adapter_gradient_steps=nonzero_gradient_steps, initial_head_sha256=initial_head_hash,
                skipped_overflow_steps=overflow_steps,
                fit=fit_indices.tolist(), pred=pred_indices.tolist())
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(head.cpu().state_dict(), save_dir/'head.pt')
        if model is not None:
            model.save_adapter(str(save_dir/'adapter'), 'preference')
        (save_dir/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    values = {f'epoch{e}': p for e, p in predictions.items()}
    if include_train:
        values['train'] = last_train
    np.savez_compressed(path, fit=fit_indices, pred=pred_indices, epochs=epochs, **values)
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    del model, head, optimizer, initial_adapter, adapter_params
    torch.cuda.empty_cache()
    return predictions, meta, last_train


def reload_predict(ctx, indices, directory):
    info = json.loads((directory/'manifest.json').read_text())
    model = None
    if info['arm'] == 'adapted':
        model = encoder(False)
        model.delete_adapter('preference')
        model.load_adapter(str(directory/'adapter'), load_as='preference', set_active=True)
        model.to('cuda').eval()
    head = Head()
    head.load_state_dict(torch.load(directory/'head.pt', map_location='cpu', weights_only=True))
    head.to('cuda').eval()
    values = predict(model, head, ctx, indices)
    del model, head
    torch.cuda.empty_cache()
    return values
