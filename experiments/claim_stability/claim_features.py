"""Evidence-preserving text extraction, conservative flag repair and frozen vectors."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

SEED=20261018
ASPECTS=['problem','method','result']
CUES={
 'problem': r'\b(?:we (?:study|investigate|consider|address|examine|analy[sz]e|focus on)|this (?:paper|work|article) (?:studies|investigates|addresses|considers|examines)|(?:the|our) (?:goal|aim|objective) (?:is|of)|we are concerned with)\b',
 'method': r'\b(?:we(?:\s+(?:also|further|first|then|formally))? (?:propose|present|introduce|develop|extend|apply|implement|define|construct|design|formulate|provide|give|build)|this (?:paper|work|article) (?:proposes|presents|introduces|develops|defines|constructs)|our (?:approach|method|framework|algorithm)|(?:main|first|second) (?:approach|method|construction))\b',
 'result': r'\b(?:we(?:\s+(?:also|further|first|then|formally|fully))? (?:prove|show|establish|demonstrate|find|obtain|characteri[sz]e)|(?:our |the )?(?:results|experiments|evaluation) (?:show|demonstrate|indicate)|(?:main|first|second) result|it is shown)\b',
}
EDUCATION=r'\b(?:educat\w*|teach\w*|schools?|curricul\w*|classrooms?|pedagog\w*)\b'
TUTORIAL=r'\btutorials?\b'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentences(text):
    bounds=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])',text)]+[len(text)]
    out=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        raw=text[a:b];left=len(raw)-len(raw.lstrip());right=len(raw.rstrip())
        if right>left:out.append((a+left,a+right,text[a+left:a+right]))
    return out


def extract(data):
    records=[];spans=[]
    for row in data.itertuples():
        candidates=sentences(row.abstract)
        record=dict(split=row.split,id=row.id)
        for role,pattern in CUES.items():
            selected=[s for s in candidates if len(s[2].split())>=5 and re.search(pattern,s[2],re.I)][:2]
            record['claims_'+role]=' '.join(s[2] for s in selected)
            for start,end,evidence in selected:
                spans.append(dict(split=row.split,id=row.id,kind='claims',role=role,
                                  field='abstract',start=start,end=end,evidence=evidence))
        for block,role in enumerate(ASPECTS):
            selected=candidates[2*block:2*block+2]
            record['prefix_'+role]=' '.join(s[2] for s in selected)
            for start,end,evidence in selected:
                spans.append(dict(split=row.split,id=row.id,kind='prefix',role=role,
                                  field='abstract',start=start,end=end,evidence=evidence))
        records.append(record)
    return pd.DataFrame(records),pd.DataFrame(spans)


def repair(data,content):
    fixed=content.copy();ledger=[]
    for field in ['title','abstract']:
        education=data[field].str.contains(EDUCATION,regex=True,case=False).astype(float)
        tutorial=data[field].str.contains(TUTORIAL,regex=True,case=False).astype(float)
        column=field+'_domain_education'
        changed=np.flatnonzero(content[column].to_numpy()!=education.to_numpy())
        for i in changed:
            ledger.append(dict(split=data.split.iloc[i],id=data.id.iloc[i],feature=column,
                before=content[column].iloc[i],after=education.iloc[i],reason='require educational wording; standalone student is insufficient'))
        fixed[column]=education
        old_domain=[c for c in fixed if c.startswith(field+'_domain_') and not c.endswith('_unknown')]
        fixed[field+'_domain_unknown']=fixed[old_domain].sum(axis=1).eq(0).astype(float)
        fixed[field+'_type_tutorial']=tutorial
        old_type=[c for c in fixed if c.startswith(field+'_type_') and not c.endswith('_unknown')]
        fixed[field+'_type_unknown']=fixed[old_type].sum(axis=1).eq(0).astype(float)
        for i in np.flatnonzero(tutorial.to_numpy()):
            ledger.append(dict(split=data.split.iloc[i],id=data.id.iloc[i],feature=field+'_type_tutorial',
                               before=0.,after=1.,reason='literal tutorial evidence'))
    return fixed,pd.DataFrame(ledger)


def encode(texts,out,model_id,revision):
    target=out/'sentence_embeddings.npz'
    input_digest=hashlib.sha256(texts.to_csv(index=False).encode()).hexdigest()
    if target.exists():
        meta=json.loads((out/'embedding_manifest.json').read_text())
        assert meta['input_sha256']==input_digest and meta['revision']==revision
        return
    import torch
    from transformers import AutoTokenizer,AutoModel
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4);torch.manual_seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    location=snapshot_download(model_id,revision=revision,local_files_only=True,
        allow_patterns=['*.json','vocab.txt','pytorch_model.bin','model.safetensors'])
    tokenizer=AutoTokenizer.from_pretrained(location)
    model=AutoModel.from_pretrained(location).to(device).eval().requires_grad_(False)
    arrays={};statistics=[]
    for kind in ['prefix','claims']:
        blocks=[]
        for role in ASPECTS:
            values=texts[kind+'_'+role].tolist()
            vectors=np.zeros((len(texts),model.config.hidden_size),dtype=np.float32)
            ix=np.flatnonzero(texts[kind+'_'+role].str.len().gt(0))
            lengths=np.array([len(v) for v in tokenizer([values[i] for i in ix],truncation=False)['input_ids']])
            order=ix[np.argsort(lengths,kind='stable')]
            for start in range(0,len(order),32):
                query=order[start:start+32]
                batch=tokenizer([values[i] for i in query],padding=True,truncation=True,max_length=128,
                                return_tensors='pt',return_token_type_ids=False)
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.inference_mode(),torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):
                    hidden=model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1).float()
                    pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                vectors[query]=normalize(pooled.cpu().numpy())
            blocks.append(vectors)
            statistics.append(dict(kind=kind,role=role,available=len(ix),truncated=int((lengths>128).sum()),
                mean_tokens=float(lengths.mean()) if len(lengths) else 0.))
            print('Encoded',kind,role,'available',len(ix),flush=True)
        arrays[kind]=np.stack(blocks,axis=1)
    arrays['keys']=(texts.split+':'+texts.id).to_numpy(dtype=str)
    np.savez_compressed(target,**arrays)
    (out/'embedding_manifest.json').write_text(json.dumps(dict(model=model_id,revision=revision,
        input_sha256=input_digest,pooling='attention-mask mean then L2; max128; frozen',
        device=device,statistics=statistics),indent=2)+'\n')


def project(vectors,fit,pred):
    train=[];query=[];models=[]
    for block in range(3):
        model=TruncatedSVD(n_components=16,random_state=SEED)
        a=model.fit_transform(vectors[fit,block].astype(float))
        b=model.transform(vectors[pred,block].astype(float))
        present_fit=np.linalg.norm(vectors[fit,block],axis=1)>0
        present_pred=np.linalg.norm(vectors[pred,block],axis=1)>0
        # TruncatedSVD is uncentered: unknown-zero inputs stay zero.
        train.append(np.column_stack([a,present_fit.astype(float)]))
        query.append(np.column_stack([b,present_pred.astype(float)]));models.append(model)
    names=[f'text_block{b}_{i:02}' for b in range(3) for i in range(17)]
    return pd.DataFrame(np.hstack(train),columns=names),pd.DataFrame(np.hstack(query),columns=names),models


def project_saved(vectors,indices,models):
    arrays=[]
    for block,model in enumerate(models):
        value=model.transform(vectors[indices,block].astype(float))
        presence=np.linalg.norm(vectors[indices,block],axis=1)>0
        arrays.append(np.column_stack([value,presence.astype(float)]))
    names=[f'text_block{b}_{i:02}' for b in range(3) for i in range(17)]
    return pd.DataFrame(np.hstack(arrays),columns=names)
