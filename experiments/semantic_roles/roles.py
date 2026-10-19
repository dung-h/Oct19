"""Frozen, label-independent evidence selection and full-size role encodings."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'read_gate_repair'))
import gate_lib as source

LLM='Qwen/Qwen2.5-7B-Instruct'
REV='a09a35458c702b33eeacc393d103063234e8bc28'
ENCODER='sentence-transformers/all-MiniLM-L6-v2'
ENC_REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
KINDS=['research','tool','evaluation','survey','tutorial','collection']
ROLES=['object','method','result']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(title,abstract):
    result=[dict(number=0,field='title',start=0,end=len(title),text=title)]
    bounds=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])',abstract)]+[len(abstract)]
    for a,b in zip(bounds[:-1],bounds[1:]):
        value=abstract[a:b];left=len(value)-len(value.lstrip());right=len(value.rstrip())
        if right>left:result.append(dict(number=len(result),field='abstract',start=a+left,end=a+right,text=abstract[a+left:a+right]))
    return result


def parse(value,sentences):
    try:
        match=re.search(r'\[.*\]|\{.*\}',value,flags=re.S)
        obj=json.loads(match.group(0))
        if isinstance(obj,list):
            if len(obj)!=4 or type(obj[3]) is not int or not 0<=obj[3]<len(KINDS):raise ValueError('schema')
            obj=dict(zip(ROLES,obj[:3]),kind=KINDS[obj[3]])
        if set(obj)!=set(ROLES+['kind']) or obj['kind'] not in KINDS:raise ValueError('schema')
        for role in ROLES:
            ix=obj[role]
            if ix is not None and (type(ix) is not int or ix<0 or ix>=len(sentences)):raise ValueError('index')
            if len(sentences)==1 and role!='object' and ix is not None:raise ValueError('missing abstract')
        return obj,False
    except (ValueError,TypeError,AttributeError):
        return dict(object=None,method=None,result=None,kind='research'),True


class OutputGrammar:
    """A small token NFA: only exact JSON keys and valid sentence indices.

    Tokenize fixed fragments and alternatives separately; the concatenated
    sequence still decodes to valid JSON. Epsilon links handle alternatives
    whose token sequences are prefixes of others (e.g. 1 and 10).
    """
    def __init__(self,tokenizer,n_sentences):
        self.edges={};self.epsilon={};self.next_state=1
        def node():
            value=self.next_state;self.next_state+=1;return value
        numbers=['null']+[str(k) for k in range(n_sentences)]
        non_title=['null']+[str(k) for k in range(1,n_sentences)]
        stages=[['['],numbers,[','],non_title,[','],non_title,[','],[str(k)+']' for k in range(len(KINDS))]]
        anchor=0
        for alternatives in stages:
            target=node()
            for text in alternatives:
                current=anchor
                for token in tokenizer.encode(text,add_special_tokens=False):
                    edges=self.edges.setdefault(current,{})
                    if token not in edges:edges[token]=node()
                    current=edges[token]
                self.epsilon.setdefault(current,set()).add(target)
            anchor=target
        self.edges[anchor]={tokenizer.eos_token_id:anchor}
        self.cache={():self.closure({0})}

    def closure(self,states):
        result=set(states);todo=list(states)
        while todo:
            for other in self.epsilon.get(todo.pop(),set()):
                if other not in result:result.add(other);todo.append(other)
        return result

    def allowed(self,prefix):
        key=tuple(prefix)
        if key not in self.cache:
            self.allowed(key[:-1])
            states={self.edges[s][key[-1]] for s in self.cache[key[:-1]] if key[-1] in self.edges.get(s,{})}
            self.cache[key]=self.closure(states)
        return sorted({token for s in self.cache[key] for token in self.edges.get(s,{})})


def llm_extract(ctx,preflight=False):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    OUT.mkdir(parents=True,exist_ok=True)
    data=ctx['data'];prompt=(HERE/'extraction_prompt.txt').read_text()
    protocol=dict(sources={p.name:digest(p) for p in [HERE/'roles.py',HERE/'fast_decode.py',HERE/'extraction_prompt.txt',HERE/'PLAN.md']},
        full_input_sha256=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        raw_zip_sha256=digest(ROOT.parent/'assignment-data-mining-261.zip'),
        model=LLM,revision=REV,seed=20261018,dtype='NF4 double-quantized, float16 compute',batch_size=2,
        max_input_tokens=2048,max_new_tokens=32,sample=False,output='token NFA constrained JSON array [object,method,result,kind_index]',
        decoder='greedy with bulk grammar-forced punctuation; must match standard-generate preflight')
    protocol['packages']={p:importlib.metadata.version(p) for p in ['torch','transformers','bitsandbytes','accelerate']}
    protocol_path=OUT/'extraction_protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Extraction source/config changed'
    else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    path=OUT/('preflight_extraction.jsonl' if preflight else 'extraction.jsonl')
    done={}
    if path.exists():
        for line in path.read_text().splitlines():
            rec=json.loads(line);done[rec['key']]=rec
    ix=list(range(len(data)))
    if preflight:ix=sorted(ix,key=lambda i:hashlib.sha256(f'{data.split.iloc[i]}:{data.id.iloc[i]}'.encode()).hexdigest())[:16]
    pending=[i for i in ix if f'{data.split.iloc[i]}:{data.id.iloc[i]}' not in done]
    if not pending:return
    torch.set_num_threads(4);torch.manual_seed(20261018)
    tokenizer=AutoTokenizer.from_pretrained(LLM,revision=REV,local_files_only=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(LLM,revision=REV,local_files_only=True,
        torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa').eval().requires_grad_(False)
    started=time.monotonic();batch_size=2
    for start in range(0,len(pending),batch_size):
        rows=pending[start:start+batch_size];texts=[];numbered=[]
        for i in rows:
            s=spans(data.title.iloc[i],data.abstract.iloc[i]);numbered.append(s)
            doc='\n'.join(f"{x['number']}: {x['text']}" for x in s)
            texts.append(tokenizer.apply_chat_template([dict(role='system',content=prompt),dict(role='user',content=doc)],tokenize=False,add_generation_prompt=True))
        lengths=[len(x) for x in tokenizer(texts,truncation=False)['input_ids']]
        batch=tokenizer(texts,padding=True,truncation=True,max_length=2048,return_tensors='pt').to('cuda')
        grammars=[OutputGrammar(tokenizer,len(s)) for s in numbered]
        from fast_decode import generate
        with torch.inference_mode():
            generated=generate(model,batch,grammars,max_new_tokens=32)
        decoded=tokenizer.batch_decode(generated,skip_special_tokens=True)
        with path.open('a') as file:
            for i,s,value,length in zip(rows,numbered,decoded,lengths):
                obj,invalid=parse(value,s)
                # A truncated prompt cannot substantiate a selected sentence index.
                if length>2048:obj=dict(object=None,method=None,result=None,kind='research');invalid=True
                rec=dict(key=f'{data.split.iloc[i]}:{data.id.iloc[i]}',raw=value,selection=obj,
                    invalid=invalid,input_tokens=length,truncated=length>2048,
                    input_sha256=hashlib.sha256(json.dumps(s,ensure_ascii=False).encode()).hexdigest())
                done[rec['key']]=rec;file.write(json.dumps(rec,ensure_ascii=False)+'\n')
        elapsed=time.monotonic()-started
        print('EXTRACT',len(done),'/',len(ix),'elapsed',round(elapsed,1),'seconds_per_doc',round(elapsed/(start+len(rows)),3),flush=True)
    if preflight:
        print(json.dumps(list(done.values())[:3],ensure_ascii=False,indent=2),flush=True)


def materialize(ctx):
    records=[json.loads(x) for x in (OUT/'extraction.jsonl').read_text().splitlines()]
    lookup={r['key']:r for r in records};assert len(lookup)==len(records)==3090
    rows=[];evidence=[]
    for row in ctx['data'].itertuples():
        key=f'{row.split}:{row.id}';rec=lookup[key];sent=spans(row.title,row.abstract)
        assert rec['input_sha256']==hashlib.sha256(json.dumps(sent,ensure_ascii=False).encode()).hexdigest()
        obj,invalid=parse(rec['raw'],sent)
        if rec['truncated']:obj=dict(object=None,method=None,result=None,kind='research');invalid=True
        assert obj==rec['selection'] and invalid==rec['invalid']
        item=dict(split=row.split,id=row.id,title=row.title,abstract=row.abstract,kind=obj['kind'],invalid=int(invalid))
        for role in ROLES:
            k=obj[role];item[role]='' if k is None else sent[k]['text']
            if k is not None:evidence.append(dict(split=row.split,id=row.id,role=role,**sent[k]))
        rows.append(item)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'texts.csv',index=False)
    pd.DataFrame(evidence).to_csv(OUT/'evidence.csv',index=False)
    sample=frame.assign(key=frame.split+':'+frame.id)
    sample['hash']=sample.key.map(lambda k:hashlib.sha256(k.encode()).hexdigest())
    sample.sort_values('hash').head(24).drop(columns='hash').to_csv(OUT/'extraction_audit_sample.csv',index=False)
    return frame


def encode(ctx,texts):
    target=OUT/'embeddings.npz'
    if target.exists():return
    import torch
    from transformers import AutoTokenizer,AutoModel
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4);torch.manual_seed(20261018)
    location=snapshot_download(ENCODER,revision=ENC_REV,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(location)
    model=AutoModel.from_pretrained(location).to('cuda').eval().requires_grad_(False)
    arrays={};stats=[]
    for field in ['title','abstract']+ROLES:
        values=texts[field].fillna('').astype(str).tolist();vec=np.zeros((len(values),384),np.float32)
        ix=np.array([i for i,v in enumerate(values) if v.strip()],int)
        lengths=np.array([len(t) for t in tokenizer([values[i] for i in ix],truncation=False)['input_ids']])
        ordered=ix[np.argsort(lengths,kind='stable')]
        for start in range(0,len(ordered),24):
            rows=ordered[start:start+24];batch=tokenizer([values[i] for i in rows],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                hidden=model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
            vec[rows]=pooled.cpu().numpy()
        arrays[field]=vec;stats.append(dict(field=field,available=len(ix),truncated=int((lengths>512).sum())))
        print('ENCODE',stats[-1],flush=True)
    arrays['keys']=(texts.split+':'+texts.id).to_numpy(str)
    np.savez_compressed(target,**arrays)
    (OUT/'embedding_manifest.json').write_text(json.dumps(dict(model=ENCODER,revision=ENC_REV,
        input_sha256=digest(OUT/'texts.csv'),extraction_sha256=digest(OUT/'extraction.jsonl'),stats=stats),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','extract','encode']);stage=parser.parse_args().stage
    ctx=source.context()
    if stage=='preflight':llm_extract(ctx,True)
    elif stage=='extract':llm_extract(ctx);materialize(ctx)
    else:encode(ctx,materialize(ctx))


if __name__=='__main__':main()
