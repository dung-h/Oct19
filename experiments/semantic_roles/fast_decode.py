"""Greedy constrained decoding that batches consecutive grammar-forced tokens.

Only branches require model logits. Forced punctuation is appended without an
argmax; the next model call processes the whole appended chunk with KV cache.
The selection rule is identical to greedy decoding with prefix token masks.
"""
import torch


def generate(model,batch,grammars,max_new_tokens=32):
    device=batch['input_ids'].device
    count=batch['input_ids'].shape[0]
    original_length=batch['input_ids'].shape[1]
    full=batch['input_ids'];mask=batch['attention_mask']
    prefixes=[[] for _ in range(count)];finished=[False]*count
    eos=model.config.eos_token_id
    if isinstance(eos,list):eos=eos[0]
    pending=[];past=None

    def append(tokens):
        pending.append(tokens)
        for i,token in enumerate(tokens):
            prefixes[i].append(token)
            if token==eos:finished[i]=True

    def allowed():
        return [[eos] if finished[i] else grammars[i].allowed(prefixes[i]) for i in range(count)]

    def forced():
        while len(prefixes[0])<max_new_tokens and not all(finished):
            choices=allowed()
            if not all(len(x)==1 for x in choices):break
            append([x[0] for x in choices])

    forced()
    while len(prefixes[0])<max_new_tokens and not all(finished):
        if pending:
            extra=torch.tensor(pending,dtype=torch.long,device=device).T
            full=torch.cat([full,extra],dim=1)
            mask=torch.cat([mask,torch.ones_like(extra)],dim=1)
            pending.clear()
        prepared=model.prepare_inputs_for_generation(full,attention_mask=mask,past_key_values=past,use_cache=True)
        outputs=model(**prepared,return_dict=True)
        logits=outputs.logits[:,-1,:].float()
        past=outputs.past_key_values
        del outputs
        choices=allowed();tokens=[]
        for i,options in enumerate(choices):
            assert options,'No legal continuation'
            if len(options)==1:tokens.append(options[0])
            else:
                index=int(torch.argmax(logits[i,options]).item())
                tokens.append(options[index])
        del logits
        append(tokens);forced()
    return torch.tensor(prefixes,dtype=torch.long,device=device)
