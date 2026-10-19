"""Deterministic difficult-pair sampling and verbatim discourse evidence."""
import re
import numpy as np
import pandas as pd

CUES = {
 'problem': r'\b(?:we (?:study|investigate|consider|address|examine|analy[sz]e)|this (?:paper|work|article) (?:studies|investigates|addresses|considers)|(?:the|our) (?:goal|aim|objective) (?:is|of)|we focus on|we are concerned with)\b',
 'method': r'\b(?:we (?:propose|present|introduce|develop|extend|apply|implement)|this (?:paper|work|article) (?:proposes|presents|introduces|develops)|our (?:approach|method|framework|algorithm))\b',
 'result': r'\b(?:we(?:\s+\w+){0,2} (?:prove|show|establish|demonstrate|find)|(?:our |the )?(?:results|experiments|evaluation) (?:show|demonstrate|indicate)|it is shown)\b',
}


def sentences(text):
    bounds=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])',text)]+[len(text)]
    result=[]
    for start,end in zip(bounds[:-1],bounds[1:]):
        raw=text[start:end];left=len(raw)-len(raw.lstrip());right=len(raw.rstrip())
        if right>left:
            result.append((start+left,start+right,text[start+left:start+right]))
    return result


def claim_spans(text):
    result=[]
    for aspect,pattern in CUES.items():
        count=0
        for start,end,sentence in sentences(text):
            if len(sentence.split())>=5 and re.search(pattern,sentence,re.I):
                result.append(dict(aspect=aspect,start=start,end=end,evidence=sentence))
                count+=1
                if count==2:break
    return result


def select_pairs(data,labels,groups,retrieval):
    result=[]
    for venue in sorted(data.venue.iloc[:2494].unique()):
        a=np.flatnonzero((data.venue.to_numpy()==venue)&(labels==1))
        b=np.flatnonzero((data.venue.to_numpy()==venue)&(labels>=4))
        sims=retrieval[a]@retrieval[b].T
        ia,ib=np.meshgrid(np.arange(len(a)),np.arange(len(b)),indexing='ij')
        aa,bb=a[ia.ravel()],b[ib.ravel()];ss=sims.ravel()
        ordered=np.lexsort((data.id.iloc[bb].to_numpy(),data.id.iloc[aa].to_numpy(),-ss))
        used=set();count=0
        for j in ordered:
            left,right=int(aa[j]),int(bb[j])
            if groups[left]==groups[right] or left in used or right in used:continue
            used.update([left,right]);count+=1
            result.append(dict(pair_id=f'P{len(result)+1:02}',venue=venue,a_index=left,b_index=right,
                               retrieval_cosine=float(ss[j]),selection='sample'))
            if count==4:break
        assert count==4
    assert len(result)==20
    return result


def export_audit(ctx,out):
    data,_,labels,groups,_,retrieval,content,_=ctx
    pairs=select_pairs(data,labels,groups,retrieval)
    lookup=dict(zip(data.id.iloc[:2494],range(2494)))
    for a,b in [('217','355'),('1045','1184'),('1000','1001'),('2383','1634')]:
        ia,ib=lookup[a],lookup[b]
        pairs.append(dict(pair_id=f'K{a}_{b}',venue=data.venue.iloc[ia],a_index=ia,b_index=ib,
                          retrieval_cosine=float(retrieval[ia]@retrieval[ib]),selection='previously_discussed'))
    flag_cols=[c for c in content if not c.endswith(('_words','_unknown','_available'))]
    exports=[];spans=[];readme=['# Difficult training pairs: content review packet','',
        'Twenty sampled pairs, four per venue, label1 versus label4/5. Selected by retrieval similarity, not model errors. Known examples are separate. No preference labels are inferred or changed.','']
    for pair in pairs:
        a,b=pair['a_index'],pair['b_index'];record=dict(pair)
        different=[c for c in flag_cols if content.loc[a,c]!=content.loc[b,c]]
        record.update(category_flag_differences=len(different),different_flags=';'.join(different))
        readme.extend([f"## {pair['pair_id']} | {pair['venue']} | {pair['selection']} | similarity {pair['retrieval_cosine']:.4f}",'',
                       'Different category flags: '+('; '.join(different) if different else 'NONE'), ''])
        for role,index in [('a',a),('b',b)]:
            row=data.iloc[index]
            for field in ['id','title','abstract','doi']:
                record[role+'_'+field]=row.get(field,'')
            record[role+'_label']=int(labels[index])
            readme.extend([f"### {role.upper()}: ID {row.id}, original Label {int(labels[index])}",'',row.title,'',row.abstract or '[No usable abstract]', ''])
            for span in claim_spans(row.abstract):
                spans.append(dict(pair_id=pair['pair_id'],role=role,id=row.id,field='abstract',**span))
        exports.append(record)
    pd.DataFrame(exports).to_csv(out/'content_pairs.csv',index=False)
    pd.DataFrame(spans).to_csv(out/'claim_spans.csv',index=False)
    (out/'PAIR_PACKET.md').write_text('\n'.join(readme)+'\n')
    return pd.DataFrame(exports)
