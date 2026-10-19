"""Independent implementation of the assignment metric for verification."""
from __future__ import annotations
import numpy as np

def official(labels, priorities, ids):
    labels=np.asarray(labels,int); priorities=np.asarray(priorities,float); ids=np.asarray(ids,str)
    take=np.flatnonzero(priorities>0)
    take=np.asarray(sorted(take,key=lambda i:(-priorities[i],ids[i])),int)
    gain=(labels[take]-1)/4
    weights=1/np.log2(np.arange(len(take))+2)
    q=float((weights*gain).sum()/weights.sum()) if len(take) else 0.
    r=float((labels[take]>1).sum()/(labels>1).sum()) if (labels>1).any() else 0.
    return {"H":2*q*r/(q+r) if q+r else 0.,"Q":q,"R":r}
