"""Cumulative targets and label-independent least-squares consistency repair."""
import numpy as np

THRESHOLDS=np.array([2,3,4,5])


def targets(labels):
    labels=np.asarray(labels)
    assert np.isfinite(labels).all() and np.isin(labels,[1,2,3,4,5]).all()
    return (labels[:,None]>=THRESHOLDS[None,:]).astype(int)


def project(raw):
    """Equal-weight PAV onto q2 >= q3 >= q4 >= q5 for each individual row."""
    raw=np.asarray(raw,dtype=float)
    assert raw.ndim==2 and raw.shape[1]==4
    assert np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
    result=np.empty_like(raw)
    for i,row in enumerate(raw):
        means=[];counts=[]
        for value in row:
            means.append(float(value));counts.append(1)
            while len(means)>1 and means[-2]<means[-1]:
                total=counts[-2]+counts[-1]
                pooled=(means[-2]*counts[-2]+means[-1]*counts[-1])/total
                means[-2:]=[pooled];counts[-2:]=[total]
        result[i]=np.repeat(means,counts)
    assert (np.diff(result,axis=1)<=1e-14).all()
    np.testing.assert_allclose(result.sum(1),raw.sum(1),atol=1e-14,rtol=1e-14)
    return result


def distribution(tails):
    tails=np.asarray(tails,dtype=float)
    assert tails.ndim==2 and tails.shape[1]==4
    assert np.isfinite(tails).all() and (tails>=0).all() and (tails<=1).all()
    assert (np.diff(tails,axis=1)<=0).all()
    p=np.column_stack([1-tails[:,0],tails[:,:-1]-tails[:,1:],tails[:,-1]])
    np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
    assert (p>=0).all()
    return p


def ordinal_view(raw):
    q=project(raw)
    return dict(probability=distribution(q),read=q[:,0],gain=q.mean(1),raw=raw,tails=q)


def multiclass_view(p):
    p=np.asarray(p)
    q=1-np.cumsum(p[:,:4],axis=1)
    # Keep incumbent arithmetic exactly, including its tie behavior.
    return dict(probability=p,read=1-p[:,0],gain=p@(np.arange(5)/4),raw=q,tails=q)


def priorities(view,threshold):
    return np.where(view['read']>threshold,10*view['gain'],0.)
