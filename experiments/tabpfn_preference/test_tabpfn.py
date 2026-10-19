import numpy as np
import tabpfn_lib as x
from metric_reference import official

p=np.asarray([[.1,.2,.2,.2,.3],[.8,.1,.05,.03,.02],[.2,.2,.2,.2,.2]])
v=x.priority(p,.5); assert (v>0).tolist()==[True,False,True]
labels=np.asarray([5,1,2]); ids=np.asarray(["b","a","c"])
a=x.metric(labels,v,ids); b=official(labels,v,ids)
assert all(abs(a[k]-b[k])<1e-12 for k in b) and a["FP"]==0 and a["FN"]==0
chosen,rows=x.choose(p,labels,ids,np.asarray([0,0,1])); assert len(rows)==19 and chosen["threshold"] in x.THRESHOLDS
print("PASS: priority, independent metric, threshold grid.")
