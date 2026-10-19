#!/usr/bin/env python3
import numpy as np
import ndg_lib as n

p = {name: np.tile(np.array([[.2, .2, .2, .2, .2]]), (3, 1)) for name in n.SOURCE_NAMES}
x = n.meta_features(p)
assert x.shape == (3, 9) and np.isfinite(x).all()
labels = np.array([1, 5, 2]); ids = np.array(["b", "a", "c"])
priority = n.priority(np.array([.4, .8, .6]), p["incumbent"], .5)
m = n.metrics(labels, priority, ids)
assert m["TP"] == 2 and m["FP"] == 0 and m["FN"] == 0 and m["accepted"] == 2
choice, rows = n.choose_threshold(np.array([.4, .8, .6]), p["incumbent"], labels, ids, np.array([0, 0, 1]))
assert choice["threshold"] in set(n.THRESHOLDS) and len(rows) == 5
print("PASS: meta features, strict gate, per-fold threshold scoring.")
