#!/usr/bin/env python3
"""Small invariants for author histories and strict gate decisions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import author_features as af
import gate_lib as g


def test_parser():
    tokens, uncertain = af.parse_authors(' Alice Smith; Béla Németh and et al. ')
    assert tokens == ['alice smith', 'béla németh'] and uncertain
    assert af.parse_authors(None) == ([], False)


def test_group_exclusion_and_unknown_prior():
    # Query 2 shares group 0 with row 0, so row 0 must be removed from both
    # author support and the global prior; only row 1 (negative) remains.
    tokens=[['alice'], ['bob'], ['alice'], []]
    labels=np.array([5,1,1,4])
    groups=np.array([0,1,0,2])
    f=af.history_features(tokens,np.zeros(4,bool),np.zeros(4,bool),labels,groups,np.array([0,1]),np.array([2,3]))
    assert f.loc[0,'known_author_count']==0 and abs(f.loc[0,'author_read_mean']) < 1e-12
    assert f.loc[1,'author_missing']==1 and abs(f.loc[1,'author_read_mean'] - .5) < 1e-12


def test_metrics_and_boundary():
    assert not (np.array([.5]) > .5)[0]
    m=g.selection_metrics(np.array([1,2,5]),np.array([True,False,True]))
    assert m['TP']==1 and m['FP']==1 and m['FN']==1 and m['TN']==0 and m['total_error']==2


if __name__ == '__main__':
    test_parser();test_group_exclusion_and_unknown_prior();test_metrics_and_boundary()
    print('PASS 3 tests')
