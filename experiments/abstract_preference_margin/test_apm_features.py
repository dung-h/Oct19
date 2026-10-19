#!/usr/bin/env python3
"""Hand-calculated feature edge cases and lineage-independent invariants."""
import numpy as np

from apm_features import feature_names, features_for_queries


def toy_context():
    vectors = np.asarray([
        [1.0, 0.0],       # query
        [1.0, 0.0],       # N, same venue
        [0.8, 0.6],       # N, other venue
        [0.9, np.sqrt(.19)],  # P, same venue
        [0.7, np.sqrt(.51)],  # P, same venue
        [0.85, np.sqrt(1-.85**2)], # G, same venue
        [0.65, np.sqrt(1-.65**2)], # G, other venue
        [1.0, 0.0],       # duplicate of query group; must be excluded
    ], dtype=np.float32)
    return dict(
        spaces={name: vectors.copy() for name in
                ["retrieval_margin", "title_margin", "abstract_margin"]},
        ids=np.asarray(["q", "n1", "n2", "p1", "p2", "g1", "g2", "dup"], dtype=str),
        groups=np.asarray([7, 1, 2, 3, 4, 5, 6, 7]),
        venues=np.asarray(["v", "v", "w", "v", "v", "v", "w", "v"], dtype=str),
        abstract_available=np.ones(8, dtype=bool),
        labels=np.asarray([5, 1, 1, 2, 3, 4, 5, 1], dtype=np.int8),
    )


def main():
    names = feature_names()
    assert len(names) == len(set(names)) == 43
    col = {name: ix for ix, name in enumerate(names)}
    ctx = toy_context()
    values, support, evidence = features_for_queries(
        ctx, np.asarray([0]), np.arange(1, 8), "abstract_margin", keep_evidence=True)
    row = values[0]
    assert np.isclose(row[col["all_N_max"]], 1.0)
    assert np.isclose(row[col["all_N_mean5"]], .9)
    assert np.isclose(row[col["all_N_support15"]], 2/15)
    assert np.isclose(row[col["all_P_max"]], .9)
    assert np.isclose(row[col["all_P_mean5"]], (.9+.85+.7+.65)/4)
    assert np.isclose(row[col["all_G_mean5"]], (.85+.65)/2)
    assert np.isclose(row[col["all_PminusN_max"]], -.1)
    assert np.isclose(row[col["all_GminusN_mean5"]], (.75-.9))
    assert np.isclose(row[col["venue_N_support15"]], 1/15)
    assert np.isclose(row[col["venue_P_support15"]], 3/15)
    assert np.isclose(row[col["query_abstract_available"]], 1.0)
    assert not any(item["reference_id"] == "dup" for item in evidence)
    assert all(item["excluded_same_publication_group"] == 1 for item in support)

    # Missing class margins are defined as zero and present flags identify them.
    ctx["labels"][3:] = 1
    values, _, _ = features_for_queries(ctx, [0], [1, 2, 3], "title_margin")
    row = values[0]
    assert row[col["all_P_present"]] == 0
    assert row[col["all_G_present"]] == 0
    assert row[col["all_PminusN_max"]] == 0

    # A missing query abstract receives no neighborhood features.
    ctx["abstract_available"][0] = False
    values, support, evidence = features_for_queries(ctx, [0], np.arange(1, 8), "abstract_margin",
                                                     keep_evidence=True)
    assert np.all(values == 0)
    assert not support and not evidence
    print("PASS: 43 features, hand-calculated top-k/margins, group exclusion, empty classes, missing query.")


if __name__ == "__main__":
    main()
