"""Cross-fitted class-conditional cosine features for abstract preference."""
from __future__ import annotations

import numpy as np

from apm_context import NEW_ARMS

CLASSES = {
    "N": np.asarray([1], dtype=np.int8),
    "P": np.asarray([2, 3, 4, 5], dtype=np.int8),
    "G": np.asarray([4, 5], dtype=np.int8),
}
SCOPES = ["all", "venue"]
STATS = ["max", "mean5", "mean15", "support15", "present"]
MARGINS = ["max", "mean5", "mean15"]


def feature_names():
    names = []
    for scope in SCOPES:
        for group in ["N", "P", "G"]:
            names.extend(f"{scope}_{group}_{stat}" for stat in STATS)
        for pair in ["PminusN", "GminusN"]:
            names.extend(f"{scope}_{pair}_{stat}" for stat in MARGINS)
    names.append("query_abstract_available")
    assert len(names) == 43 and len(set(names)) == 43
    return names


def _stable_top(values, ref_ids):
    # Similarity descending, then string ID ascending.
    return np.lexsort((ref_ids.astype(str), -values))


def _class_stats(values, ref_ids):
    order = _stable_top(values, ref_ids)
    n = len(order)
    if n == 0:
        return dict(max=0.0, mean5=0.0, mean15=0.0, support15=0.0, present=0.0), order
    ranked = values[order]
    return dict(
        max=float(ranked[0]),
        mean5=float(ranked[:min(5, n)].mean()),
        mean15=float(ranked[:min(15, n)].mean()),
        support15=float(min(n, 15) / 15.0),
        present=1.0,
    ), order


def features_for_queries(ctx, queries, refs, arm, keep_evidence=False):
    """Build features from reference labels only; never reads query labels."""
    if arm not in NEW_ARMS:
        raise ValueError(f"unknown preference-neighborhood arm: {arm}")
    queries = np.asarray(queries, dtype=int)
    refs = np.asarray(refs, dtype=int)
    if np.any(queries >= len(ctx["ids"])) or np.any(refs >= len(ctx["labels"])):
        raise AssertionError("query/reference indices are outside their source partitions")
    if len(np.unique(refs)) != len(refs):
        raise AssertionError("duplicate reference indices")

    x = ctx["spaces"][arm]
    matrix = x[queries] @ x[refs].T
    output = np.zeros((len(queries), 43), dtype=np.float32)
    names = feature_names()
    name_to_col = {name: j for j, name in enumerate(names)}
    supports, evidence = [], []
    ids = ctx["ids"]
    groups = ctx["groups"]
    yref = ctx["labels"][refs]
    valid_ref = ctx["abstract_available"][refs] & (np.linalg.norm(x[refs], axis=1) > 0)

    for row_pos, q in enumerate(queries):
        # Matching abstract coverage across arms. Missing abstracts use incumbent
        # probabilities at prediction time and contribute no new-neighbor values.
        if not ctx["abstract_available"][q] or np.linalg.norm(x[q]) == 0:
            continue
        same_group = groups[refs] == groups[q]
        eligible = valid_ref & ~same_group
        allowed = refs[eligible]
        sim = matrix[row_pos, eligible]
        label = yref[eligible]
        id_values = ids[allowed]
        same_venue = ctx["venues"][allowed] == ctx["venues"][q]

        for scope in SCOPES:
            in_scope = np.ones(len(allowed), dtype=bool) if scope == "all" else same_venue
            scope_sim = sim[in_scope]
            scope_labels = label[in_scope]
            scope_ids = id_values[in_scope]
            group_results = {}
            group_counts = {}
            for group_name, accepted_labels in CLASSES.items():
                take = np.isin(scope_labels, accepted_labels)
                values = scope_sim[take]
                reference_ids = scope_ids[take]
                stats, order = _class_stats(values, reference_ids)
                group_results[group_name] = stats
                group_counts[group_name] = len(values)
                for stat, value in stats.items():
                    output[row_pos, name_to_col[f"{scope}_{group_name}_{stat}"]] = value
                if keep_evidence and len(order):
                    for rank, ix in enumerate(order[:5], start=1):
                        evidence.append(dict(
                            query_id=str(ids[q]), query_index=int(q), arm=arm, scope=scope,
                            reference_id=str(reference_ids[ix]), reference_index=int(allowed[in_scope][take][ix]),
                            reference_label=int(scope_labels[take][ix]), group=group_name,
                            rank=rank, cosine=float(values[ix]),
                        ))

            for left, right in [("P", "N"), ("G", "N")]:
                can_compare = group_counts[left] > 0 and group_counts[right] > 0
                for stat in MARGINS:
                    value = group_results[left][stat] - group_results[right][stat] if can_compare else 0.0
                    output[row_pos, name_to_col[f"{scope}_{left}minus{right}_{stat}"]] = value

            supports.append(dict(
                query_id=str(ids[q]), query_index=int(q), arm=arm, scope=scope,
                references_total=int(len(refs)),
                excluded_same_publication_group=int(same_group.sum()),
                references_abstract_eligible=int(valid_ref.sum()),
                references_allowed=int(len(allowed)),
                references_in_scope=int(in_scope.sum()),
                references_N=int(group_counts["N"]),
                references_P=int(group_counts["P"]),
                references_G=int(group_counts["G"]),
            ))
        output[row_pos, name_to_col["query_abstract_available"]] = 1.0

    assert np.isfinite(output).all()
    return output, supports, evidence


def build_partition_features(ctx, name, fit, pred, source_pack, protocol_sha256):
    """Construct OOF training rows and three matched-reference prediction views."""
    import json
    import joblib
    from pathlib import Path

    from apm_context import OUT

    fit, pred = np.asarray(fit, dtype=int), np.asarray(pred, dtype=int)
    cache = OUT / f"{name}_margin_features.joblib"
    lineage_path = OUT / f"{name}_margin_lineage.json"
    if cache.exists() and lineage_path.exists():
        pack = joblib.load(cache)
        assert pack["protocol_sha256"] == protocol_sha256
        np.testing.assert_array_equal(pack["fit"], fit)
        np.testing.assert_array_equal(pack["pred"], pred)
        return pack

    lineage = source_pack["lineage"]
    if len(lineage) != 3:
        raise AssertionError(f"{name}: expected exactly three source crossfit views")
    fit_position = {int(index): pos for pos, index in enumerate(fit)}
    train_new = np.full((len(NEW_ARMS), len(fit), 43), np.nan, dtype=np.float32)
    pred_new = np.zeros((len(NEW_ARMS), len(lineage), len(pred), 43), dtype=np.float32)
    seen = np.zeros(len(fit), dtype=bool)
    pred_support, pred_evidence = [], []
    lineage_rows = []

    for view_index, entry in enumerate(lineage):
        refs = np.asarray(entry["refs"], dtype=int)
        queries = np.asarray(entry["queries"], dtype=int)
        if not set(refs).issubset(set(fit)) or not set(queries).issubset(set(fit)):
            raise AssertionError(f"{name}: source crossfit escaped outer fit")
        if set(ctx["groups"][refs]) & set(ctx["groups"][queries]):
            raise AssertionError(f"{name}: source crossfit split a publication group")
        query_pos = np.asarray([fit_position[int(q)] for q in queries], dtype=int)
        if seen[query_pos].any():
            raise AssertionError(f"{name}: duplicate OOF query in source lineage")
        seen[query_pos] = True

        for arm_index, arm in enumerate(NEW_ARMS):
            values, _, _ = features_for_queries(ctx, queries, refs, arm)
            train_new[arm_index, query_pos] = values
            values, supports, evidence = features_for_queries(ctx, pred, refs, arm, keep_evidence=True)
            pred_new[arm_index, view_index] = values
            for row in supports:
                row["partition"] = name
                row["reference_view"] = view_index
                pred_support.append(row)
            for row in evidence:
                row["partition"] = name
                row["reference_view"] = view_index
                pred_evidence.append(row)
        lineage_rows.append(dict(
            reference_view=view_index,
            source_fold=int(entry["fold"]),
            refs=refs.tolist(),
            queries=queries.tolist(),
            refs_sha256=hashlib_sha(refs),
            queries_sha256=hashlib_sha(queries),
        ))

    if not seen.all() or not np.isfinite(train_new).all() or not np.isfinite(pred_new).all():
        raise AssertionError(f"{name}: incomplete/non-finite crossfit features")
    pack = dict(protocol_sha256=protocol_sha256, fit=fit, pred=pred,
                train_new=train_new, pred_new=pred_new,
                pred_support=pred_support, pred_evidence=pred_evidence)
    joblib.dump(pack, cache, compress=3)
    lineage_record = dict(
        partition=name,
        fit=fit.tolist(), pred=pred.tolist(),
        fit_sha256=hashlib_sha(fit), pred_sha256=hashlib_sha(pred),
        views=lineage_rows,
    )
    lineage_path.write_text(json.dumps(lineage_record, indent=2) + "\n")
    return pack


def hashlib_sha(indices):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).view(np.uint8)).hexdigest()

