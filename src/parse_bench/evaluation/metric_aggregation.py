"""Shared metric aggregation helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

CountTriple = tuple[int, int, int]


def add_precision_recall_f1_aggregates(
    aggregate: dict[str, float],
    metric_counts: Mapping[str, Sequence[CountTriple]],
    *,
    counted_metric_names: Collection[str] = (),
) -> None:
    """Add total TP/FP/FN and pooled ``micro_*`` aggregates from tp/fp/fn metadata.

    ``avg_*`` stays the document-weighted macro average; the pooled counters are
    exposed through explicit ``micro_*`` keys. Each metric is pooled from its
    own tp/fp/fn, so a standalone ``*_f1`` / ``*accuracy`` / ``*pass_rate``
    metric gets a micro value even when the precision/recall/f1 trio is not
    emitted together:

    - ``*_precision`` -> tp / (tp + fp)
    - ``*_recall``    -> tp / (tp + fn)
    - ``*_f1``        -> harmonic mean of the two
    - ``*accuracy`` and ``*pass_rate`` -> tp / (tp + fp + fn)

    ``counted_metric_names`` lists metrics that already have a pooled
    ``micro_*`` from ``passed``/``total`` metadata; their pass-rate micro is
    left alone so the rule-count denominator wins over the tp/fp/fn one.
    """
    for metric_name, counts in metric_counts.items():
        tp = sum(item[0] for item in counts)
        fp = sum(item[1] for item in counts)
        fn = sum(item[2] for item in counts)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if metric_name == "precision" or metric_name.endswith("_precision"):
            aggregate[f"micro_{metric_name}"] = precision
        elif metric_name == "recall" or metric_name.endswith("_recall"):
            aggregate[f"micro_{metric_name}"] = recall
        elif metric_name == "f1" or metric_name.endswith("_f1"):
            aggregate[f"micro_{metric_name}"] = f1
        elif metric_name.endswith("accuracy"):
            total = tp + fp + fn
            if total > 0:
                aggregate[f"micro_{metric_name}"] = tp / total
        elif metric_name.endswith("pass_rate") and metric_name not in counted_metric_names:
            total = tp + fp + fn
            if total > 0:
                aggregate[f"micro_{metric_name}"] = tp / total
        aggregate[f"total_{metric_name}_tp"] = float(tp)
        aggregate[f"total_{metric_name}_fp"] = float(fp)
        aggregate[f"total_{metric_name}_fn"] = float(fn)
