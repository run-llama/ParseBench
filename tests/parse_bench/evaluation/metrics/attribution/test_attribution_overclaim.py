"""Prediction-side duplicate-claim / overclaim diagnostics for attribution."""

from parse_bench.evaluation.metrics.attribution.core import (
    GTElement,
    PredBlock,
    compute_attribution_metrics,
    parse_pred_blocks,
)
from parse_bench.evaluation.metrics.attribution.text_utils import (
    normalize_attribution_text,
    tokenize,
)


def _make_gt(bbox_xyxy, text, cls="Text", ro_index=0, attrs=None):
    norm = normalize_attribution_text(text)
    return GTElement(
        bbox_coco=[
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1],
        ],
        bbox_xyxy=bbox_xyxy,
        canonical_class=cls,
        text=text,
        normalized_text=norm,
        tokens=tokenize(norm),
        ro_index=ro_index,
        content_type="text",
        attributes=attrs or {},
    )


def _make_pred(bbox_xyxy, text, label="text", order=0, block_type="text"):
    norm = normalize_attribution_text(text)
    return PredBlock(
        bbox_xyxy=bbox_xyxy,
        block_type=block_type,
        label=label,
        text=text,
        normalized_text=norm,
        tokens=tokenize(norm),
        order_index=order,
    )


class TestGroundingOverclaim:
    def test_exact_duplicate_local_segment_is_redundant_overclaim(self):
        gt = [_make_gt([0.1, 0.1, 0.6, 0.2], "notes")]
        pred = [
            _make_pred([0.1, 0.1, 0.6, 0.2], "notes", order=0),
            _make_pred([0.1, 0.1, 0.6, 0.2], "notes", order=1),
        ]

        result = compute_attribution_metrics(gt, pred)

        assert result.overclaim_metric_available is True
        assert result.supported_claim_count == 2
        assert result.duplicate_claim_count == 1
        assert result.claimed_supported_token_instances == 1
        assert result.duplicate_supported_token_instances == 1
        assert abs(result.grounding_overclaim_rate - 0.5) < 1e-9
        assert result.grounding_duplicate_supported_token_rate == 1.0
        assert result.num_scored_pred_blocks == 2
        assert result.supported_tp_blocks == 1
        assert result.redundant_fp_blocks == 1
        assert abs(result.grounding_supported_block_precision - 0.5) < 1e-9
        assert abs(result.grounding_redundant_block_rate - 0.5) < 1e-9

    def test_repeated_tokens_use_distinct_gt_token_instances(self):
        gt = [_make_gt([0.1, 0.1, 0.6, 0.2], "notes notes")]
        pred = [
            _make_pred([0.1, 0.1, 0.6, 0.2], "notes notes", order=0),
            _make_pred([0.1, 0.1, 0.6, 0.2], "notes", order=1),
        ]

        result = compute_attribution_metrics(gt, pred)

        assert result.claimed_supported_token_instances == 2
        assert result.duplicate_supported_token_instances == 1
        assert result.supported_claim_count == 3
        assert result.duplicate_claim_count == 1
        assert abs(result.grounding_overclaim_rate - (1.0 / 3.0)) < 1e-9
        assert result.supported_tp_blocks == 1
        assert result.redundant_fp_blocks == 1

    def test_remote_substring_claim_is_unsupported_fp(self):
        gt = [
            _make_gt([0.0, 0.0, 0.4, 0.2], "alpha beta", ro_index=0),
            _make_gt([0.6, 0.0, 1.0, 0.2], "notes", ro_index=1),
        ]
        pred = [
            _make_pred([0.0, 0.0, 0.4, 0.2], "alpha beta", order=0),
            _make_pred([0.0, 0.0, 0.4, 0.2], "notes", order=1),
        ]

        result = compute_attribution_metrics(gt, pred)

        assert result.overclaim_metric_available is True
        assert result.num_scored_pred_blocks == 2
        assert result.supported_tp_blocks == 1
        assert result.unsupported_fp_blocks == 1
        assert result.redundant_fp_blocks == 0
        assert result.duplicate_claim_count == 0
        assert abs(result.grounding_supported_block_precision - 0.5) < 1e-9
        assert abs(result.grounding_unsupported_block_rate - 0.5) < 1e-9

    def test_overlapping_empty_image_like_block_is_unsupported_fp(self):
        gt = [_make_gt([0.1, 0.1, 0.6, 0.2], "alpha beta")]
        pred = [
            _make_pred([0.1, 0.1, 0.6, 0.2], "alpha beta", order=0),
            _make_pred([0.1, 0.1, 0.6, 0.2], "", label="image", order=1, block_type="image"),
        ]

        result = compute_attribution_metrics(gt, pred)

        assert result.overclaim_metric_available is True
        assert result.num_scored_pred_blocks == 2
        assert result.supported_tp_blocks == 1
        assert result.unsupported_fp_blocks == 1
        assert result.spatial_fp_blocks == 0
        assert abs(result.grounding_unsupported_block_rate - 0.5) < 1e-9

    def test_existing_metrics_unchanged_by_diagnostics(self):
        """LAP/LAR/AF1/grounding accuracy are untouched by the new fields."""
        gt = [_make_gt([0.1, 0.1, 0.6, 0.2], "alpha beta")]
        pred = [_make_pred([0.1, 0.1, 0.6, 0.2], "alpha beta")]
        result = compute_attribution_metrics(gt, pred)
        assert result.lap == 1.0
        assert result.lar == 1.0
        assert result.af1 == 1.0
        assert result.grounding_accuracy == 1.0
        assert result.grounding_overclaim_rate == 0.0


class TestRequireLayoutAwareSegments:
    def test_default_keeps_bbox_only_items(self):
        items = [
            {
                "type": "text",
                "value": "container fallback",
                "bBox": {"x": 100, "y": 100, "w": 200, "h": 50, "label": "text"},
            }
        ]
        blocks = parse_pred_blocks(items, page_md="", page_width=1000.0, page_height=1000.0)
        assert len(blocks) == 1
        assert blocks[0].text == "container fallback"

    def test_require_layout_aware_segments_skips_bbox_only_items(self):
        items = [
            {
                "type": "text",
                "value": "container fallback",
                "bBox": {"x": 100, "y": 100, "w": 200, "h": 50, "label": "text"},
            }
        ]

        blocks = parse_pred_blocks(
            items,
            page_md="",
            page_width=1000.0,
            page_height=1000.0,
            require_layout_aware_segments=True,
        )
        assert blocks == []

    def test_require_layout_aware_segments_keeps_segmented_items(self):
        items = [
            {
                "type": "text",
                "value": "abcdef",
                "bBox": {"x": 100, "y": 100, "w": 200, "h": 50, "label": "text"},
                "layoutAwareBbox": [
                    {
                        "x": 100,
                        "y": 100,
                        "w": 100,
                        "h": 50,
                        "label": "text",
                        "startIndex": 1,
                        "endIndex": 3,
                    }
                ],
            }
        ]
        blocks = parse_pred_blocks(
            items,
            page_md="",
            page_width=1000.0,
            page_height=1000.0,
            require_layout_aware_segments=True,
        )
        assert len(blocks) == 1
        assert blocks[0].text == "bcd"
