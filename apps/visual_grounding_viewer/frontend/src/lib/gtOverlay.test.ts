import { describe, expect, it } from 'vitest'

import type { GroundTruthRuleMatch } from '../types/api'
import { computeGtOverlayMetrics, partitionGtOverlayRegions } from './gtOverlay'

const baseRule: GroundTruthRuleMatch = {
  rule_id: 'rule-1',
  rule_type: 'extract_field',
  page_number: 1,
  field_path: 'record_id',
  expected_value: 'REC-0000',
  evidence_index: 0,
  gt_bbox: { x: 10, y: 10, w: 20, h: 10, label: 'GT', confidence: null, start_index: null, end_index: null },
  predicted_bbox: { x: 15, y: 10, w: 20, h: 10, label: 'Pred', confidence: null, start_index: null, end_index: null },
  predicted_bboxes: [{ x: 15, y: 10, w: 20, h: 10, label: 'word', confidence: null, start_index: null, end_index: null }],
  predicted_text: 'REC-0000',
  predicted_granularity: 'word',
  matched_unit_ids: ['word-2'],
  iou: 0.6,
  bbox_recall: 0.75,
  text_score: 1,
}

describe('partitionGtOverlayRegions', () => {
  it('splits overlap, gt-only, and pred-only regions for a rule', () => {
    const partition = partitionGtOverlayRegions(baseRule)

    expect(partition.overlap).toHaveLength(1)
    expect(partition.overlap[0]).toMatchObject({ x: 15, y: 10, w: 15, h: 10 })

    expect(partition.gtOnly).toHaveLength(1)
    expect(partition.gtOnly[0]).toMatchObject({ x: 10, y: 10, w: 5, h: 10 })

    expect(partition.predOnly).toHaveLength(1)
    expect(partition.predOnly[0]).toMatchObject({ x: 30, y: 10, w: 5, h: 10 })
  })

  it('shows the full gt box as gt-only when there is no prediction', () => {
    const partition = partitionGtOverlayRegions({
      ...baseRule,
      predicted_bbox: null,
      predicted_bboxes: [],
      predicted_text: null,
      predicted_granularity: null,
      matched_unit_ids: [],
      iou: null,
      bbox_recall: null,
      text_score: null,
    })

    expect(partition.overlap).toEqual([])
    expect(partition.predOnly).toEqual([])
    expect(partition.gtOnly).toHaveLength(1)
    expect(partition.gtOnly[0]).toMatchObject({ x: 10, y: 10, w: 20, h: 10 })
  })

  it('uses explicit prediction bboxes when supplied for display partitioning', () => {
    const broadMetricPrediction = {
      ...baseRule,
      predicted_bbox: { x: 0, y: 0, w: 100, h: 100, label: 'Pred', confidence: null, start_index: null, end_index: null },
      predicted_bboxes: [
        { x: 0, y: 0, w: 100, h: 100, label: 'Pred', confidence: null, start_index: null, end_index: null },
      ],
    }
    const partition = partitionGtOverlayRegions(broadMetricPrediction, [
      { x: 15, y: 10, w: 20, h: 10, label: 'word', confidence: null, start_index: null, end_index: null },
    ])

    expect(partition.overlap).toHaveLength(1)
    expect(partition.overlap[0]).toMatchObject({ x: 15, y: 10, w: 15, h: 10 })
    expect(partition.predOnly).toHaveLength(1)
    expect(partition.predOnly[0]).toMatchObject({ x: 30, y: 10, w: 5, h: 10 })
  })
})

describe('computeGtOverlayMetrics', () => {
  it('computes geometry precision, recall, f1, and iou from support regions', () => {
    const metrics = computeGtOverlayMetrics(baseRule)

    expect(metrics.precision).toBeCloseTo(0.75, 6)
    expect(metrics.recall).toBeCloseTo(0.75, 6)
    expect(metrics.f1).toBeCloseTo(0.75, 6)
    expect(metrics.iou).toBeCloseTo(0.6, 6)
  })
})
