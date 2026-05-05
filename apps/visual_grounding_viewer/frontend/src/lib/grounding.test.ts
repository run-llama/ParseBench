import { describe, expect, it } from 'vitest'

import type { GroundingPage } from '../types/api'
import {
  boxesForPage,
  findGranularLayer,
  findGranularUnitById,
  findItemById,
  formatGranularUnitLabel,
  formatGranularUnitMetadata,
  isContainerItem,
  itemCountForLayer,
} from './grounding'

const page: GroundingPage = {
  page_number: 1,
  page_width: 100,
  page_height: 200,
  markdown: 'hello',
  items: [
    {
      item_id: 'p1-i0',
      item_index: 0,
      page_number: 1,
      depth: 0,
      type: 'text',
      md: 'hello',
      value: null,
      source_path: 'items.0',
      raw_payload: null,
      bboxes: [
        {
          x: 10,
          y: 20,
          w: 30,
          h: 40,
          label: 'text',
          confidence: 0.9,
          start_index: 0,
          end_index: 5,
        },
      ],
    },
    {
      item_id: 'p1-i1',
      item_index: 1,
      page_number: 1,
      depth: 0,
      type: 'list',
      md: '* hello',
      value: null,
      source_path: 'items.1',
      raw_payload: null,
      bboxes: [
        {
          x: 50,
          y: 30,
          w: 20,
          h: 30,
          label: 'list-item',
          confidence: 0.8,
          start_index: 0,
          end_index: 7,
        },
      ],
    },
  ],
  granular_layers: [
    {
      granularity: 'line',
      availability: 'available',
      reason: null,
      source: 'textract',
      units: [
        {
          unit_id: 'line-1',
          granularity: 'line',
          order_index: 1,
          text: 'hello world',
          bbox: {
            x: 8,
            y: 18,
            w: 36,
            h: 14,
            label: null,
            confidence: null,
            start_index: null,
            end_index: null,
          },
          bboxes: [],
          row_index: null,
          column_index: null,
          row_span: null,
          column_span: null,
          source_path: 'pages.0.lines.0',
          provider: 'textract',
        },
      ],
    },
    {
      granularity: 'word',
      availability: 'available',
      reason: null,
      source: 'textract',
      units: [
        {
          unit_id: 'word-1',
          granularity: 'word',
          order_index: 2,
          text: 'hello',
          bbox: {
            x: 10,
            y: 20,
            w: 12,
            h: 10,
            label: null,
            confidence: null,
            start_index: null,
            end_index: null,
          },
          bboxes: [],
          row_index: null,
          column_index: null,
          row_span: null,
          column_span: null,
          source_path: 'pages.0.words.0',
          provider: 'textract',
        },
      ],
    },
    {
      granularity: 'cell',
      availability: 'available',
      reason: null,
      source: 'llamaparse',
      units: [
        {
          unit_id: 'cell-1',
          granularity: 'cell',
          order_index: 3,
          text: '42',
          bbox: {
            x: 60,
            y: 80,
            w: 20,
            h: 12,
            label: null,
            confidence: null,
            start_index: null,
            end_index: null,
          },
          bboxes: [
            {
              x: 60,
              y: 80,
              w: 8,
              h: 12,
              label: null,
              confidence: null,
              start_index: null,
              end_index: null,
            },
            {
              x: 72,
              y: 80,
              w: 8,
              h: 12,
              label: null,
              confidence: null,
              start_index: null,
              end_index: null,
            },
          ],
          row_index: 0,
          column_index: 1,
          row_span: 1,
          column_span: 2,
          source_path: 'tables.0.rows.0.cells.1',
          provider: 'llamaparse',
        },
      ],
    },
  ],
}

describe('boxesForPage', () => {
  it('flattens item and granular bboxes into overlay boxes', () => {
    const boxes = boxesForPage(page)
    expect(boxes).toHaveLength(5)
    expect(boxes.map((box) => box.layer)).toEqual(['layout', 'cell', 'cell', 'line', 'word'])
    expect(boxes[0].itemId).toBe('p1-i0')
    expect(boxes[0].colorKey).toBe('layout-text')
    expect(boxes[1].unitId).toBe('cell-1')
    expect(boxes[2].x).toBe(72)
    expect(boxes[4].unitId).toBe('word-1')
    expect(boxes[4].colorKey).toBe('granular-word')
  })

  it('filters out disabled layers', () => {
    const boxes = boxesForPage(page, {
      layout: true,
      container: false,
      line: false,
      word: true,
      cell: false,
      field: false,
    })

    expect(boxes.map((box) => box.layer)).toEqual(['layout', 'word'])
  })

  it('surfaces container items on their own layer', () => {
    const boxes = boxesForPage(page, {
      layout: false,
      container: true,
      line: false,
      word: false,
      cell: false,
      field: false,
    })

    expect(boxes).toHaveLength(1)
    expect(boxes[0].layer).toBe('container')
    expect(boxes[0].colorKey).toBe('container-list')
  })

  it('colors by item type even when bbox labels are generic', () => {
    const pageWithGenericLabel: GroundingPage = {
      ...page,
      items: [
        {
          ...page.items[0],
          type: 'table',
          bboxes: [
            {
              ...page.items[0].bboxes[0],
              label: 'Text',
            },
          ],
        },
      ],
    }

    const boxes = boxesForPage(pageWithGenericLabel)
    expect(boxes[0].label).toBe('Text')
    expect(boxes[0].colorKey).toBe('layout-table')
  })

  it('colors generic text items by bbox class when available', () => {
    const pageWithSectionHeader: GroundingPage = {
      ...page,
      items: [
        {
          ...page.items[0],
          type: 'text',
          bboxes: [
            {
              ...page.items[0].bboxes[0],
              label: 'Section-header',
            },
          ],
        },
      ],
    }

    const boxes = boxesForPage(pageWithSectionHeader)
    expect(boxes[0].label).toBe('Section-header')
    expect(boxes[0].colorKey).toBe('layout-section-header')
  })

  it('suppresses reading-order badges for extract evidence boxes', () => {
    const pageWithExtractEvidence: GroundingPage = {
      ...page,
      items: [
        {
          ...page.items[0],
          item_id: 'p1-extract-citation-0',
          type: 'extract_field',
          source_path: 'field_citations.0',
        },
      ],
      granular_layers: [],
    }

    const boxes = boxesForPage(pageWithExtractEvidence)
    expect(boxes).toHaveLength(1)
    expect(boxes[0].layer).toBe('field')
    expect(boxes[0].colorKey).toBe('field-unmatched')
    expect(boxes[0].readingOrder).toBe(0)
    expect(boxes[0].showReadingOrder).toBe(false)
    expect(boxes[0].isExtractEvidence).toBe(true)
    expect(itemCountForLayer(pageWithExtractEvidence, 'layout')).toBe(0)
    expect(itemCountForLayer(pageWithExtractEvidence, 'field')).toBe(1)
  })
})

describe('finders', () => {
  it('finds matching layout item', () => {
    expect(findItemById(page.items, 'p1-i0')?.md).toBe('hello')
    expect(findItemById(page.items, 'missing')).toBeNull()
  })

  it('identifies container items separately from layout items', () => {
    expect(isContainerItem(page.items[0])).toBe(false)
    expect(isContainerItem(page.items[1])).toBe(true)
  })

  it('finds granular layers and units', () => {
    expect(findGranularLayer(page, 'cell')?.source).toBe('llamaparse')
    expect(findGranularUnitById(page, 'cell-1')?.text).toBe('42')
    expect(findGranularUnitById(page, 'missing')).toBeNull()
  })
})

describe('granular labeling', () => {
  it('formats cell labels and metadata for inspection', () => {
    const unit = page.granular_layers[2].units[0]
    expect(formatGranularUnitLabel(unit)).toBe('cell r1 c2')
    expect(formatGranularUnitMetadata(unit)).toBe('row 1 · col 2 · colspan 2')
  })
})
