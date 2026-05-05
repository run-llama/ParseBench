import { describe, expect, it } from 'vitest'

import type { GroundingItem } from '../types/api'
import { groundMarkdownBlocks, splitMarkdownBlocks } from './markdownGrounding'

const items: GroundingItem[] = [
  {
    item_id: 'p1-i0',
    item_index: 0,
    page_number: 1,
    depth: 0,
    type: 'heading',
    md: '# SAMPLE REPORT',
    value: null,
    source_path: 'items.0',
    raw_payload: null,
    bboxes: [],
  },
  {
    item_id: 'p1-i1',
    item_index: 1,
    page_number: 1,
    depth: 0,
    type: 'text',
    md: 'The table immediately below sets out the total\n**EXAMPLE RECORDS**',
    value: null,
    source_path: 'items.1',
    raw_payload: null,
    bboxes: [],
  },
  {
    item_id: 'p1-i2',
    item_index: 2,
    page_number: 1,
    depth: 0,
    type: 'table',
    md: '| Name | Office |\n| --- | --- |\n| Example Person | Example Role |',
    value: null,
    source_path: 'items.2',
    raw_payload: null,
    bboxes: [],
  },
]

describe('splitMarkdownBlocks', () => {
  it('keeps headings and html tables as separate preview blocks', () => {
    const blocks = splitMarkdownBlocks(`# Heading\n\nParagraph\n\n<table>\n<tr><td>A</td></tr>\n</table>\nAfter`)
    expect(blocks).toEqual(['# Heading', 'Paragraph', '<table>\n<tr><td>A</td></tr>\n</table>', 'After'])
  })
})

describe('groundMarkdownBlocks', () => {
  it('zips blocks by order when block count matches item count', () => {
    const blocks = groundMarkdownBlocks(
      `# SAMPLE REPORT\n\nThe table immediately below sets out the total\n**EXAMPLE RECORDS**\n\n<table>\n<tr><th>Name</th><th>Office</th></tr>\n<tr><td>Example Person</td><td>Example Role</td></tr>\n</table>`,
      items,
    )

    expect(blocks).toHaveLength(3)
    expect(blocks.map((block) => block.itemId)).toEqual(['p1-i0', 'p1-i1', 'p1-i2'])
    expect(blocks[2].matchKind).toBe('ordered')
  })

  it('falls back to similarity when markdown blocks and items do not align one-to-one', () => {
    const blocks = groundMarkdownBlocks(
      `<table>\n<tr><th>Name</th><th>Office</th></tr>\n<tr><td>Example Person</td><td>Example Role</td></tr>\n</table>`,
      items,
    )

    expect(blocks).toHaveLength(1)
    expect(blocks[0].itemId).toBe('p1-i2')
    expect(blocks[0].matchKind).toBe('similarity')
  })
})
