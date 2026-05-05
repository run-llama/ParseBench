import { describe, expect, it } from 'vitest'

import type { FolderNode, VisualizableDocument } from '../types/api'
import { filterDocuments, flattenTree } from './folder'

const docs: VisualizableDocument[] = [
  {
    doc_id: 'a',
    base_name: 'doc-a',
    relative_dir: 'suite/one',
    source_kind: 'pdf',
    source_ext: '.pdf',
    last_modified_ms: 2_000,
    artifact_flags: {
      has_v2_items_file: true,
      has_raw_file: true,
      has_result_file: true,
      has_v2_items_payload: true,
    },
  },
  {
    doc_id: 'b',
    base_name: 'doc-b',
    relative_dir: 'suite/two',
    source_kind: 'image',
    source_ext: '.png',
    last_modified_ms: 1_000,
    artifact_flags: {
      has_v2_items_file: true,
      has_raw_file: false,
      has_result_file: false,
      has_v2_items_payload: true,
    },
  },
]

describe('filterDocuments', () => {
  it('filters by folder subtree', () => {
    const results = filterDocuments(docs, 'suite/one', '')
    expect(results.map((doc) => doc.doc_id)).toEqual(['a'])
  })

  it('filters by search query', () => {
    const results = filterDocuments(docs, '.', 'doc-b')
    expect(results.map((doc) => doc.doc_id)).toEqual(['b'])
  })
})

describe('flattenTree', () => {
  it('returns all nodes in preorder', () => {
    const tree: FolderNode = {
      name: '.',
      path: '.',
      document_count: 0,
      total_document_count: 2,
      children: [
        {
          name: 'suite',
          path: 'suite',
          document_count: 0,
          total_document_count: 2,
          children: [
            {
              name: 'one',
              path: 'suite/one',
              document_count: 1,
              total_document_count: 1,
              children: [],
            },
          ],
        },
      ],
    }

    expect(flattenTree(tree).map((node) => node.path)).toEqual(['.', 'suite', 'suite/one'])
  })
})
