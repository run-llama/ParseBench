import { describe, expect, it } from 'vitest'

import type { VisualizableDocument } from '../types/api'
import {
  buildDeepLinkSearch,
  findDocumentByFilePath,
  normalizeFilePath,
  parsePageParam,
  readDeepLinkConfig,
  resolveDocumentFilePath,
  shouldAutoIndexFromDeepLink,
} from './deepLink'

const docs: VisualizableDocument[] = [
  {
    doc_id: 'root',
    base_name: 'doc',
    relative_dir: '.',
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
    doc_id: 'nested',
    base_name: 'report',
    relative_dir: 'suite/one',
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

describe('readDeepLinkConfig', () => {
  it('reads legacy deep-link params plus the file target', () => {
    const config = readDeepLinkConfig(
      '?root_path=/tmp/results&test_cases_path=/tmp/tests&auto_index=1&file=suite/one/report.png&page=3',
    )

    expect(config).toEqual({
      rootPath: '/tmp/results',
      testCasesPath: '/tmp/tests',
      filePath: 'suite/one/report.png',
      pageNumber: 3,
      autoIndex: true,
    })
  })
})

describe('parsePageParam', () => {
  it('accepts positive page numbers and rejects invalid values', () => {
    expect(parsePageParam('2')).toBe(2)
    expect(parsePageParam('0')).toBeNull()
    expect(parsePageParam('-1')).toBeNull()
    expect(parsePageParam('abc')).toBeNull()
  })
})

describe('normalizeFilePath', () => {
  it('trims leading slashes, dot prefixes, and duplicate separators', () => {
    expect(normalizeFilePath('  /./suite//one/report.png  ')).toBe('suite/one/report.png')
  })
})

describe('resolveDocumentFilePath', () => {
  it('builds root and nested relative source paths', () => {
    expect(resolveDocumentFilePath(docs[0])).toBe('doc.pdf')
    expect(resolveDocumentFilePath(docs[1])).toBe('suite/one/report.png')
  })
})

describe('findDocumentByFilePath', () => {
  it('matches a document by normalized relative path', () => {
    expect(findDocumentByFilePath(docs, '/suite/one/report.png')?.doc_id).toBe('nested')
  })

  it('returns null for unknown file targets', () => {
    expect(findDocumentByFilePath(docs, 'missing/file.pdf')).toBeNull()
  })
})

describe('shouldAutoIndexFromDeepLink', () => {
  it('auto-indexes when a file target is present even without auto_index', () => {
    expect(
      shouldAutoIndexFromDeepLink({
        rootPath: '/tmp/results',
        testCasesPath: '',
        filePath: 'suite/one/report.png',
        pageNumber: null,
        autoIndex: false,
      }),
    ).toBe(true)
  })
})

describe('buildDeepLinkSearch', () => {
  it('writes canonical deep-link params while preserving unrelated ones', () => {
    expect(
      buildDeepLinkSearch('?view=compact&autoIndex=true', {
        rootPath: '/tmp/results',
        testCasesPath: '/tmp/tests',
        filePath: 'suite/one/report.png',
        pageNumber: 4,
      }),
    ).toBe('?view=compact&root_path=%2Ftmp%2Fresults&test_cases_path=%2Ftmp%2Ftests&file=suite%2Fone%2Freport.png&page=4')
  })

  it('omits page when there is no selected file', () => {
    expect(
      buildDeepLinkSearch('', {
        rootPath: '/tmp/results',
        testCasesPath: '',
        filePath: '',
        pageNumber: 2,
      }),
    ).toBe('?root_path=%2Ftmp%2Fresults')
  })
})
