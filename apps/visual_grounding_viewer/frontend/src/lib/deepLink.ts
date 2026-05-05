import type { VisualizableDocument } from '../types/api'

export interface DeepLinkConfig {
  rootPath: string
  testCasesPath: string
  filePath: string
  pageNumber: number | null
  autoIndex: boolean
}

export interface DeepLinkUrlState {
  rootPath: string
  testCasesPath: string
  filePath: string
  pageNumber: number | null
}

const MANAGED_QUERY_KEYS = ['root_path', 'rootPath', 'test_cases_path', 'testCasesPath', 'file', 'page', 'auto_index', 'autoIndex']

function readQueryParam(params: URLSearchParams, ...keys: string[]): string {
  for (const key of keys) {
    const value = params.get(key)
    if (value !== null) {
      return value
    }
  }
  return ''
}

export function parseAutoIndexParam(value: string | null): boolean {
  if (!value) {
    return false
  }
  const normalized = value.trim().toLowerCase()
  return normalized === '1' || normalized === 'true' || normalized === 'yes'
}

export function parsePageParam(value: string | null): number | null {
  if (!value) {
    return null
  }

  const parsed = Number.parseInt(value.trim(), 10)
  if (!Number.isFinite(parsed) || parsed < 1) {
    return null
  }
  return parsed
}

export function normalizeFilePath(path: string): string {
  const trimmed = path.trim()
  if (!trimmed) {
    return ''
  }

  let normalized = trimmed.replaceAll('\\', '/').replace(/\/{2,}/g, '/').replace(/^\/+/, '')
  while (normalized.startsWith('./')) {
    normalized = normalized.slice(2)
  }
  return normalized
}

export function readDeepLinkConfig(search?: string): DeepLinkConfig {
  const rawSearch = search ?? (typeof window === 'undefined' ? '' : window.location.search)
  const params = new URLSearchParams(rawSearch)
  return {
    rootPath: readQueryParam(params, 'root_path', 'rootPath'),
    testCasesPath: readQueryParam(params, 'test_cases_path', 'testCasesPath'),
    filePath: normalizeFilePath(readQueryParam(params, 'file')),
    pageNumber: parsePageParam(readQueryParam(params, 'page') || null),
    autoIndex: parseAutoIndexParam(readQueryParam(params, 'auto_index', 'autoIndex') || null),
  }
}

export function shouldAutoIndexFromDeepLink(config: DeepLinkConfig): boolean {
  return config.autoIndex || Boolean(config.filePath)
}

export function resolveDocumentFilePath(doc: VisualizableDocument): string {
  const fileName = `${doc.base_name}${doc.source_ext}`
  if (!doc.relative_dir || doc.relative_dir === '.') {
    return fileName
  }
  return `${normalizeFilePath(doc.relative_dir)}/${fileName}`
}

export function findDocumentByFilePath(
  documents: VisualizableDocument[],
  filePath: string,
): VisualizableDocument | null {
  const normalizedTarget = normalizeFilePath(filePath)
  if (!normalizedTarget) {
    return null
  }

  return documents.find((doc) => resolveDocumentFilePath(doc) === normalizedTarget) ?? null
}

export function buildDeepLinkSearch(search: string, state: DeepLinkUrlState): string {
  const params = new URLSearchParams(search)

  for (const key of MANAGED_QUERY_KEYS) {
    params.delete(key)
  }

  if (state.rootPath.trim()) {
    params.set('root_path', state.rootPath.trim())
  }
  if (state.testCasesPath.trim()) {
    params.set('test_cases_path', state.testCasesPath.trim())
  }

  const normalizedFilePath = normalizeFilePath(state.filePath)
  if (normalizedFilePath) {
    params.set('file', normalizedFilePath)
    if (state.pageNumber !== null && state.pageNumber >= 1) {
      params.set('page', String(state.pageNumber))
    }
  }

  const nextSearch = params.toString()
  return nextSearch ? `?${nextSearch}` : ''
}

export function syncDeepLinkUrl(state: DeepLinkUrlState): void {
  if (typeof window === 'undefined') {
    return
  }

  const nextSearch = buildDeepLinkSearch(window.location.search, state)
  if (window.location.search === nextSearch) {
    return
  }

  const nextUrl = `${window.location.pathname}${nextSearch}${window.location.hash}`
  window.history.replaceState(null, '', nextUrl)
}
