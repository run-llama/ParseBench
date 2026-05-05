import type { BrowseResponse, DocumentResponse, IndexResponse } from '../types/api'

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? 'http://127.0.0.1:8011' : '')
const FALLBACK_ORIGIN = 'http://127.0.0.1'

export interface IndexFolderParams {
  rootPath: string
  testCasesPath?: string
}

function apiUrl(path: string): URL {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  if (API_BASE) {
    return new URL(normalizedPath, API_BASE.endsWith('/') ? API_BASE : `${API_BASE}/`)
  }

  const origin = typeof window === 'undefined' ? FALLBACK_ORIGIN : window.location.origin
  return new URL(normalizedPath, origin)
}

async function fetchJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed: ${response.status}`)
  }
  return (await response.json()) as T
}

export async function indexFolder(params: IndexFolderParams): Promise<IndexResponse> {
  return fetchJson<IndexResponse>(apiUrl('/api/index').toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      root_path: params.rootPath,
      test_cases_path: params.testCasesPath?.trim() || null,
      page: 1,
      page_size: 10000,
    }),
  })
}

export async function browseDirectory(path?: string): Promise<BrowseResponse> {
  const url = apiUrl('/api/browse')
  if (path && path.trim()) {
    url.searchParams.set('path', path.trim())
  }
  return fetchJson<BrowseResponse>(url.toString())
}

export async function loadDocument(sessionId: string, docId: string): Promise<DocumentResponse> {
  const url = apiUrl('/api/document')
  url.searchParams.set('session_id', sessionId)
  url.searchParams.set('doc_id', docId)
  return fetchJson<DocumentResponse>(url.toString())
}

export function pageAssetUrl(sessionId: string, docId: string, page: number): string {
  const url = apiUrl('/api/page_asset')
  url.searchParams.set('session_id', sessionId)
  url.searchParams.set('doc_id', docId)
  url.searchParams.set('page', String(page))
  return url.toString()
}

export function sourceAssetUrl(sessionId: string, docId: string): string {
  const url = apiUrl('/api/source_asset')
  url.searchParams.set('session_id', sessionId)
  url.searchParams.set('doc_id', docId)
  return url.toString()
}

export function healthUrl(): string {
  return apiUrl('/api/health').toString()
}
