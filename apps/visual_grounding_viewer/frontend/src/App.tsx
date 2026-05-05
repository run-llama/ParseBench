import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import './App.css'
import { indexFolder, loadDocument, pageAssetUrl, sourceAssetUrl } from './api/client'
import { DirectoryBrowserModal } from './components/DirectoryBrowserModal'
import { FolderTree } from './components/FolderTree'
import { MarkdownPane } from './components/MarkdownPane'
import { RightPanel } from './components/RightPanel'
import { ViewerPane } from './components/ViewerPane'
import {
  findDocumentByFilePath,
  readDeepLinkConfig,
  resolveDocumentFilePath,
  shouldAutoIndexFromDeepLink,
  syncDeepLinkUrl,
} from './lib/deepLink'
import { findGranularUnitById, findItemById, type OverlayLayerName, type OverlayLayerVisibility } from './lib/grounding'
import type {
  DocumentResponse,
  GroundingGranularUnit,
  GroundingGranularity,
  IndexResponse,
  VisualizableDocument,
} from './types/api'

const DEFAULT_ROOT = import.meta.env.VITE_DEFAULT_ROOT_PATH ?? ''
const MARKDOWN_PANEL_DEFAULT_WIDTH = 420
const MARKDOWN_PANEL_MIN_WIDTH = 280
const MARKDOWN_PANEL_MAX_WIDTH = 720
const RIGHT_PANEL_DEFAULT_WIDTH = 420
const RIGHT_PANEL_MIN_WIDTH = 280
const RIGHT_PANEL_MAX_WIDTH_FALLBACK = 720
const DEFAULT_VISIBLE_LAYERS: OverlayLayerVisibility = {
  layout: true,
  container: false,
  line: true,
  word: false,
  cell: true,
  field: true,
}
const LLAMAINDEX_LOGO_URL = `${import.meta.env.BASE_URL}llamaindex-favicon.ico`

type DocumentSortDirection = 'highest' | 'lowest'

type BrowseTarget = 'results' | 'test_cases'
type ResizeTarget = 'markdown' | 'right'

const STORAGE_KEYS = {
  markdownPanelOpen: 'visual-grounding-viewer:markdown-panel-open:v2',
  markdownPanelWidth: 'visual-grounding-viewer:markdown-panel-width',
  rightPanelOpen: 'visual-grounding-viewer:right-panel-open',
  rightPanelWidth: 'visual-grounding-viewer:right-panel-width',
} as const

function readStoredBoolean(key: string, fallback: boolean): boolean {
  if (typeof window === 'undefined') {
    return fallback
  }
  const stored = window.localStorage.getItem(key)
  if (stored === null) {
    return fallback
  }
  return stored === 'true'
}

function readStoredNumber(key: string, fallback: number): number {
  if (typeof window === 'undefined') {
    return fallback
  }
  const stored = Number(window.localStorage.getItem(key))
  return Number.isFinite(stored) && stored > 0 ? stored : fallback
}

function rightPanelMaxWidth(): number {
  if (typeof window === 'undefined') {
    return RIGHT_PANEL_MAX_WIDTH_FALLBACK
  }
  return Math.max(RIGHT_PANEL_MIN_WIDTH, Math.floor(window.innerWidth * 0.5))
}

function clampRightPanelWidth(width: number): number {
  return Math.max(RIGHT_PANEL_MIN_WIDTH, Math.min(rightPanelMaxWidth(), width))
}

function isTextInputTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null
  if (!element) {
    return false
  }
  const tagName = element.tagName
  return (
    tagName === 'INPUT' ||
    tagName === 'TEXTAREA' ||
    tagName === 'SELECT' ||
    element.isContentEditable
  )
}

function formatMarkdownSource(source: DocumentResponse['selected_markdown_source']): string | null {
  if (source === 'sidecar_md') {
    return 'sidecar markdown'
  }
  if (source === 'raw') {
    return 'raw.json'
  }
  if (source === 'result') {
    return 'result.json'
  }
  return null
}

function formatDocumentDisplayName(relativeDir: string, baseName: string): string {
  return relativeDir && relativeDir !== '.' ? `${relativeDir}/${baseName}` : baseName
}

function formatDocumentMetricLabel(metricName: string): string {
  return metricName.replaceAll('_', ' ')
}

function pickDefaultDocumentMetric(metricNames: string[]): string {
  const preferredOrder = [
    'mean_f1',
    'mAP@[.50:.95]',
    'layout_rule_pass_rate',
    'layout_element_rule_pass_rate',
    'parse_field_element_pass_rate',
    'parse_field_rule_pass_rate',
    'extract_element_pass_rate',
    'extract_value_f1',
    'extract_value_pass_rate',
    'f1',
  ]
  for (const preferred of preferredOrder) {
    if (metricNames.includes(preferred)) {
      return preferred
    }
  }
  return metricNames[0] ?? ''
}

function App() {
  const deepLinkConfig = useMemo(() => readDeepLinkConfig(), [])
  const deepLinkFilePath = deepLinkConfig.filePath
  const deepLinkPageNumber = deepLinkConfig.pageNumber
  const [rootPath, setRootPath] = useState(() => deepLinkConfig.rootPath || DEFAULT_ROOT)
  const [testCasesPath, setTestCasesPath] = useState(() => deepLinkConfig.testCasesPath)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [indexedRootPath, setIndexedRootPath] = useState('')
  const [indexedTestCasesPath, setIndexedTestCasesPath] = useState('')

  const [indexData, setIndexData] = useState<IndexResponse | null>(null)
  const [indexError, setIndexError] = useState<string | null>(null)
  const [indexLoading, setIndexLoading] = useState(false)
  const [deepLinkError, setDeepLinkError] = useState<string | null>(null)

  const [search, setSearch] = useState('')
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [pendingFilePath, setPendingFilePath] = useState(() => deepLinkFilePath)
  const [pendingPageNumber, setPendingPageNumber] = useState<number | null>(() => deepLinkPageNumber)

  const [documentData, setDocumentData] = useState<DocumentResponse | null>(null)
  const [documentLoading, setDocumentLoading] = useState(false)
  const [documentError, setDocumentError] = useState<string | null>(null)

  const [currentPageIndex, setCurrentPageIndex] = useState(0)
  const [activeItemId, setActiveItemId] = useState<string | null>(null)
  const [hoveredItemId, setHoveredItemId] = useState<string | null>(null)
  const [activeGranularUnitId, setActiveGranularUnitId] = useState<string | null>(null)
  const [hoveredGranularUnitId, setHoveredGranularUnitId] = useState<string | null>(null)
  const [activeGranularPreview, setActiveGranularPreview] = useState<GroundingGranularUnit | null>(null)
  const [hoveredGranularPreview, setHoveredGranularPreview] = useState<GroundingGranularUnit | null>(null)
  const [activeGtRuleId, setActiveGtRuleId] = useState<string | null>(null)
  const [hoveredGtRuleId, setHoveredGtRuleId] = useState<string | null>(null)
  const [activeEvidenceGtRuleIds, setActiveEvidenceGtRuleIds] = useState<string[]>([])
  const [hoveredEvidenceGtRuleIds, setHoveredEvidenceGtRuleIds] = useState<string[]>([])
  const [hoverSource, setHoverSource] = useState<'viewer' | 'sidebar' | null>(null)
  const [visibleLayers, setVisibleLayers] = useState<OverlayLayerVisibility>(DEFAULT_VISIBLE_LAYERS)

  const [browseTarget, setBrowseTarget] = useState<BrowseTarget | null>(null)
  const [indexControlsOpen, setIndexControlsOpen] = useState(true)
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true)
  const [documentSortDirection, setDocumentSortDirection] = useState<DocumentSortDirection>('highest')
  const [documentSortMetric, setDocumentSortMetric] = useState<string>('')
  const [hasConfiguredDocumentSort, setHasConfiguredDocumentSort] = useState(false)
  const [markdownPanelOpen, setMarkdownPanelOpen] = useState(() =>
    readStoredBoolean(STORAGE_KEYS.markdownPanelOpen, false),
  )
  const [rightPanelOpen, setRightPanelOpen] = useState(() =>
    readStoredBoolean(STORAGE_KEYS.rightPanelOpen, true),
  )
  const [markdownPanelWidth, setMarkdownPanelWidth] = useState(() =>
    readStoredNumber(STORAGE_KEYS.markdownPanelWidth, MARKDOWN_PANEL_DEFAULT_WIDTH),
  )
  const [rightPanelWidth, setRightPanelWidth] = useState(() =>
    clampRightPanelWidth(readStoredNumber(STORAGE_KEYS.rightPanelWidth, RIGHT_PANEL_DEFAULT_WIDTH)),
  )
  const resizeStateRef = useRef<{ target: ResizeTarget; startX: number; startWidth: number } | null>(null)
  const autoIndexTriggeredRef = useRef(false)

  useEffect(() => {
    const onMouseMove = (event: MouseEvent) => {
      const resizeState = resizeStateRef.current
      if (!resizeState) {
        return
      }

      const delta = resizeState.startX - event.clientX
      if (resizeState.target === 'markdown') {
        const nextWidth = Math.max(
          MARKDOWN_PANEL_MIN_WIDTH,
          Math.min(MARKDOWN_PANEL_MAX_WIDTH, resizeState.startWidth + delta),
        )
        setMarkdownPanelWidth(nextWidth)
        return
      }

      const nextWidth = Math.max(
        RIGHT_PANEL_MIN_WIDTH,
        Math.min(rightPanelMaxWidth(), resizeState.startWidth + delta),
      )
      setRightPanelWidth(nextWidth)
    }

    const onMouseUp = () => {
      if (!resizeStateRef.current) {
        return
      }
      resizeStateRef.current = null
      document.body.classList.remove('is-resizing')
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isTextInputTarget(event.target)) {
        return
      }

      if (event.key === '[') {
        event.preventDefault()
        setLeftSidebarOpen((value) => !value)
      }

      if (event.key === ']') {
        event.preventDefault()
        setRightPanelOpen((value) => !value)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  useEffect(() => {
    const onResize = () => {
      setRightPanelWidth((current) => clampRightPanelWidth(current))
    }

    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
    }
  }, [])

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEYS.markdownPanelOpen, String(markdownPanelOpen))
  }, [markdownPanelOpen])

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEYS.rightPanelOpen, String(rightPanelOpen))
  }, [rightPanelOpen])

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEYS.markdownPanelWidth, String(markdownPanelWidth))
  }, [markdownPanelWidth])

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEYS.rightPanelWidth, String(rightPanelWidth))
  }, [rightPanelWidth])

  const availableDocumentMetrics = useMemo(() => {
    if (!indexData) {
      return []
    }
    const metricNames = new Set<string>()
    for (const doc of indexData.documents) {
      for (const metricName of Object.keys(doc.evaluation_metrics ?? {})) {
        metricNames.add(metricName)
      }
    }
    return [...metricNames].sort((left, right) => left.localeCompare(right))
  }, [indexData])
  const effectiveDocumentSortMetric = availableDocumentMetrics.includes(documentSortMetric) ? documentSortMetric : ''

  useEffect(() => {
    if (availableDocumentMetrics.length === 0) {
      if (documentSortMetric !== '') {
        setDocumentSortMetric('')
      }
      if (hasConfiguredDocumentSort) {
        setHasConfiguredDocumentSort(false)
      }
      return
    }
    if (!hasConfiguredDocumentSort && !effectiveDocumentSortMetric) {
      setDocumentSortMetric(pickDefaultDocumentMetric(availableDocumentMetrics))
      return
    }
    if (documentSortMetric && !availableDocumentMetrics.includes(documentSortMetric)) {
      setDocumentSortMetric(pickDefaultDocumentMetric(availableDocumentMetrics))
    }
  }, [availableDocumentMetrics, documentSortMetric, effectiveDocumentSortMetric, hasConfiguredDocumentSort])

  const visibleDocuments = useMemo(() => {
    if (!indexData) {
      return []
    }
    const query = search.trim().toLowerCase()
    const filtered = indexData.documents.filter((doc) => {
      const haystack = `${doc.base_name} ${doc.relative_dir}`.toLowerCase()
      return haystack.includes(query)
    })
    if (!effectiveDocumentSortMetric) {
      return filtered
    }

    return [...filtered].sort((left, right) => {
      const leftValue = left.evaluation_metrics?.[effectiveDocumentSortMetric]
      const rightValue = right.evaluation_metrics?.[effectiveDocumentSortMetric]
      const leftMissing = leftValue === undefined || Number.isNaN(leftValue)
      const rightMissing = rightValue === undefined || Number.isNaN(rightValue)
      if (leftMissing !== rightMissing) {
        return leftMissing ? 1 : -1
      }
      const safeLeftValue = leftValue ?? Number.NEGATIVE_INFINITY
      const safeRightValue = rightValue ?? Number.NEGATIVE_INFINITY
      if (safeLeftValue !== safeRightValue) {
        return documentSortDirection === 'highest' ? safeRightValue - safeLeftValue : safeLeftValue - safeRightValue
      }
      return `${left.relative_dir}/${left.base_name}`.localeCompare(`${right.relative_dir}/${right.base_name}`)
    })
  }, [documentSortDirection, effectiveDocumentSortMetric, indexData, search])

  const selectedDocumentSummary: VisualizableDocument | null = useMemo(() => {
    if (!indexData || !selectedDocId) {
      return null
    }
    return indexData.documents.find((doc) => doc.doc_id === selectedDocId) ?? null
  }, [indexData, selectedDocId])

  const selectedDocIndex = useMemo(() => {
    if (!selectedDocId) {
      return -1
    }
    return visibleDocuments.findIndex((doc) => doc.doc_id === selectedDocId)
  }, [selectedDocId, visibleDocuments])

  const currentPageData = useMemo(() => {
    if (!documentData) {
      return null
    }
    if (documentData.pages.length === 0) {
      return null
    }
    return documentData.pages[currentPageIndex] ?? documentData.pages[0]
  }, [currentPageIndex, documentData])

  const currentPageGtRules = useMemo(() => {
    if (!currentPageData) {
      return []
    }
    return (currentPageData.gt_rules ?? []).filter((rule) => rule.page_number === currentPageData.page_number)
  }, [currentPageData])

  const selectedItem = useMemo(() => {
    if (!currentPageData) {
      return null
    }
    return findItemById(currentPageData.items, activeItemId)
  }, [activeItemId, currentPageData])

  const selectedGranularUnit = useMemo(() => {
    if (!currentPageData) {
      return null
    }
    return findGranularUnitById(currentPageData, activeGranularUnitId)
  }, [activeGranularUnitId, currentPageData])

  const hoveredGranularUnit = useMemo(() => {
    if (!currentPageData) {
      return null
    }
    return findGranularUnitById(currentPageData, hoveredGranularUnitId)
  }, [currentPageData, hoveredGranularUnitId])

  const selectedGtRule = useMemo(() => {
    if (!activeGtRuleId) {
      return null
    }
    return currentPageGtRules.find((rule) => rule.rule_id === activeGtRuleId) ?? null
  }, [activeGtRuleId, currentPageGtRules])

  const hoveredGtRule = useMemo(() => {
    if (!hoveredGtRuleId) {
      return null
    }
    return currentPageGtRules.find((rule) => rule.rule_id === hoveredGtRuleId) ?? null
  }, [currentPageGtRules, hoveredGtRuleId])

  const selectedEvidenceGtRules = useMemo(() => {
    if (!activeGtRuleId || !activeEvidenceGtRuleIds.includes(activeGtRuleId)) {
      return []
    }
    const activeIds = new Set(activeEvidenceGtRuleIds)
    return currentPageGtRules.filter((rule) => activeIds.has(rule.rule_id))
  }, [activeEvidenceGtRuleIds, activeGtRuleId, currentPageGtRules])

  const hoveredEvidenceGtRules = useMemo(() => {
    if (!hoveredGtRuleId || !hoveredEvidenceGtRuleIds.includes(hoveredGtRuleId)) {
      return []
    }
    const hoveredIds = new Set(hoveredEvidenceGtRuleIds)
    return currentPageGtRules.filter((rule) => hoveredIds.has(rule.rule_id))
  }, [currentPageGtRules, hoveredEvidenceGtRuleIds, hoveredGtRuleId])

  const viewerActiveGtRules = useMemo(
    () => (selectedEvidenceGtRules.length > 0 ? selectedEvidenceGtRules : selectedGtRule ? [selectedGtRule] : []),
    [selectedEvidenceGtRules, selectedGtRule],
  )
  const viewerHoveredGtRules = useMemo(
    () => (hoveredEvidenceGtRules.length > 0 ? hoveredEvidenceGtRules : hoveredGtRule ? [hoveredGtRule] : []),
    [hoveredEvidenceGtRules, hoveredGtRule],
  )

  const currentPreviewMarkdown = useMemo(() => {
    if (!documentData || !currentPageData) {
      return null
    }
    return currentPageData.markdown ?? documentData.document_markdown
  }, [currentPageData, documentData])

  const currentPreviewSource = useMemo(
    () => formatMarkdownSource(documentData?.selected_markdown_source ?? null),
    [documentData?.selected_markdown_source],
  )
  const currentSourceUrl = useMemo(() => {
    if (!sessionId || !documentData) {
      return null
    }
    return sourceAssetUrl(sessionId, documentData.doc_id)
  }, [documentData, sessionId])

  const hasPreviewPanel = Boolean(currentPreviewMarkdown)
  const previewPanelVisible = hasPreviewPanel && markdownPanelOpen
  const viewerTitle = useMemo(
    () => (selectedDocumentSummary ? formatDocumentDisplayName(selectedDocumentSummary.relative_dir, selectedDocumentSummary.base_name) : ''),
    [selectedDocumentSummary],
  )

  const onIndex = useCallback(async () => {
    setIndexLoading(true)
    setIndexError(null)
    setDeepLinkError(null)
    setDocumentData(null)
    setDocumentError(null)
    setSelectedDocId(null)
    setSessionId(null)
    setPendingFilePath(deepLinkFilePath)
    setPendingPageNumber(deepLinkPageNumber)
    setHasConfiguredDocumentSort(false)
    setDocumentSortMetric('')
    try {
      const data = await indexFolder({ rootPath, testCasesPath })
      setIndexData(data)
      setSessionId(data.session_id)
      setIndexedRootPath(rootPath)
      setIndexedTestCasesPath(testCasesPath)
    } catch (error) {
      setIndexError(error instanceof Error ? error.message : String(error))
    } finally {
      setIndexLoading(false)
    }
  }, [deepLinkFilePath, deepLinkPageNumber, rootPath, testCasesPath])

  useEffect(() => {
    if (!shouldAutoIndexFromDeepLink(deepLinkConfig) || autoIndexTriggeredRef.current) {
      return
    }
    if (!rootPath.trim()) {
      return
    }
    autoIndexTriggeredRef.current = true
    void onIndex()
  }, [deepLinkConfig, onIndex, rootPath])

  useEffect(() => {
    if (!deepLinkError) {
      return
    }

    const timeoutId = window.setTimeout(() => {
      setDeepLinkError(null)
    }, 5000)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [deepLinkError])

  const onSelectDoc = useCallback(
    async (docId: string, pageMode: 'first' | 'last' = 'first', explicitPageNumber: number | null = null) => {
      if (!sessionId) {
        setDocumentError('Missing session_id. Re-index the folder.')
        return
      }
      setSelectedDocId(docId)
      setDeepLinkError(null)
      setDocumentLoading(true)
      setDocumentError(null)
      setActiveItemId(null)
      setHoveredItemId(null)
      setActiveGranularUnitId(null)
      setHoveredGranularUnitId(null)
      setActiveGranularPreview(null)
      setHoveredGranularPreview(null)
      setActiveGtRuleId(null)
      setHoveredGtRuleId(null)
      setHoverSource(null)

      try {
        const document = await loadDocument(sessionId, docId)
        setDocumentData(document)
        const initialIndex =
          explicitPageNumber !== null
            ? Math.min(Math.max(explicitPageNumber - 1, 0), Math.max(document.pages.length - 1, 0))
            : pageMode === 'last'
              ? Math.max(0, document.pages.length - 1)
              : 0
        setCurrentPageIndex(initialIndex)
      } catch (error) {
        setDocumentError(error instanceof Error ? error.message : String(error))
        setDocumentData(null)
      } finally {
        setDocumentLoading(false)
      }
    },
    [sessionId],
  )

  useEffect(() => {
    if (!indexData || !sessionId || !pendingFilePath) {
      return
    }

    const matchedDocument = findDocumentByFilePath(indexData.documents, pendingFilePath)
    const requestedPageNumber = pendingPageNumber
    setPendingFilePath('')
    setPendingPageNumber(null)

    if (!matchedDocument) {
      setDeepLinkError(`Deep-linked file not found in indexed results: ${pendingFilePath}`)
      return
    }

    void onSelectDoc(matchedDocument.doc_id, 'first', requestedPageNumber)
  }, [indexData, onSelectDoc, pendingFilePath, pendingPageNumber, sessionId])

  useEffect(() => {
    if (!visibleDocuments.length || pendingFilePath) {
      return
    }

    if (!selectedDocId || !visibleDocuments.some((doc) => doc.doc_id === selectedDocId)) {
      void onSelectDoc(visibleDocuments[0].doc_id)
    }
  }, [onSelectDoc, pendingFilePath, selectedDocId, visibleDocuments])

  useEffect(() => {
    if (!selectedDocumentSummary || !currentPageData || !documentData) {
      return
    }
    if (documentData.doc_id !== selectedDocumentSummary.doc_id) {
      return
    }

    syncDeepLinkUrl({
      rootPath: indexedRootPath,
      testCasesPath: indexedTestCasesPath,
      filePath: resolveDocumentFilePath(selectedDocumentSummary),
      pageNumber: currentPageData.page_number,
    })
  }, [currentPageData, documentData, indexedRootPath, indexedTestCasesPath, selectedDocumentSummary])

  const goToDocByOffset = async (delta: number, pageMode: 'first' | 'last' = 'first') => {
    if (!selectedDocId || visibleDocuments.length === 0) {
      return false
    }

    const currentIndex = visibleDocuments.findIndex((doc) => doc.doc_id === selectedDocId)
    if (currentIndex < 0) {
      return false
    }

    const nextIndex = currentIndex + delta
    if (nextIndex < 0 || nextIndex >= visibleDocuments.length) {
      return false
    }

    await onSelectDoc(visibleDocuments[nextIndex].doc_id, pageMode)
    return true
  }

  const goToPrevPage = () => {
    if (!documentData) {
      return
    }
    setActiveItemId(null)
    setHoveredItemId(null)
    setActiveGranularUnitId(null)
    setHoveredGranularUnitId(null)
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setHoverSource(null)
    if (currentPageIndex > 0) {
      setCurrentPageIndex((value) => value - 1)
      return
    }
    void goToDocByOffset(-1, 'last')
  }

  const goToNextPage = () => {
    if (!documentData) {
      return
    }
    setActiveItemId(null)
    setHoveredItemId(null)
    setActiveGranularUnitId(null)
    setHoveredGranularUnitId(null)
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setHoverSource(null)
    if (currentPageIndex < documentData.pages.length - 1) {
      setCurrentPageIndex((value) => value + 1)
      return
    }
    void goToDocByOffset(1, 'first')
  }

  const handleViewerHover = (itemId: string | null) => {
    setHoveredItemId(itemId)
    setHoveredGranularUnitId(null)
    setHoveredGranularPreview(null)
    setHoveredGtRuleId(null)
    setHoverSource(itemId ? 'viewer' : null)
  }

  const handleSidebarHover = (itemId: string | null) => {
    setHoveredItemId(itemId)
    setHoveredGranularUnitId(null)
    setHoveredGranularPreview(null)
    setHoveredGtRuleId(null)
    setHoverSource(itemId ? 'sidebar' : null)
  }

  const handleSelectItem = (itemId: string) => {
    setActiveItemId(itemId)
    setActiveGranularUnitId(null)
    setHoveredGranularUnitId(null)
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setHoverSource(null)
  }

  const handleViewerGranularHover = (unitId: string | null, _granularity: GroundingGranularity | null) => {
    void _granularity
    setHoveredItemId(null)
    setHoveredGranularUnitId(unitId)
    setHoveredGranularPreview(null)
    setHoveredGtRuleId(null)
    setHoverSource(unitId ? 'viewer' : null)
  }

  const handleSidebarGranularHover = (unitId: string | null, _granularity: GroundingGranularity | null) => {
    void _granularity
    setHoveredItemId(null)
    setHoveredGranularUnitId(unitId)
    setHoveredGranularPreview(null)
    setHoveredGtRuleId(null)
    setHoverSource(unitId ? 'sidebar' : null)
  }

  const handleSelectGranularUnit = (unitId: string, _granularity: GroundingGranularity) => {
    void _granularity
    setActiveItemId(null)
    setHoveredItemId(null)
    setActiveGranularUnitId(unitId)
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setHoverSource(null)
  }

  const handleSidebarGranularPreviewHover = (unit: GroundingGranularUnit | null) => {
    setHoveredItemId(null)
    setHoveredGranularUnitId(null)
    setHoveredGranularPreview(unit)
    setHoveredGtRuleId(null)
    setHoverSource(unit ? 'sidebar' : null)
  }

  const handleSelectGranularPreview = (unit: GroundingGranularUnit | null) => {
    setActiveItemId(null)
    setHoveredItemId(null)
    setActiveGranularUnitId(null)
    setHoveredGranularUnitId(null)
    setActiveGranularPreview(unit)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setHoverSource(null)
  }

  const handleSidebarGtRuleHover = (ruleId: string | null) => {
    setHoveredItemId(null)
    setHoveredGranularUnitId(null)
    setHoveredGranularPreview(null)
    setHoveredEvidenceGtRuleIds([])
    setHoveredGtRuleId(ruleId)
    setHoverSource(ruleId ? 'sidebar' : null)
  }

  const handleSidebarEvidenceHover = (itemId: string | null, ruleIds: string[]) => {
    setHoveredItemId(itemId)
    setHoveredGranularUnitId(null)
    setHoveredGranularPreview(null)
    setHoveredEvidenceGtRuleIds(ruleIds)
    setHoveredGtRuleId(ruleIds[0] ?? null)
    setHoverSource(itemId || ruleIds.length > 0 ? 'sidebar' : null)
  }

  const handleSelectGtRule = (ruleId: string) => {
    setActiveItemId(null)
    setHoveredItemId(null)
    setActiveGranularUnitId(null)
    setHoveredGranularUnitId(null)
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveEvidenceGtRuleIds([])
    setHoveredEvidenceGtRuleIds([])
    setActiveGtRuleId(ruleId)
    setHoveredGtRuleId(null)
    setHoverSource(null)
  }

  const handleSelectEvidence = (itemId: string | null, ruleIds: string[]) => {
    setActiveItemId(itemId)
    setHoveredItemId(null)
    setActiveGranularUnitId(null)
    setHoveredGranularUnitId(null)
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveEvidenceGtRuleIds(ruleIds)
    setHoveredEvidenceGtRuleIds([])
    setActiveGtRuleId(ruleIds[0] ?? null)
    setHoveredGtRuleId(null)
    setHoverSource(null)
  }

  const toggleLayer = (layer: OverlayLayerName) => {
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setVisibleLayers((current) => ({
      ...current,
      [layer]: !current[layer],
    }))
  }

  const showAllLayers = () => {
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setVisibleLayers({
      layout: true,
      container: true,
      line: true,
      word: true,
      cell: true,
      field: true,
    })
  }

  const showLayoutOnly = () => {
    setActiveGranularPreview(null)
    setHoveredGranularPreview(null)
    setActiveGtRuleId(null)
    setHoveredGtRuleId(null)
    setVisibleLayers({
      layout: true,
      container: false,
      line: false,
      word: false,
      cell: false,
      field: false,
    })
  }

  const openBrowse = (target: BrowseTarget) => setBrowseTarget(target)

  const browseTitle = browseTarget === 'results' ? 'Select results directory' : 'Select test-cases directory'
  const browseInitialPath = browseTarget === 'results' ? rootPath : testCasesPath

  const handleBrowseSelect = (selectedPath: string) => {
    if (browseTarget === 'results') {
      setRootPath(selectedPath)
    } else if (browseTarget === 'test_cases') {
      setTestCasesPath(selectedPath)
    }
  }

  const startResize = (target: ResizeTarget, startWidth: number, event: React.MouseEvent<HTMLDivElement>) => {
    resizeStateRef.current = {
      target,
      startX: event.clientX,
      startWidth,
    }
    document.body.classList.add('is-resizing')
    event.preventDefault()
  }

  const workspaceClassName = ['workspace-grid', leftSidebarOpen ? '' : 'sidebar-collapsed'].filter(Boolean).join(' ')

  return (
    <div className="app-shell">
      <section className="index-panel">
        <button
          className="index-panel-header"
          onClick={() => setIndexControlsOpen((current) => !current)}
          aria-expanded={indexControlsOpen}
        >
          <div className="index-panel-header-main">
            <span className="index-panel-brand">
              <img src={LLAMAINDEX_LOGO_URL} alt="" aria-hidden="true" />
              <span className="index-panel-title">
                ParseBench <span>Grounding</span>
              </span>
            </span>
            {indexData ? (
              <div className="index-panel-summary">
                <span>Visualizable: {indexData.counts.visualizable}</span>
                <span>Skipped: {indexData.counts.skipped}</span>
                <span>Warnings: {indexData.counts.warnings}</span>
              </div>
            ) : (
              <span className="index-panel-subtitle">Results path, optional test cases path, and indexing controls</span>
            )}
          </div>
          <span className="index-panel-chevron" aria-hidden="true">
            {indexControlsOpen ? '▾' : '▸'}
          </span>
        </button>

        {indexControlsOpen ? (
          <div className="index-panel-body">
            <section className="index-controls">
              <div className="path-input-group">
                <div className="path-input-row">
                  <label className="path-input-inline-label" htmlFor="results-path-input">
                    Results Path
                  </label>
                  <input
                    id="results-path-input"
                    type="text"
                    value={rootPath}
                    onChange={(event) => setRootPath(event.target.value)}
                    placeholder="/path/to/benchmark/run"
                  />
                  <button onClick={() => openBrowse('results')}>Browse</button>
                </div>
              </div>

              <div className="path-input-group">
                <div className="path-input-row">
                  <label className="path-input-inline-label" htmlFor="test-cases-path-input">
                    Test Cases Path (Optional)
                  </label>
                  <input
                    id="test-cases-path-input"
                    type="text"
                    value={testCasesPath}
                    onChange={(event) => setTestCasesPath(event.target.value)}
                    placeholder="Auto-detected from _metadata.json when empty"
                  />
                  <button onClick={() => openBrowse('test_cases')}>Browse</button>
                </div>
              </div>

              <button onClick={onIndex} disabled={indexLoading || !rootPath.trim()}>
                {indexLoading ? 'Indexing…' : 'Index Folder'}
              </button>
            </section>

            {indexData && indexData.warnings.length > 0 ? (
              <details className="warning-box">
                <summary>View warnings</summary>
                <ul>
                  {indexData.warnings.slice(0, 200).map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </details>
            ) : null}
          </div>
        ) : null}
      </section>

      {indexError ? <div className="error-box">{indexError}</div> : null}
      {deepLinkError ? <div className="error-box">{deepLinkError}</div> : null}
      {documentError ? <div className="error-box">{documentError}</div> : null}

      <main className={workspaceClassName}>
        <aside className={leftSidebarOpen ? 'left-sidebar' : 'left-sidebar collapsed'}>
          <header className="panel-header sidebar-panel-header">
            <div>
              <h3>Documents</h3>
              <span>{indexData ? `${indexData.document_total} indexed` : 'Index a folder to begin'}</span>
            </div>
          </header>

          {indexData ? (
            <>
              <div className="sidebar-controls">
                <input
                  type="text"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search documents…"
                />
                {availableDocumentMetrics.length > 0 ? (
                  <div className="search-controls-row">
                    <select
                      value={documentSortDirection}
                      onChange={(event) => setDocumentSortDirection(event.target.value as DocumentSortDirection)}
                      disabled={!effectiveDocumentSortMetric}
                    >
                      <option value="highest">Highest</option>
                      <option value="lowest">Lowest</option>
                    </select>
                    <select
                      value={effectiveDocumentSortMetric}
                      onChange={(event) => {
                        setHasConfiguredDocumentSort(true)
                        setDocumentSortMetric(event.target.value)
                      }}
                    >
                      <option value="">Default order</option>
                      {availableDocumentMetrics.map((metricName) => (
                        <option key={metricName} value={metricName}>
                          {formatDocumentMetricLabel(metricName)}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : null}
              </div>
              <div className="sidebar-content">
                <FolderTree
                  root={indexData.tree}
                  documents={visibleDocuments}
                  selectedDocId={selectedDocId}
                  sortMetric={effectiveDocumentSortMetric || null}
                  sortDirection={documentSortDirection}
                  onSelectDoc={(docId) => void onSelectDoc(docId)}
                />
              </div>
            </>
          ) : (
            <p className="muted">Index a folder to begin.</p>
          )}
        </aside>

        <section className="viewer-column">
          {documentLoading ? <p className="muted">Loading document…</p> : null}

          {documentData && currentPageData && selectedDocumentSummary ? (
            <>
              <div className="viewer-toolbar">
                <div className="viewer-title">
                  <div className="viewer-title-row">
                    <button
                      className="viewer-sidebar-toggle"
                      onClick={() => setLeftSidebarOpen((current) => !current)}
                      aria-label={leftSidebarOpen ? 'Collapse document sidebar' : 'Open document sidebar'}
                      title={leftSidebarOpen ? 'Collapse document sidebar' : 'Open document sidebar'}
                    >
                      ☰
                    </button>
                    <strong>{viewerTitle}</strong>
                    {documentData.source_kind === 'pdf' && currentSourceUrl ? (
                      <a
                        className="viewer-source-link"
                        href={currentSourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        [show original pdf]
                      </a>
                    ) : null}
                  </div>
                </div>
                <div className="viewer-actions">
                  {hasPreviewPanel ? (
                    <button onClick={() => setMarkdownPanelOpen((current) => !current)}>
                      {markdownPanelOpen ? 'Hide Preview' : 'Show Preview'}
                    </button>
                  ) : null}
                  <button onClick={() => void goToDocByOffset(-1)} disabled={selectedDocIndex <= 0}>
                    Prev Doc
                  </button>
                  <button
                    onClick={() => void goToDocByOffset(1)}
                    disabled={selectedDocIndex < 0 || selectedDocIndex >= visibleDocuments.length - 1}
                  >
                    Next Doc
                  </button>
                  <button onClick={goToPrevPage} disabled={selectedDocIndex <= 0 && currentPageIndex <= 0}>
                    Prev Page
                  </button>
                  <button
                    onClick={goToNextPage}
                    disabled={
                      (selectedDocIndex < 0 || selectedDocIndex >= visibleDocuments.length - 1) &&
                      currentPageIndex >= documentData.pages.length - 1
                    }
                  >
                    Next Page
                  </button>
                </div>
              </div>

              <div className="viewer-layout">
                <div className="viewer-main">
                  <ViewerPane
                    key={`${documentData.doc_id}-${currentPageData.page_number}`}
                    page={currentPageData}
                    sourceKind={documentData.source_kind}
                    sourceUrl={currentSourceUrl}
                    assetUrl={pageAssetUrl(sessionId ?? '', documentData.doc_id, currentPageData.page_number)}
                    hoverSource={hoverSource}
                    visibleLayers={visibleLayers}
                    activeItemId={activeItemId}
                    hoveredItemId={hoveredItemId}
                    activeGranularUnitId={activeGranularUnitId}
                    hoveredGranularUnitId={hoveredGranularUnitId}
                    activeGranularPreview={activeGranularPreview}
                    hoveredGranularPreview={hoveredGranularPreview}
                    activeGtRules={viewerActiveGtRules}
                    hoveredGtRules={viewerHoveredGtRules}
                    onToggleLayer={toggleLayer}
                    onShowAllLayers={showAllLayers}
                    onShowLayoutOnly={showLayoutOnly}
                    onHoverItem={handleViewerHover}
                    onSelectItem={handleSelectItem}
                    onSelectEvidence={handleSelectEvidence}
                    onHoverGranularUnit={handleViewerGranularHover}
                    onSelectGranularUnit={handleSelectGranularUnit}
                  />
                </div>

                {previewPanelVisible ? (
                  <>
                    <div
                      className="panel-resizer"
                      role="separator"
                      aria-label="Resize markdown preview"
                      aria-orientation="vertical"
                      onMouseDown={(event) => startResize('markdown', markdownPanelWidth, event)}
                    />
                    <div className="markdown-panel-wrap" style={{ width: `${markdownPanelWidth}px` }}>
                      <MarkdownPane
                        markdown={currentPreviewMarkdown}
                        pageLabel={`Page ${currentPageData.page_number}/${documentData.pages.length}`}
                        markdownSource={currentPreviewSource}
                        items={currentPageData.items}
                        activeItemId={activeItemId}
                        hoveredItemId={hoveredItemId}
                        hoverSource={hoverSource}
                        onCollapse={() => setMarkdownPanelOpen(false)}
                        onHoverItem={handleSidebarHover}
                        onSelectItem={handleSelectItem}
                      />
                    </div>
                  </>
                ) : null}

                {hasPreviewPanel && !markdownPanelOpen ? (
                  <button
                    className="panel-toggle-strip"
                    onClick={() => setMarkdownPanelOpen(true)}
                    aria-label="Open markdown preview"
                  >
                    ←
                  </button>
                ) : null}

                {rightPanelOpen ? (
                  <>
                    <div
                      className="panel-resizer"
                      role="separator"
                      aria-label="Resize right sidebar"
                      aria-orientation="vertical"
                      onMouseDown={(event) => startResize('right', rightPanelWidth, event)}
                    />
                    <div className="right-panel-wrap" style={{ width: `${rightPanelWidth}px` }}>
                      <RightPanel
                        document={documentData}
                        pageItems={currentPageData.items}
                        pageGranularLayers={currentPageData.granular_layers}
                        pageGtRules={currentPageGtRules}
                        visibleLayers={visibleLayers}
                        activeItemId={activeItemId}
                        hoveredItemId={hoveredItemId}
                        activeGranularUnit={selectedGranularUnit}
                        hoveredGranularUnit={hoveredGranularUnit}
                        activeGranularPreview={activeGranularPreview}
                        hoveredGranularPreview={hoveredGranularPreview}
                        activeGtRule={selectedGtRule}
                        hoveredGtRule={hoveredGtRule}
                        hoverSource={hoverSource}
                        onHoverItem={handleSidebarHover}
                        onSelectItem={handleSelectItem}
                        onHoverGranularUnit={handleSidebarGranularHover}
                        onSelectGranularUnit={handleSelectGranularUnit}
                        onHoverGranularPreview={handleSidebarGranularPreviewHover}
                        onSelectGranularPreview={handleSelectGranularPreview}
                        onHoverGtRule={handleSidebarGtRuleHover}
                        onSelectGtRule={handleSelectGtRule}
                        onHoverEvidence={handleSidebarEvidenceHover}
                        onSelectEvidence={handleSelectEvidence}
                        onCollapse={() => setRightPanelOpen(false)}
                      />
                    </div>
                  </>
                ) : (
                  <button
                    className="panel-toggle-float panel-toggle-float-right"
                    onClick={() => setRightPanelOpen(true)}
                    aria-label="Open right sidebar"
                  >
                    ←
                  </button>
                )}
              </div>

              <footer className="selection-footer">
                {selectedItem ? (
                  <>
                    <strong>{selectedItem.type}</strong> · page {currentPageIndex + 1}/{documentData.pages.length} ·{' '}
                    {selectedItem.md.slice(0, 160)}
                  </>
                ) : selectedGranularUnit ? (
                  <>
                    <strong>{selectedGranularUnit.granularity}</strong> · page {currentPageIndex + 1}/
                    {documentData.pages.length} · {selectedGranularUnit.text || selectedGranularUnit.unit_id}
                  </>
                ) : activeGranularPreview ? (
                  <>
                    <strong>{activeGranularPreview.granularity}</strong> · page {currentPageIndex + 1}/
                    {documentData.pages.length} · {activeGranularPreview.text || activeGranularPreview.unit_id}
                  </>
                ) : hoveredGtRule ?? selectedGtRule ? (
                  <>
                    {(() => {
                      const focusedRule = hoveredGtRule ?? selectedGtRule
                      if (!focusedRule) {
                        return null
                      }
                      if (focusedRule.rule_type === 'layout') {
                        return (
                          <>
                            <strong>layout</strong> · page {currentPageIndex + 1}/{documentData.pages.length} ·{' '}
                            {focusedRule.canonical_class ?? 'layout'}
                            {focusedRule.gt_ro_index !== null ? ` · ro:${focusedRule.gt_ro_index}` : ''}
                          </>
                        )
                      }
                      const isStray = (focusedRule.tags ?? []).includes('stray_evidence')
                      return (
                        <>
                          <strong>{focusedRule.rule_type}</strong> · page {currentPageIndex + 1}/
                          {documentData.pages.length} · {focusedRule.field_path} ·{' '}
                          {String(focusedRule.expected_value ?? '')}
                          {isStray ? ' · stray' : ''}
                          {focusedRule.verified === false ? ' · unverified' : ''}
                        </>
                      )
                    })()}
                  </>
                ) : (
                  <span className="muted">Hover/click markdown or bounding boxes to inspect grounding.</span>
                )}
              </footer>
            </>
          ) : (
            <p className="muted">Select a document to visualize.</p>
          )}
        </section>
      </main>

      <DirectoryBrowserModal
        open={browseTarget !== null}
        title={browseTitle}
        initialPath={browseInitialPath}
        onClose={() => setBrowseTarget(null)}
        onSelect={handleBrowseSelect}
      />
    </div>
  )
}

export default App
