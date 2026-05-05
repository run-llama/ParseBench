import { useMemo, useState } from 'react'

import { formatLastModified } from '../lib/time'
import type { FolderNode, VisualizableDocument } from '../types/api'

interface FolderTreeProps {
  root: FolderNode
  documents: VisualizableDocument[]
  selectedDocId: string | null
  sortMetric: string | null
  sortDirection: 'highest' | 'lowest'
  onSelectDoc: (docId: string) => void
}

function formatMetricLabel(metricName: string): string {
  return metricName.replaceAll('_', ' ')
}

function ArtifactBadges({ doc }: { doc: VisualizableDocument }) {
  const flags = doc.artifact_flags
  return (
    <span className="artifact-badges">
      {flags.has_v2_items_file ? <span className="badge">v2</span> : null}
      {flags.has_raw_file ? <span className="badge">raw</span> : null}
      {flags.has_result_file ? <span className="badge">result</span> : null}
    </span>
  )
}

function compareMetricValues(
  leftValue: number | undefined,
  rightValue: number | undefined,
  direction: 'highest' | 'lowest',
): number {
  const leftMissing = leftValue === undefined || Number.isNaN(leftValue)
  const rightMissing = rightValue === undefined || Number.isNaN(rightValue)
  if (leftMissing !== rightMissing) {
    return leftMissing ? 1 : -1
  }
  if (leftMissing && rightMissing) {
    return 0
  }
  return direction === 'highest' ? (rightValue ?? 0) - (leftValue ?? 0) : (leftValue ?? 0) - (rightValue ?? 0)
}

export function FolderTree({ root, documents, selectedDocId, sortMetric, sortDirection, onSelectDoc }: FolderTreeProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const docsByFolder = useMemo(() => {
    const map = new Map<string, VisualizableDocument[]>()
    for (const doc of documents) {
      const key = doc.relative_dir
      if (!map.has(key)) {
        map.set(key, [])
      }
      map.get(key)!.push(doc)
    }

    return map
  }, [documents])

  const visibleFolderPaths = useMemo(() => {
    const visible = new Set<string>()

    function markVisible(node: FolderNode): boolean {
      const hasDirectDocs = (docsByFolder.get(node.path)?.length ?? 0) > 0
      let hasVisibleChild = false
      for (const child of node.children) {
        if (markVisible(child)) {
          hasVisibleChild = true
        }
      }
      const include = hasDirectDocs || hasVisibleChild || node.path === '.'
      if (include) {
        visible.add(node.path)
      }
      return include
    }

    markVisible(root)
    return visible
  }, [docsByFolder, root])

  const subtreeCounts = useMemo(() => {
    const counts = new Map<string, number>()

    function walk(node: FolderNode): number {
      let total = docsByFolder.get(node.path)?.length ?? 0
      for (const child of node.children) {
        total += walk(child)
      }
      counts.set(node.path, total)
      return total
    }

    walk(root)
    return counts
  }, [docsByFolder, root])

  const subtreeLatestModified = useMemo(() => {
    const latest = new Map<string, number>()

    function walk(node: FolderNode): number {
      let maxMtime = 0
      for (const doc of docsByFolder.get(node.path) ?? []) {
        maxMtime = Math.max(maxMtime, doc.last_modified_ms)
      }
      for (const child of node.children) {
        maxMtime = Math.max(maxMtime, walk(child))
      }
      latest.set(node.path, maxMtime)
      return maxMtime
    }

    walk(root)
    return latest
  }, [docsByFolder, root])

  const subtreeMetricValue = useMemo(() => {
    const metricByPath = new Map<string, number | undefined>()

    function walk(node: FolderNode): number | undefined {
      const values: number[] = []
      for (const doc of docsByFolder.get(node.path) ?? []) {
        const metricValue = sortMetric ? doc.evaluation_metrics?.[sortMetric] : undefined
        if (metricValue !== undefined && !Number.isNaN(metricValue)) {
          values.push(metricValue)
        }
      }
      for (const child of node.children) {
        const childValue = walk(child)
        if (childValue !== undefined && !Number.isNaN(childValue)) {
          values.push(childValue)
        }
      }
      const aggregate =
        values.length === 0
          ? undefined
          : sortDirection === 'highest'
            ? Math.max(...values)
            : Math.min(...values)
      metricByPath.set(node.path, aggregate)
      return aggregate
    }

    walk(root)
    return metricByPath
  }, [docsByFolder, root, sortDirection, sortMetric])

  const toggleFolder = (path: string) => {
    setCollapsed((prev) => ({
      ...prev,
      [path]: !prev[path],
    }))
  }

  function renderNode(node: FolderNode, depth: number) {
    if (!visibleFolderPaths.has(node.path)) {
      return null
    }

    const totalCount = subtreeCounts.get(node.path) ?? 0
    const isRoot = node.path === '.'
    const isCollapsed = isRoot ? false : Boolean(collapsed[node.path])
    const directDocs = [...(docsByFolder.get(node.path) ?? [])]

    if (sortMetric) {
      directDocs.sort((left, right) => {
        const metricDiff = compareMetricValues(
          left.evaluation_metrics?.[sortMetric],
          right.evaluation_metrics?.[sortMetric],
          sortDirection,
        )
        if (metricDiff !== 0) {
          return metricDiff
        }
        return left.base_name.localeCompare(right.base_name)
      })
    }

    return (
      <li key={node.path}>
        <button
          className="folder-row"
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
          onClick={() => toggleFolder(node.path)}
        >
          <span className="folder-name">
            {!isRoot ? (isCollapsed ? '▸ ' : '▾ ') : ''}
            {isRoot ? 'Root' : node.name}
          </span>
          <span className="folder-count">{totalCount}</span>
        </button>

        {!isCollapsed ? (
          <>
            {directDocs.length > 0 ? (
              <ul className="tree-files">
                {directDocs.map((doc) => {
                  const selected = selectedDocId === doc.doc_id
                  return (
                    <li key={doc.doc_id}>
                      <button
                        className={selected ? 'file-row selected' : 'file-row'}
                        style={{ paddingLeft: `${(depth + 1) * 12 + 18}px` }}
                        onClick={() => onSelectDoc(doc.doc_id)}
                      >
                        <span className="file-name">{doc.base_name}</span>
                        <span className="document-meta">
                          {sortMetric ? (
                            <span className="document-timestamp">
                              {formatMetricLabel(sortMetric)}:{' '}
                              {doc.evaluation_metrics?.[sortMetric] !== undefined
                                ? doc.evaluation_metrics?.[sortMetric]?.toFixed(3)
                                : 'n/a'}
                            </span>
                          ) : (
                            <span className="document-timestamp" title={new Date(doc.last_modified_ms).toLocaleString()}>
                              {formatLastModified(doc.last_modified_ms)}
                            </span>
                          )}
                          <span className="document-detail-row">
                            <span>{doc.source_kind.toUpperCase()}</span>
                            <ArtifactBadges doc={doc} />
                          </span>
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            ) : null}

            {node.children.length > 0 ? (
              <ul className="folder-tree-children">
                {[...node.children]
                  .sort((a, b) => {
                    if (sortMetric) {
                      const metricDiff = compareMetricValues(
                        subtreeMetricValue.get(a.path),
                        subtreeMetricValue.get(b.path),
                        sortDirection,
                      )
                      if (metricDiff !== 0) {
                        return metricDiff
                      }
                    }
                    const latestDiff =
                      (subtreeLatestModified.get(b.path) ?? 0) - (subtreeLatestModified.get(a.path) ?? 0)
                    if (latestDiff !== 0) {
                      return latestDiff
                    }
                    return a.name.localeCompare(b.name)
                  })
                  .map((child) => renderNode(child, depth + 1))}
              </ul>
            ) : null}
          </>
        ) : null}
      </li>
    )
  }

  return (
    <div className="folder-tree-panel">
      <h3>Folders & Files ({documents.length})</h3>
      <ul className="folder-tree-root">{renderNode(root, 0)}</ul>
    </div>
  )
}
