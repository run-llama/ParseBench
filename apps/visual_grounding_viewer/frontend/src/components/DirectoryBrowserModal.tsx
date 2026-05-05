import { useEffect, useState } from 'react'

import { browseDirectory } from '../api/client'
import { formatLastModified } from '../lib/time'
import type { BrowseItem } from '../types/api'

interface DirectoryBrowserModalProps {
  open: boolean
  title: string
  initialPath: string
  onClose: () => void
  onSelect: (path: string) => void
}

export function DirectoryBrowserModal({
  open,
  title,
  initialPath,
  onClose,
  onSelect,
}: DirectoryBrowserModalProps) {
  const [currentPath, setCurrentPath] = useState('')
  const [parentPath, setParentPath] = useState<string | null>(null)
  const [items, setItems] = useState<BrowseItem[]>([])
  const [pathInput, setPathInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDirectory = async (path?: string) => {
    setLoading(true)
    setError(null)
    try {
      const response = await browseDirectory(path)
      setCurrentPath(response.current)
      setParentPath(response.parent)
      setItems(response.items)
      setPathInput(response.current)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!open) {
      return
    }
    void loadDirectory(initialPath || undefined)
  }, [open, initialPath])

  if (!open) {
    return null
  }

  const onConfirm = () => {
    const selected = pathInput.trim()
    if (!selected) {
      setError('Select or enter a directory path.')
      return
    }
    onSelect(selected)
    onClose()
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <h3>{title}</h3>
          <button onClick={onClose} className="modal-close" aria-label="Close directory browser">
            Close
          </button>
        </header>

        <div className="modal-controls">
          <button onClick={() => (parentPath ? void loadDirectory(parentPath) : null)} disabled={!parentPath}>
            Up
          </button>
          <input
            type="text"
            value={pathInput}
            onChange={(event) => setPathInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                void loadDirectory(pathInput)
              }
            }}
          />
          <button onClick={() => void loadDirectory(pathInput)}>Go</button>
        </div>

        {error ? <div className="error-box">{error}</div> : null}

        <div className="modal-body">
          {loading ? <p className="muted">Loading directories…</p> : null}
          {!loading && items.length === 0 ? <p className="muted">No subdirectories found.</p> : null}
          {!loading && items.length > 0 ? (
            <ul className="browse-list">
              {items.map((item) => (
                <li key={item.path}>
                  <button
                    className={pathInput === item.path ? 'browse-item selected' : 'browse-item'}
                    onClick={() => setPathInput(item.path)}
                    onDoubleClick={() => void loadDirectory(item.path)}
                  >
                    <span className="browse-item-name">{item.name}</span>
                    <span className="document-meta">
                      <span className="document-timestamp" title={new Date(item.last_modified_ms).toLocaleString()}>
                        {formatLastModified(item.last_modified_ms)}
                      </span>
                      <span className="document-detail-row">DIR</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        <footer className="modal-footer">
          <span className="muted modal-current-path" title={currentPath}>
            {currentPath}
          </span>
          <div className="modal-actions">
            <button onClick={onClose}>Cancel</button>
            <button onClick={onConfirm}>Select</button>
          </div>
        </footer>
      </div>
    </div>
  )
}
