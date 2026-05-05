import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'

import { useEffect, useMemo, useRef } from 'react'

import { groundMarkdownBlocks } from '../lib/markdownGrounding'
import type { GroundingItem } from '../types/api'

interface MarkdownPaneProps {
  markdown: string | null
  pageLabel: string
  markdownSource: string | null
  items: GroundingItem[]
  activeItemId: string | null
  hoveredItemId: string | null
  hoverSource: 'viewer' | 'sidebar' | null
  onCollapse: () => void
  onHoverItem: (itemId: string | null) => void
  onSelectItem: (itemId: string) => void
}

export function MarkdownPane({
  markdown,
  pageLabel,
  markdownSource,
  items,
  activeItemId,
  hoveredItemId,
  hoverSource,
  onCollapse,
  onHoverItem,
  onSelectItem,
}: MarkdownPaneProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const blocks = useMemo(() => groundMarkdownBlocks(markdown ?? '', items), [items, markdown])
  const targetItemId = hoveredItemId ?? activeItemId

  useEffect(() => {
    if (!targetItemId) {
      return
    }

    const target = containerRef.current?.querySelector(
      `article[data-item-id="${targetItemId}"]`,
    ) as HTMLElement | null
    if (!target) {
      return
    }

    target.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [hoverSource, targetItemId])

  return (
    <section className="markdown-preview-panel">
      <header className="panel-header markdown-preview-header">
        <div>
          <h3>Final Markdown Preview</h3>
          <span>
            {pageLabel}
            {markdownSource ? ` · ${markdownSource}` : ''}
          </span>
        </div>
        <button
          className="panel-collapse-button"
          onClick={onCollapse}
          aria-label="Collapse markdown preview"
        >
          →
        </button>
      </header>

      {!markdown ? (
        <div className="markdown-pane muted" ref={containerRef}>
          No markdown artifact found for this document.
        </div>
      ) : (
        <div className="markdown-pane" ref={containerRef}>
          {blocks.map((block) => {
            const isActive = activeItemId === block.itemId
            const isHovered = hoveredItemId === block.itemId
            const isViewerHovered = hoverSource === 'viewer' && isHovered
            const className = [
              'markdown-segment',
              'markdown-preview-segment',
              isActive ? 'active' : '',
              isHovered ? 'hovered' : '',
              isViewerHovered ? 'viewer-hovered' : '',
              block.itemId ? '' : 'ungrounded',
            ]
              .filter(Boolean)
              .join(' ')

            return (
              <article
                key={`${block.blockIndex}:${block.itemId ?? 'unmatched'}`}
                className={className}
                onMouseEnter={() => onHoverItem(block.itemId)}
                onMouseLeave={() => onHoverItem(null)}
                onClick={() => {
                  if (block.itemId) {
                    onSelectItem(block.itemId)
                  }
                }}
                data-item-id={block.itemId ?? undefined}
                data-item-index={block.itemIndex ?? undefined}
              >
                <header>
                  <div className="markdown-segment-title">
                    <span className="segment-order">#{block.blockIndex + 1}</span>
                    {block.itemIndex !== null ? <span className="segment-order">ro:{block.itemIndex}</span> : null}
                  </div>
                  <div className="markdown-segment-meta">
                    {block.itemType ? <span className="segment-type">{block.itemType}</span> : null}
                    <span>{block.itemId ? block.matchKind : 'unmatched'}</span>
                  </div>
                </header>
                <div className="markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, rehypeSanitize]}>
                    {block.markdown}
                  </ReactMarkdown>
                </div>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
