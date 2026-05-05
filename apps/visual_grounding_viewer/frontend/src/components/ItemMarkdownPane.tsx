import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'

import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'

import type { OverlayLayerVisibility } from '../lib/grounding'
import {
  buildItemInteractionData,
  caretTextOffsetFromPoint,
  matchUnitsToTextContent,
  unitsForMode,
  type MatchedTextUnit,
} from '../lib/itemGranularPreview'
import type { GroundingGranularUnit, GroundingItem } from '../types/api'

interface ItemMarkdownPaneProps {
  items: GroundingItem[]
  visibleLayers: OverlayLayerVisibility
  activeItemId: string | null
  hoveredItemId: string | null
  activeGranularPreview: GroundingGranularUnit | null
  hoveredGranularPreview: GroundingGranularUnit | null
  hoverSource: 'viewer' | 'sidebar' | null
  onHoverItem: (itemId: string | null) => void
  onSelectItem: (itemId: string) => void
  onHoverGranularPreview: (unit: GroundingGranularUnit | null) => void
  onSelectGranularPreview: (unit: GroundingGranularUnit | null) => void
}

function InteractiveMarkdownContent({
  item,
  visibleLayers,
  activeGranularPreview,
  hoveredGranularPreview,
  onHoverItem,
  onHoverGranularPreview,
  onSelectGranularPreview,
}: {
  item: GroundingItem
  visibleLayers: OverlayLayerVisibility
  activeGranularPreview: GroundingGranularUnit | null
  hoveredGranularPreview: GroundingGranularUnit | null
  onHoverItem: (itemId: string | null) => void
  onHoverGranularPreview: (unit: GroundingGranularUnit | null) => void
  onSelectGranularPreview: (unit: GroundingGranularUnit | null) => void
}) {
  const contentRef = useRef<HTMLDivElement | null>(null)
  const lastHoveredUnitIdRef = useRef<string | null>(null)
  const interaction = useMemo(() => buildItemInteractionData(item, visibleLayers), [item, visibleLayers])
  const interactionUnits = useMemo(() => unitsForMode(interaction), [interaction])
  const [matchedUnits, setMatchedUnits] = useState<MatchedTextUnit[]>([])

  useEffect(() => {
    lastHoveredUnitIdRef.current = null
    if (!contentRef.current || (interaction.mode !== 'line' && interaction.mode !== 'word')) {
      const frameId = window.requestAnimationFrame(() => setMatchedUnits([]))
      return () => window.cancelAnimationFrame(frameId)
    }

    const root = contentRef.current
    const frameId = window.requestAnimationFrame(() => {
      const textContent = root.textContent ?? ''
      setMatchedUnits(matchUnitsToTextContent(textContent, interactionUnits))
    })

    return () => {
      window.cancelAnimationFrame(frameId)
    }
  }, [interaction.mode, interactionUnits, item.item_id, item.md])

  const handleTextMouseMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (interaction.mode !== 'line' && interaction.mode !== 'word') {
      return
    }

    const root = contentRef.current
    if (!root) {
      return
    }

    const offset = caretTextOffsetFromPoint(root, event.clientX, event.clientY)
    if (offset === null) {
      if (lastHoveredUnitIdRef.current !== null) {
        lastHoveredUnitIdRef.current = null
        onHoverGranularPreview(null)
      }
      return
    }

    const nextMatch = matchedUnits.find((entry) => offset >= entry.start && offset < entry.end) ?? null
    const nextUnitId = nextMatch?.unit.unit_id ?? null
    if (nextUnitId === lastHoveredUnitIdRef.current) {
      return
    }

    lastHoveredUnitIdRef.current = nextUnitId
    onHoverItem(null)
    onHoverGranularPreview(nextMatch?.unit ?? null)
  }

  const handleTextMouseLeave = () => {
    lastHoveredUnitIdRef.current = null
    onHoverGranularPreview(null)
  }

  const handleTextClick = () => {
    if (interaction.mode !== 'line' && interaction.mode !== 'word') {
      return
    }
    const hoveredUnit = matchedUnits.find((entry) => entry.unit.unit_id === lastHoveredUnitIdRef.current)?.unit ?? null
    onSelectGranularPreview(hoveredUnit)
  }

  const renderedMarkdown = item.md || item.value || ''
  const cellUnitsByPosition = useMemo(() => {
    const map = new Map<string, GroundingGranularUnit>()
    for (const unit of interaction.cellUnits) {
      if (unit.row_index === null || unit.column_index === null) {
        continue
      }
      map.set(`${unit.row_index}:${unit.column_index}`, unit)
    }
    return map
  }, [interaction.cellUnits])

  const cellUnitsById = useMemo(() => {
    const map = new Map<string, GroundingGranularUnit>()
    for (const unit of interaction.cellUnits) {
      map.set(unit.unit_id, unit)
    }
    return map
  }, [interaction.cellUnits])

  useEffect(() => {
    if (interaction.mode !== 'cell' || !contentRef.current) {
      return
    }

    const rows = Array.from(contentRef.current.querySelectorAll('tr'))
    for (const [rowIndex, row] of rows.entries()) {
      const cells = Array.from(row.children).filter(
        (cell): cell is HTMLTableCellElement => cell instanceof HTMLTableCellElement,
      )
      for (const [columnIndex, cell] of cells.entries()) {
        const unit = cellUnitsByPosition.get(`${rowIndex}:${columnIndex}`) ?? null
        if (unit) {
          cell.dataset.previewUnitId = unit.unit_id
        } else {
          delete cell.dataset.previewUnitId
        }
      }
    }
  }, [cellUnitsByPosition, interaction.mode, renderedMarkdown])

  useEffect(() => {
    if (interaction.mode !== 'cell' || !contentRef.current) {
      return
    }

    const cells = Array.from(contentRef.current.querySelectorAll('[data-preview-unit-id]'))
    for (const cell of cells) {
      const element = cell as HTMLElement
      const unitId = element.dataset.previewUnitId ?? null
      const isActive = unitId !== null && activeGranularPreview?.unit_id === unitId
      const isHovered = unitId !== null && hoveredGranularPreview?.unit_id === unitId
      element.classList.toggle('markdown-cell-hover-target', true)
      element.classList.toggle('active', isActive)
      element.classList.toggle('hovered', isHovered)
    }
  }, [activeGranularPreview?.unit_id, hoveredGranularPreview?.unit_id, interaction.mode])

  if (interaction.mode === 'cell' && interaction.cellUnits.length > 0) {
    return (
      <div
        className="markdown-content interactive-markdown"
        ref={contentRef}
        onMouseMove={(event) => {
          const cell = (event.target as HTMLElement | null)?.closest('[data-preview-unit-id]') as HTMLElement | null
          const unitId = cell?.dataset.previewUnitId ?? null
          const unit = unitId ? cellUnitsById.get(unitId) ?? null : null
          if (lastHoveredUnitIdRef.current === unitId) {
            return
          }
          lastHoveredUnitIdRef.current = unitId
          onHoverItem(null)
          onHoverGranularPreview(unit)
        }}
        onMouseLeave={() => {
          lastHoveredUnitIdRef.current = null
          onHoverGranularPreview(null)
        }}
        onClick={(event) => {
          const cell = (event.target as HTMLElement | null)?.closest('[data-preview-unit-id]') as HTMLElement | null
          const unitId = cell?.dataset.previewUnitId ?? null
          onSelectGranularPreview(unitId ? cellUnitsById.get(unitId) ?? null : null)
        }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, rehypeSanitize]}>
          {renderedMarkdown}
        </ReactMarkdown>
      </div>
    )
  }

  return (
    <div
      className={[
        'markdown-content',
        interaction.mode === 'line' || interaction.mode === 'word' ? 'interactive-markdown interactive-text' : '',
        interaction.mode ? `interaction-${interaction.mode}` : '',
      ]
        .filter(Boolean)
        .join(' ')}
      ref={contentRef}
      onMouseMove={interaction.mode === 'line' || interaction.mode === 'word' ? handleTextMouseMove : undefined}
      onMouseLeave={interaction.mode === 'line' || interaction.mode === 'word' ? handleTextMouseLeave : undefined}
      onClick={interaction.mode === 'line' || interaction.mode === 'word' ? handleTextClick : undefined}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        >
          {renderedMarkdown}
      </ReactMarkdown>
    </div>
  )
}

export function ItemMarkdownPane({
  items,
  visibleLayers,
  activeItemId,
  hoveredItemId,
  activeGranularPreview,
  hoveredGranularPreview,
  hoverSource,
  onHoverItem,
  onSelectItem,
  onHoverGranularPreview,
  onSelectGranularPreview,
}: ItemMarkdownPaneProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
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
    <div className="markdown-pane" ref={containerRef}>
      {items.map((item) => {
        const isActive = activeItemId === item.item_id
        const isHovered = hoveredItemId === item.item_id
        const isViewerHovered = hoverSource === 'viewer' && isHovered
        const interaction = buildItemInteractionData(item, visibleLayers)
        const className = [
          'markdown-segment',
          interaction.mode ? `interaction-card-${interaction.mode}` : '',
          isActive ? 'active' : '',
          isHovered ? 'hovered' : '',
          isViewerHovered ? 'viewer-hovered' : '',
        ]
          .filter(Boolean)
          .join(' ')

        return (
          <article
            key={item.item_id}
            className={className}
            onMouseEnter={() => {
              if (!interaction.mode) {
                onHoverItem(item.item_id)
              }
            }}
            onMouseLeave={() => {
              onHoverItem(null)
              onHoverGranularPreview(null)
            }}
            onClick={() => {
              if (!interaction.mode) {
                onSelectItem(item.item_id)
              }
            }}
            data-item-id={item.item_id}
            data-item-index={item.item_index}
          >
            <header>
              <div className="markdown-segment-title">
                <span className="segment-order">ro:{item.item_index}</span>
              </div>
              <div className="markdown-segment-meta">
                <span className="segment-type">{item.type}</span>
                <span>bbox:{item.bboxes.length}</span>
                {interaction.mode ? <span>hover:{interaction.mode}</span> : null}
              </div>
            </header>
            <InteractiveMarkdownContent
              item={item}
              visibleLayers={visibleLayers}
              activeGranularPreview={activeGranularPreview}
              hoveredGranularPreview={hoveredGranularPreview}
              onHoverItem={onHoverItem}
              onHoverGranularPreview={onHoverGranularPreview}
              onSelectGranularPreview={onSelectGranularPreview}
            />
          </article>
        )
      })}
    </div>
  )
}
