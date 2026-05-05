import type { GroundingItem } from '../types/api'

export interface MarkdownGroundedBlock {
  blockIndex: number
  markdown: string
  plainText: string
  itemId: string | null
  itemIndex: number | null
  itemType: string | null
  matchKind: 'ordered' | 'similarity' | 'unmatched'
}

const HTML_ENTITY_REPLACEMENTS: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&nbsp;': ' ',
}

function decodeHtmlEntities(value: string): string {
  return value.replace(
    /&(amp|lt|gt|quot|#39|nbsp);/g,
    (entity) => HTML_ENTITY_REPLACEMENTS[entity] ?? entity,
  )
}

export function splitMarkdownBlocks(markdown: string): string[] {
  const normalized = markdown
    .replace(/\r\n/g, '\n')
    .replace(/<\/table>/gi, '</table>\n')
    .replace(/<\/(ul|ol|blockquote|pre)>/gi, '</$1>\n')

  const lines = normalized.split('\n')
  const blocks: string[] = []
  let current: string[] = []
  let inHtmlTable = false

  const flush = () => {
    const block = current.join('\n').trim()
    if (block) {
      blocks.push(block)
    }
    current = []
  }

  for (const line of lines) {
    const trimmed = line.trim()

    if (!inHtmlTable && trimmed.length === 0) {
      flush()
      continue
    }

    if (!inHtmlTable && /^#{1,6}\s/.test(trimmed)) {
      flush()
      blocks.push(trimmed)
      continue
    }

    const startsTable = /<table\b/i.test(trimmed)
    const endsTable = /<\/table>/i.test(trimmed)
    if (startsTable) {
      inHtmlTable = true
    }

    current.push(line)

    if (inHtmlTable && endsTable) {
      flush()
      inHtmlTable = false
    }
  }

  flush()
  return blocks
}

export function markdownToComparableText(markdown: string): string {
  return decodeHtmlEntities(markdown)
    .replace(/<[^>]+>/g, ' ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, ' $1 ')
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, ' $1 ')
    .replace(/^\s{0,3}(#{1,6}|>+|-|\*|\+|\d+\.)\s+/gm, '')
    .replace(/\|/g, ' ')
    .replace(/[*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function tokenOverlapScore(left: string, right: string): number {
  const leftTokens = new Set(left.split(/\s+/).filter(Boolean))
  const rightTokens = new Set(right.split(/\s+/).filter(Boolean))
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return 0
  }

  let overlap = 0
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      overlap += 1
    }
  }

  return overlap / Math.max(1, Math.min(leftTokens.size, rightTokens.size))
}

function scoreMatch(blockText: string, itemText: string): number {
  if (!blockText || !itemText) {
    return 0
  }

  if (blockText === itemText) {
    return 1
  }

  const shorter = blockText.length <= itemText.length ? blockText : itemText
  const longer = shorter === blockText ? itemText : blockText
  if (shorter.length >= 24 && longer.includes(shorter)) {
    return 0.96
  }

  const overlapScore = tokenOverlapScore(blockText, itemText)
  const lengthScore = Math.min(blockText.length, itemText.length) / Math.max(blockText.length, itemText.length)
  return overlapScore * 0.8 + lengthScore * 0.2
}

export function groundMarkdownBlocks(markdown: string, items: GroundingItem[]): MarkdownGroundedBlock[] {
  const blocks = splitMarkdownBlocks(markdown)
  const contentItems = items.filter((item) => markdownToComparableText(item.md || item.value || '').length > 0)

  if (blocks.length === 0) {
    return []
  }

  const orderedZip = blocks.length === contentItems.length
  const lookAheadWindow = 8
  let nextItemCursor = 0

  return blocks.map((block, blockIndex) => {
    const plainText = markdownToComparableText(block)
    let matchedItem: GroundingItem | null = null
    let matchKind: MarkdownGroundedBlock['matchKind'] = 'unmatched'

    if (plainText) {
      if (orderedZip && nextItemCursor < contentItems.length) {
        matchedItem = contentItems[nextItemCursor] ?? null
        nextItemCursor += 1
        matchKind = matchedItem ? 'ordered' : 'unmatched'
      } else {
        let bestIndex = -1
        let bestScore = 0

        for (
          let candidateIndex = nextItemCursor;
          candidateIndex < Math.min(contentItems.length, nextItemCursor + lookAheadWindow);
          candidateIndex += 1
        ) {
          const candidate = contentItems[candidateIndex]
          const candidateText = markdownToComparableText(candidate.md || candidate.value || '')
          const score = scoreMatch(plainText, candidateText)
          if (score > bestScore) {
            bestScore = score
            bestIndex = candidateIndex
          }
        }

        if (bestIndex >= 0 && bestScore >= 0.4) {
          matchedItem = contentItems[bestIndex] ?? null
          nextItemCursor = bestIndex + 1
          matchKind = 'similarity'
        }
      }
    }

    return {
      blockIndex,
      markdown: block,
      plainText,
      itemId: matchedItem?.item_id ?? null,
      itemIndex: matchedItem?.item_index ?? null,
      itemType: matchedItem?.type ?? null,
      matchKind,
    }
  })
}
