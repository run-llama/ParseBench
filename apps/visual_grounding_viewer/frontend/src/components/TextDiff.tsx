import { useMemo } from 'react'

import { computeDiffHtml } from '../lib/textDiff'

interface TextDiffProps {
  expected: string
  actual: string
  /** Optional summary label; defaults to "Show normalized text diff". */
  summary?: string
}

/**
 * Collapsible LCS token diff for an extract_field rule's GT expected value
 * vs the matched predicted text. Shared tokens render plain; pred-only
 * tokens are highlighted green (`.diff-add`), GT-only tokens red-strike
 * (`.diff-del`). Matches the legacy HTML report behavior.
 */
export function TextDiff({ expected, actual, summary = 'Show normalized text diff' }: TextDiffProps) {
  const html = useMemo(() => computeDiffHtml(expected, actual), [expected, actual])

  return (
    <details className="text-diff">
      <summary>{summary}</summary>
      <div className="text-diff-body" dangerouslySetInnerHTML={{ __html: html }} />
    </details>
  )
}
