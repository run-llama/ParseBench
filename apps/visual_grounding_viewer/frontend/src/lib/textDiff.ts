// LCS-based token diff, matching the legacy HTML report behavior.
//
// Renders a whitespace-token diff of two strings as inline HTML:
//   - tokens present in both → plain text
//   - tokens only in `pred` → <span class="diff-add">
//   - tokens only in `gt`   → <span class="diff-del">
//
// The algorithm matches the older HTML report's client-side JS so reviewers
// switching between the two UIs see the same highlighting.

export type DiffOpType = 'eq' | 'add' | 'del'

export interface DiffOp {
  type: DiffOpType
  token: string
}

export function computeDiffOps(gt: string, pred: string): DiffOp[] {
  const gtTokens = gt.trim().length > 0 ? gt.trim().split(/\s+/) : []
  const predTokens = pred.trim().length > 0 ? pred.trim().split(/\s+/) : []
  const n = gtTokens.length
  const m = predTokens.length

  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0))
  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      if (gtTokens[i - 1] === predTokens[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  const ops: DiffOp[] = []
  let i = n
  let j = m
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && gtTokens[i - 1] === predTokens[j - 1]) {
      ops.push({ type: 'eq', token: gtTokens[i - 1] })
      i -= 1
      j -= 1
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ type: 'add', token: predTokens[j - 1] })
      j -= 1
    } else if (i > 0) {
      ops.push({ type: 'del', token: gtTokens[i - 1] })
      i -= 1
    } else {
      break
    }
  }

  ops.reverse()
  return ops
}

export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      case "'":
        return '&#39;'
      default:
        return char
    }
  })
}

export function computeDiffHtml(gt: string, pred: string): string {
  const ops = computeDiffOps(gt, pred)
  return ops
    .map((op) => {
      const safe = escapeHtml(op.token)
      if (op.type === 'eq') return safe
      if (op.type === 'add') return `<span class="diff-add">${safe}</span>`
      return `<span class="diff-del">${safe}</span>`
    })
    .join(' ')
}
