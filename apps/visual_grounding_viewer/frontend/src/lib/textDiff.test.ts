import { describe, expect, it } from 'vitest'

import { computeDiffHtml, computeDiffOps, escapeHtml } from './textDiff'

describe('computeDiffOps', () => {
  it('returns no ops for empty inputs', () => {
    expect(computeDiffOps('', '')).toEqual([])
  })

  it('returns only eq ops when strings match token-for-token', () => {
    const ops = computeDiffOps('alpha beta gamma', 'alpha beta gamma')
    expect(ops.every((op) => op.type === 'eq')).toBe(true)
    expect(ops.map((op) => op.token)).toEqual(['alpha', 'beta', 'gamma'])
  })

  it('flags pred-only tokens as add', () => {
    const ops = computeDiffOps('alpha', 'alpha beta')
    expect(ops).toContainEqual({ type: 'eq', token: 'alpha' })
    expect(ops).toContainEqual({ type: 'add', token: 'beta' })
  })

  it('flags gt-only tokens as del', () => {
    const ops = computeDiffOps('alpha beta', 'alpha')
    expect(ops).toContainEqual({ type: 'eq', token: 'alpha' })
    expect(ops).toContainEqual({ type: 'del', token: 'beta' })
  })

  it('flags entirely disjoint inputs as all-add + all-del', () => {
    const ops = computeDiffOps('foo bar', 'baz qux')
    // No `eq` ops — the two strings share no tokens.
    expect(ops.every((op) => op.type !== 'eq')).toBe(true)
    expect(ops.filter((op) => op.type === 'del').map((op) => op.token)).toEqual(['foo', 'bar'])
    expect(ops.filter((op) => op.type === 'add').map((op) => op.token)).toEqual(['baz', 'qux'])
  })

  it('preserves shared prefix and suffix as plain eq', () => {
    const ops = computeDiffOps('Big Alpha Token', 'Alpha Token')
    // Alpha and Token are shared, so they appear as eq ops.
    expect(ops.filter((op) => op.type === 'eq').map((op) => op.token)).toEqual(['Alpha', 'Token'])
    expect(ops.filter((op) => op.type === 'del').map((op) => op.token)).toEqual(['Big'])
    expect(ops.filter((op) => op.type === 'add')).toEqual([])
  })

  it('splits on runs of whitespace (tabs, multiple spaces)', () => {
    const ops = computeDiffOps('alpha\t  beta', 'alpha beta')
    expect(ops.filter((op) => op.type === 'eq').map((op) => op.token)).toEqual(['alpha', 'beta'])
  })
})

describe('computeDiffHtml', () => {
  it('wraps add/del tokens in span classes and leaves eq tokens bare', () => {
    const html = computeDiffHtml('alpha beta', 'alpha gamma')
    expect(html).toContain('alpha')
    expect(html).toContain('<span class="diff-del">beta</span>')
    expect(html).toContain('<span class="diff-add">gamma</span>')
  })

  it('produces no span wrappers for identical strings', () => {
    const html = computeDiffHtml('alpha beta', 'alpha beta')
    expect(html).not.toContain('diff-del')
    expect(html).not.toContain('diff-add')
  })

  it('returns an empty string for empty inputs', () => {
    expect(computeDiffHtml('', '')).toBe('')
  })

  it('escapes HTML-unsafe characters in tokens', () => {
    const html = computeDiffHtml('<script>', 'foo')
    expect(html).toContain('&lt;script&gt;')
    expect(html).not.toContain('<script>')
  })
})

describe('escapeHtml', () => {
  it('escapes &, <, >, ", and \'', () => {
    expect(escapeHtml('&<>"\'')).toBe('&amp;&lt;&gt;&quot;&#39;')
  })
})
