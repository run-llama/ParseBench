import type { FolderNode, VisualizableDocument } from '../types/api'

export function documentMatchesFolder(doc: VisualizableDocument, folderPath: string): boolean {
  if (folderPath === '.') {
    return true
  }

  if (doc.relative_dir === folderPath) {
    return true
  }

  return doc.relative_dir.startsWith(`${folderPath}/`)
}

export function filterDocuments(
  docs: VisualizableDocument[],
  folderPath: string,
  query: string,
): VisualizableDocument[] {
  const normalizedQuery = query.trim().toLowerCase()

  return docs.filter((doc) => {
    if (!documentMatchesFolder(doc, folderPath)) {
      return false
    }

    if (!normalizedQuery) {
      return true
    }

    const haystack = `${doc.base_name} ${doc.relative_dir}`.toLowerCase()
    return haystack.includes(normalizedQuery)
  })
}

export function flattenTree(root: FolderNode): FolderNode[] {
  const out: FolderNode[] = []

  function walk(node: FolderNode) {
    out.push(node)
    for (const child of node.children) {
      walk(child)
    }
  }

  walk(root)
  return out
}
