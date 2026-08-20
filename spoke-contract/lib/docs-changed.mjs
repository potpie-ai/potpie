/** Git paths are forward-slash; still normalize ./ and Windows separators. */
export function normalizeRepoPath(filePath) {
  return String(filePath || '')
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\//, '');
}

export function isUnderDocs(filePath, docsPath) {
  const root = normalizeRepoPath(docsPath || 'docs').replace(/\/+$/, '');
  return filePath === root || filePath.startsWith(`${root}/`);
}

/**
 * @param {string[]} changedPaths
 * @param {string} [docsPath]
 * @returns {boolean}
 */
export function docsChanged(changedPaths, docsPath = 'docs') {
  return (changedPaths || []).some((rawPath) => {
    const filePath = normalizeRepoPath(rawPath);
    return Boolean(filePath) && isUnderDocs(filePath, docsPath);
  });
}
