import { readFileSync } from 'node:fs';

export const DOCS_CONFIG_REPO_PATH = 'docs/config.json';

/**
 * Normalize a changed repository path for comparison.
 * @param {string} filePath
 * @returns {string}
 */
export function normalizeRepoPath(filePath) {
  return String(filePath || '')
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\//, '');
}

/**
 * Return whether a repository path is the docs root or one of its descendants.
 * @param {string} filePath
 * @param {string} docsPath
 * @returns {boolean}
 */
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

/**
 * Treat the Spoke configuration as part of the documentation contract.
 * @param {string[]} changedPaths
 * @param {string} [docsPath]
 * @param {string} [configRepoPath]
 * @returns {boolean}
 */
export function docsContractChanged(
  changedPaths,
  docsPath = 'docs',
  configRepoPath = DOCS_CONFIG_REPO_PATH,
) {
  const normalizedConfigPath = normalizeRepoPath(configRepoPath);
  return (
    docsChanged(changedPaths, docsPath) ||
    (changedPaths || []).some(
      (changedPath) => normalizeRepoPath(changedPath) === normalizedConfigPath,
    )
  );
}

/**
 * Read one repository-relative changed path per line.
 * @param {string} filePath
 * @returns {string[]}
 */
export function readChangedPaths(filePath) {
  if (!filePath) {
    throw new Error('CHANGED_FILES_PATH is required (one changed path per line)');
  }
  return readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}
