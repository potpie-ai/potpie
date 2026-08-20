import { readFileSync } from 'node:fs';

const KEBAB_CASE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/**
 * @param {string} configPath
 * @returns {{
 *   spokeId: string,
 *   docsPath: string,
 * }}
 */
export function loadDocsConfig(configPath) {
  let raw;
  try {
    raw = JSON.parse(readFileSync(configPath, 'utf8'));
  } catch (err) {
    throw new Error(`Unable to read Spoke docs config ${configPath}: ${err.message}`);
  }

  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('config/docs.json must be a JSON object');
  }

  const spokeId = raw.spokeId;
  if (typeof spokeId !== 'string' || !KEBAB_CASE.test(spokeId)) {
    throw new Error('spokeId must be lowercase kebab-case');
  }

  const docsPath = raw.docsPath ?? 'docs';
  if (typeof docsPath !== 'string' || !docsPath.trim()) {
    throw new Error('docsPath must be a relative directory without ..');
  }
  const normalizedDocsPath = docsPath.trim().replace(/\\/g, '/').replace(/\/+$/, '');
  if (
    normalizedDocsPath.includes('..') ||
    normalizedDocsPath.startsWith('/') ||
    /^[a-zA-Z]:\//.test(normalizedDocsPath)
  ) {
    throw new Error('docsPath must be a relative directory without ..');
  }

  return {
    spokeId,
    docsPath: normalizedDocsPath,
  };
}
