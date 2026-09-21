import { readFileSync } from 'node:fs';

const KEBAB_CASE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/**
 * @param {string} configPath
 * @returns {{
 *   spokeId: string,
 *   docsPath: string,
 *   contractVersion: number,
 *   versioning: { enabled: true },
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
    throw new Error(`docs check config must be a JSON object: ${configPath}`);
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
    !normalizedDocsPath ||
    normalizedDocsPath.includes('..') ||
    normalizedDocsPath.startsWith('/') ||
    /^[a-zA-Z]:\//.test(normalizedDocsPath)
  ) {
    throw new Error('docsPath must be a relative directory without ..');
  }

  if (raw.contractVersion !== 1) {
    throw new Error('contractVersion must be supported version 1');
  }
  if (!raw.versioning || raw.versioning.enabled !== true) {
    throw new Error('versioning.enabled must be true');
  }

  return {
    spokeId,
    docsPath: normalizedDocsPath,
    contractVersion: raw.contractVersion,
    versioning: { enabled: true },
  };
}
