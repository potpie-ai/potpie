import { readFileSync } from 'node:fs';

const STABLE_VERSION = /^(0|[1-9]\d*)\.(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?$/;

export function assertReleaseReady({ configPath, pyprojectPath }) {
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  if (config.spokeId !== 'potpie' || config.docsPath !== 'docs') {
    throw new Error('docs/config.json must match the Hub Potpie identity and docs path');
  }
  if (config.contractVersion !== 1 || config.versioning?.enabled !== true) {
    throw new Error('docs/config.json must opt into supported versioning contract 1');
  }
  const pyproject = readFileSync(pyprojectPath, 'utf8');
  const project = /\[project\]([\s\S]*?)(?:\n\[|$)/.exec(pyproject)?.[1] || '';
  const version = /^\s*version\s*=\s*["']([^"']+)["']/m.exec(project)?.[1];
  if (!version || !STABLE_VERSION.test(version)) {
    throw new Error('pyproject.toml [project].version must be MAJOR.MINOR or MAJOR.MINOR.PATCH');
  }
  return version;
}
