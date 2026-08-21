#!/usr/bin/env node
/**
 * Print whether a trusted push changed the configured docs tree or its config.
 * This keeps GitHub's static push trigger independent from a configurable docsPath.
 */

import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { docsContractChanged, readChangedPaths } from './lib/docs-changed.mjs';
import { loadDocsConfig } from './lib/load-docs-config.mjs';

/**
 * @param {{ configPath: string, changedFilesPath: string }} options
 * @returns {boolean}
 */
export function shouldDispatchDocs(options) {
  const config = loadDocsConfig(resolve(options.configPath));
  const changedPaths = readChangedPaths(options.changedFilesPath);
  return docsContractChanged(changedPaths, config.docsPath);
}

/** Print a GitHub-output-friendly boolean for the current push. */
function main() {
  const shouldDispatch = shouldDispatchDocs({
    configPath: resolve(process.env.DOCS_CONFIG_PATH || 'config/docs.json'),
    changedFilesPath: process.env.CHANGED_FILES_PATH,
  });
  process.stdout.write(shouldDispatch ? 'true' : 'false');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main();
}
