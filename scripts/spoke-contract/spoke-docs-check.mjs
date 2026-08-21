#!/usr/bin/env node
/**
 * Spoke PR docs check. Copy this folder into the Spoke; do not call private Hub workflows.
 * Validates the docs tree only when docs/ changed. Does not require docs for code changes.
 * Hub scripts/fetch-spokes.mjs remains the authoritative publish gate.
 */

import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { docsContractChanged, readChangedPaths } from './lib/docs-changed.mjs';
import { loadDocsConfig } from './lib/load-docs-config.mjs';
import { printValidationResult, validateSpokeDocs } from './lib/validate-docs.mjs';

/**
 * @param {{
 *   configPath: string,
 *   changedFilesPath: string,
 *   spokeRoot: string,
 *   log?: (...args: unknown[]) => void,
 *   error?: (...args: unknown[]) => void,
 * }} options
 * @returns {{ ok: boolean, docsChanged: boolean, message?: string }}
 */
export function runSpokeDocsCheck(options) {
  const log = options.log || console.log;
  const error = options.error || console.error;
  const config = loadDocsConfig(resolve(options.configPath));
  const changedPaths = readChangedPaths(options.changedFilesPath);
  const docsTouched = docsContractChanged(changedPaths, config.docsPath);

  log(`spokeId=${config.spokeId}`);
  log(`docsPath=${config.docsPath}`);
  log(`docsChanged=${docsTouched}`);

  if (!docsTouched) {
    log(`No ${config.docsPath}/** or config/docs.json changes; skipping docs tree validation.`);
    return { ok: true, docsChanged: false };
  }

  const docsRoot = resolve(options.spokeRoot, config.docsPath);
  const result = validateSpokeDocs(docsRoot, { spokeId: config.spokeId });
  printValidationResult(result, `${config.spokeId} docs`);
  if (!result.ok) {
    const message = result.errors.join('\n');
    error(message);
    return {
      ok: false,
      docsChanged: true,
      message,
    };
  }

  return { ok: true, docsChanged: true };
}

/** Run the PR check from environment variables. */
function main() {
  try {
    const result = runSpokeDocsCheck({
      configPath: resolve(process.env.DOCS_CONFIG_PATH || 'config/docs.json'),
      changedFilesPath: process.env.CHANGED_FILES_PATH,
      spokeRoot: resolve(process.env.SPOKE_ROOT || '.'),
    });
    if (!result.ok) process.exit(1);
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main();
}
