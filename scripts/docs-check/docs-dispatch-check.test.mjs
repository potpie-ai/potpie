import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { shouldDispatchDocs } from './docs-dispatch-check.mjs';
import { DOCS_CONFIG_REPO_PATH } from './lib/docs-changed.mjs';

function makeInputs(changedPaths, docsPath = 'docs') {
  const root = mkdtempSync(join(tmpdir(), 'docs-dispatch-'));
  const configPath = join(root, 'config/docs.json');
  const changedFilesPath = join(root, 'changed.txt');
  mkdirSync(join(root, 'config'), { recursive: true });
  writeFileSync(configPath, JSON.stringify({ spokeId: 'demo', docsPath }));
  writeFileSync(changedFilesPath, `${changedPaths.join('\n')}\n`);
  return { configPath, changedFilesPath };
}

describe('shouldDispatchDocs', () => {
  test('dispatches for the configured docsPath', () => {
    assert.equal(
      shouldDispatchDocs(makeInputs(['website/docs/index.md'], 'website/docs')),
      true,
    );
  });

  test('dispatches when scripts/docs-check/config.json changes', () => {
    assert.equal(shouldDispatchDocs(makeInputs([DOCS_CONFIG_REPO_PATH], 'website/docs')), true);
  });

  test('skips code-only changes', () => {
    assert.equal(shouldDispatchDocs(makeInputs(['potpie/cli/main.py'], 'website/docs')), false);
  });
});
