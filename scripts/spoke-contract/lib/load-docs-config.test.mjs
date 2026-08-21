import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { loadDocsConfig } from './load-docs-config.mjs';

function writeConfig(obj) {
  const dir = mkdtempSync(join(tmpdir(), 'docs-cfg-'));
  const path = join(dir, 'docs.json');
  writeFileSync(path, JSON.stringify(obj));
  return path;
}

describe('loadDocsConfig', () => {
  test('loads a valid config', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: 'docs',
    });
    const cfg = loadDocsConfig(path);
    assert.equal(cfg.spokeId, 'potpie');
    assert.equal(cfg.docsPath, 'docs');
  });

  test('ignores leftover user-facing / exception fields', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: 'docs',
      userFacingPaths: ['src/**'],
      excludedPaths: ['**/*.test.*'],
      docsNotRequiredLabel: 'docs-not-required',
    });
    const cfg = loadDocsConfig(path);
    assert.equal(cfg.spokeId, 'potpie');
    assert.equal(cfg.docsPath, 'docs');
    assert.equal('userFacingPaths' in cfg, false);
  });

  test('rejects invalid spokeId', () => {
    const path = writeConfig({ spokeId: 'Not Valid' });
    assert.throws(() => loadDocsConfig(path), /spokeId/);
  });

  test('rejects absolute docsPath', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: '/etc/passwd',
    });
    assert.throws(() => loadDocsConfig(path), /docsPath/);
  });

  test('rejects docsPath with ..', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: '../outside',
    });
    assert.throws(() => loadDocsConfig(path), /docsPath/);
  });

  test('rejects docsPath that becomes empty after normalization', () => {
    for (const docsPath of ['/', '///']) {
      const path = writeConfig({
        spokeId: 'potpie',
        docsPath,
      });
      assert.throws(() => loadDocsConfig(path), /docsPath/);
    }
  });

  test('defaults missing docsPath to docs', () => {
    const path = writeConfig({ spokeId: 'potpie' });
    const cfg = loadDocsConfig(path);
    assert.equal(cfg.docsPath, 'docs');
  });
});
