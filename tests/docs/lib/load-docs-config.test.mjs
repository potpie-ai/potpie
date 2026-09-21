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
      contractVersion: 1,
      versioning: { enabled: true },
    });
    const cfg = loadDocsConfig(path);
    assert.equal(cfg.spokeId, 'potpie');
    assert.equal(cfg.docsPath, 'docs');
  });

  test('ignores leftover user-facing / exception fields', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: 'docs',
      contractVersion: 1,
      versioning: { enabled: true },
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
    const path = writeConfig({ spokeId: 'Not Valid', contractVersion: 1, versioning: { enabled: true } });
    assert.throws(() => loadDocsConfig(path), /spokeId/);
  });

  test('rejects absolute docsPath', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: '/etc/passwd',
      contractVersion: 1,
      versioning: { enabled: true },
    });
    assert.throws(() => loadDocsConfig(path), /docsPath/);
  });

  test('rejects docsPath with ..', () => {
    const path = writeConfig({
      spokeId: 'potpie',
      docsPath: '../outside',
      contractVersion: 1,
      versioning: { enabled: true },
    });
    assert.throws(() => loadDocsConfig(path), /docsPath/);
  });

  test('rejects docsPath that becomes empty after normalization', () => {
    for (const docsPath of ['/', '///']) {
      const path = writeConfig({
        spokeId: 'potpie',
        docsPath,
        contractVersion: 1,
        versioning: { enabled: true },
      });
      assert.throws(() => loadDocsConfig(path), /docsPath/);
    }
  });

  test('defaults missing docsPath to docs', () => {
    const path = writeConfig({ spokeId: 'potpie', contractVersion: 1, versioning: { enabled: true } });
    const cfg = loadDocsConfig(path);
    assert.equal(cfg.docsPath, 'docs');
  });

  test('rejects unsupported release contract metadata', () => {
    const path = writeConfig({ spokeId: 'potpie', contractVersion: 2, versioning: { enabled: true } });
    assert.throws(() => loadDocsConfig(path), /contractVersion/);
  });
});
