import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { runSpokeDocsCheck } from './spoke-docs-check.mjs';

const page = `---
title: Home
description: Home
---

## Overview
`;

function makeSpoke(files, config) {
  const root = mkdtempSync(join(tmpdir(), 'spoke-root-'));
  for (const [rel, content] of Object.entries(files)) {
    const full = join(root, rel);
    mkdirSync(join(full, '..'), { recursive: true });
    writeFileSync(full, content);
  }
  const configPath = join(root, 'config/docs.json');
  mkdirSync(join(root, 'config'), { recursive: true });
  writeFileSync(
    configPath,
    JSON.stringify({
      spokeId: 'demo',
      docsPath: 'docs',
      ...config,
    }),
  );
  return { root, configPath };
}

function writeChanged(root, paths) {
  const changedPath = join(root, 'changed.txt');
  writeFileSync(changedPath, paths.join('\n') + '\n');
  return changedPath;
}

const silent = { log() {}, error() {} };

describe('runSpokeDocsCheck', () => {
  test('docs-only change validates the tree and passes', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': page });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['docs/index.md']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, true);
    assert.equal(result.docsChanged, true);
  });

  test('invalid docs tree fails', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': '# no frontmatter\n' });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['docs/index.md']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, false);
    assert.match(result.message, /frontmatter/i);
  });

  test('code-only change does not require docs', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': page, 'src/cli.ts': 'x' });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['src/cli.ts']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, true);
    assert.equal(result.docsChanged, false);
  });

  test('code plus valid docs change passes', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': page, 'src/cli.ts': 'x' });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['src/cli.ts', 'docs/index.md']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, true);
    assert.equal(result.docsChanged, true);
  });

  test('code plus invalid docs still fails', () => {
    const { root, configPath } = makeSpoke({
      'docs/index.md': '# broken\n',
      'src/cli.ts': 'x',
    });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['src/cli.ts', 'docs/index.md']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, false);
    assert.equal(result.docsChanged, true);
  });

  test('unrelated files pass', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': page, 'README.md': 'hi' });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['README.md']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, true);
    assert.equal(result.docsChanged, false);
  });

  test('deleted docs path still validates remaining tree', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': page });
    const result = runSpokeDocsCheck({
      configPath,
      changedFilesPath: writeChanged(root, ['docs/old-guide.md']),
      spokeRoot: root,
      ...silent,
    });
    assert.equal(result.ok, true);
    assert.equal(result.docsChanged, true);
  });

  test('missing changed-files path fails closed', () => {
    const { root, configPath } = makeSpoke({ 'docs/index.md': page });
    assert.throws(
      () =>
        runSpokeDocsCheck({
          configPath,
          changedFilesPath: '',
          spokeRoot: root,
          ...silent,
        }),
      /CHANGED_FILES_PATH/,
    );
  });
});
