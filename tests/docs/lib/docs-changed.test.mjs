import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import {
  DOCS_CONFIG_REPO_PATH,
  docsChanged,
  docsContractChanged,
  isUnderDocs,
  normalizeRepoPath,
} from './docs-changed.mjs';

describe('docsChanged', () => {
  test('docs-only match', () => {
    assert.equal(docsChanged(['docs/index.md']), true);
  });

  test('code-only match is not docs', () => {
    assert.equal(docsChanged(['src/cli.ts', 'potpie/cli.py']), false);
  });

  test('docs directory itself counts', () => {
    assert.equal(docsChanged(['docs']), true);
  });

  test('strips ./ prefix', () => {
    assert.equal(docsChanged(['./docs/index.md']), true);
  });

  test('normalizes Windows separators', () => {
    assert.equal(docsChanged(['docs\\index.md']), true);
  });

  test('deleted docs path still counts as a docs change', () => {
    assert.equal(docsChanged(['docs/old-guide.md', 'src/cli.ts']), true);
  });

  test('custom docsPath', () => {
    assert.equal(docsChanged(['website/docs/index.md'], 'website/docs'), true);
    assert.equal(docsChanged(['docs/index.md'], 'website/docs'), false);
  });

  test('docs contract changes when docs/config.json changes', () => {
    assert.equal(docsContractChanged([DOCS_CONFIG_REPO_PATH], 'website/docs'), true);
    assert.equal(docsContractChanged(['website/docs/index.md'], 'website/docs'), true);
    assert.equal(docsContractChanged(['src/cli.py'], 'website/docs'), false);
  });

  test('does not treat similarly prefixed paths as docs', () => {
    assert.equal(docsChanged(['docs-extra/readme.md']), false);
    assert.equal(isUnderDocs('documentation/index.md', 'docs'), false);
  });

  test('empty list is not a docs change', () => {
    assert.equal(docsChanged([]), false);
  });

  test('normalizeRepoPath', () => {
    assert.equal(normalizeRepoPath(' ./docs/a.md '), 'docs/a.md');
  });
});
