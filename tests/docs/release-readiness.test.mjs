import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { assertReleaseReady } from './lib/release-readiness.mjs';

function fixture({ version = '2.0.2', config = {} } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'release-readiness-'));
  mkdirSync(join(root, 'docs'));
  const configPath = join(root, 'docs/config.json');
  const pyprojectPath = join(root, 'pyproject.toml');
  writeFileSync(configPath, JSON.stringify({
    spokeId: 'potpie', docsPath: 'docs', documentationContractVersion: 1, ...config,
  }));
  writeFileSync(pyprojectPath, `[project]\nversion = "${version}"\n`);
  return { configPath, pyprojectPath };
}

describe('release readiness', () => {
  test('accepts a matching stable tag and reports its docs track', () => {
    const paths = fixture();
    assert.deepEqual(assertReleaseReady({ ...paths, tag: 'v2.0.2' }), { version: '2.0.2', track: '2.0' });
  });

  test('rejects mismatched, incomplete, prerelease, and invalid contract releases', () => {
    assert.throws(() => assertReleaseReady({ ...fixture(), tag: 'v2.0.3' }), /must equal/);
    assert.throws(() => assertReleaseReady({ ...fixture({ version: '2.0' }), tag: 'v2.0' }), /MAJOR.MINOR.PATCH/);
    assert.throws(() => assertReleaseReady({ ...fixture({ version: '2.0.2-rc.1' }), tag: 'v2.0.2-rc.1' }), /MAJOR.MINOR.PATCH/);
    assert.throws(() => assertReleaseReady({ ...fixture({ config: { documentationContractVersion: 2 } }), tag: 'v2.0.2' }), /contract version/);
  });
});
