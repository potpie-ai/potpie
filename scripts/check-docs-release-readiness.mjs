#!/usr/bin/env node
import { assertReleaseReady } from '../tests/docs/lib/release-readiness.mjs';

try {
  const tagIndex = process.argv.indexOf('--tag');
  const tag = tagIndex === -1 ? undefined : process.argv[tagIndex + 1];
  const release = assertReleaseReady({ configPath: 'docs/config.json', pyprojectPath: 'pyproject.toml', tag });
  console.log(`Documentation release readiness passed for ${release.version}; docs track: ${release.track}`);
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
