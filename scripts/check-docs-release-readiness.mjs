#!/usr/bin/env node
import { assertReleaseReady } from '../tests/docs/lib/release-readiness.mjs';

try {
  const version = assertReleaseReady({ configPath: 'docs/config.json', pyprojectPath: 'pyproject.toml' });
  console.log(`Documentation release readiness passed for ${version}`);
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
