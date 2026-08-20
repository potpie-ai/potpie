import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import {
  isAllowedAbsoluteHubRoute,
  parseFrontmatter,
  validateSpokeDocs,
} from './validate-docs.mjs';

function makeDocs(files) {
  const root = mkdtempSync(join(tmpdir(), 'spoke-docs-'));
  for (const [rel, content] of Object.entries(files)) {
    const full = join(root, rel);
    mkdirSync(join(full, '..'), { recursive: true });
    if (typeof content === 'string') writeFileSync(full, content);
    else if (content.symlink) symlinkSync(content.symlink, full);
    else if (content.bytes) writeFileSync(full, Buffer.alloc(content.bytes));
  }
  return root;
}

const validPage = `---
title: Hello
description: A valid page
---

## Overview

See [home](./index.md).
`;

describe('parseFrontmatter', () => {
  test('parses quoted values with colons', () => {
    const { data, error } = parseFrontmatter(`---
title: "Install: first run"
description: Setup
---
`);
    assert.equal(error, null);
    assert.equal(data.title, 'Install: first run');
  });

  test('parses multiline block scalars as strings', () => {
    const { data, error } = parseFrontmatter(`---
title: Hello
description: |
  Several
  lines
---
`);
    assert.equal(error, null);
    assert.equal(typeof data.description, 'string');
    assert.match(data.description, /Several/);
  });

  test('rejects invalid YAML', () => {
    const { data, error } = parseFrontmatter(`---
title: [unterminated
---
`);
    assert.equal(data, null);
    assert.match(error, /Invalid YAML/);
  });
});

describe('isAllowedAbsoluteHubRoute', () => {
  test('allows Hub top-level routes', () => {
    assert.equal(isAllowedAbsoluteHubRoute('/architecture/', 'potpie'), true);
    assert.equal(isAllowedAbsoluteHubRoute('/products', 'potpie'), true);
  });

  test('allows own product routes', () => {
    assert.equal(isAllowedAbsoluteHubRoute('/products/potpie/guides/cli/', 'potpie'), true);
  });

  test('rejects other product routes when spokeId is set', () => {
    assert.equal(isAllowedAbsoluteHubRoute('/products/other/guides/', 'potpie'), false);
  });

  test('rejects unknown absolute routes', () => {
    assert.equal(isAllowedAbsoluteHubRoute('/random/', 'potpie'), false);
  });
});

describe('validateSpokeDocs', () => {
  test('missing index.md fails', () => {
    const root = makeDocs({
      'guides/one.md': validPage,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('docs/index.md is mandatory')), true);
  });

  test('valid tree with only index.md passes (relaxed dirs)', () => {
    const root = makeDocs({
      'index.md': validPage,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('missing frontmatter fails', () => {
    const root = makeDocs({
      'index.md': '# No frontmatter\n',
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Missing YAML frontmatter')), true);
  });

  test('empty title fails', () => {
    const root = makeDocs({
      'index.md': `---
title: "  "
description: Real
---

## Hi
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Missing frontmatter title')), true);
  });

  test('literal pipe description is not treated as a real description', () => {
    const root = makeDocs({
      'index.md': `---
title: Hello
description: |
---

## Hi
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(
      result.errors.some((e) => e.includes('Missing frontmatter description')),
      true,
    );
  });

  test('disallowed asset extension fails', () => {
    const root = makeDocs({
      'index.md': validPage,
      'assets/notes.exe': 'nope',
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Unsupported asset type')), true);
  });

  test('oversized asset fails', () => {
    const root = makeDocs({
      'index.md': validPage,
      'assets/huge.png': { bytes: 50 },
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo', maxAssetBytes: 16 });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('byte limit')), true);
  });

  test('kebab-case filename violation fails', () => {
    const root = makeDocs({
      'index.md': validPage,
      'Getting Started.md': validPage,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.toLowerCase().includes('kebab-case')), true);
  });

  test('symlink rejection', () => {
    const root = makeDocs({
      'index.md': validPage,
      'alias.md': { symlink: 'index.md' },
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Symlinks are not allowed')), true);
  });

  test('unknown absolute route fails', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

See [nope](/random/).
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Unknown absolute route')), true);
  });

  test('own Hub product route is allowed', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

See [self](/products/demo/guides/).
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('broken relative link fails', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

See [missing](./nope.md).
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Broken local link')), true);
  });

  test('empty alt text fails', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

![](./assets/logo.svg)
`,
      'assets/logo.svg': '<svg></svg>',
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('alt text')), true);
  });

  test('windows absolute path in a link fails', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

See [disk](C:\\Windows\\notes.md).
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Absolute filesystem paths')), true);
  });
});
