import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import {
  isAllowedAbsoluteHubRoute,
  parseFrontmatter,
  splitFrontmatter,
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

const cleanSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><circle cx="4" cy="4" r="3"/></svg>`;

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

  test('rejects YAML aliases', () => {
    const { data, error } = parseFrontmatter(`---
title: Hello
description: &x alias-bomb
also: *x
---
`);
    assert.equal(data, null);
    assert.match(error, /Invalid YAML|alias/i);
  });

  test('splitFrontmatter body excludes the YAML block', () => {
    const split = splitFrontmatter(`---
title: Hello
description: "mentions <div> in copy"
---

## Body
`);
    assert.equal('body' in split, true);
    if ('body' in split) {
      assert.match(split.body, /## Body/);
      assert.equal(split.body.includes('mentions <div>'), false);
      assert.match(split.block, /mentions <div>/);
    }
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

  test('links and empty image syntax inside code are not validated', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

\`\`\`markdown
See [unknown route](/random/), [missing](./nope.md), and ![](./missing.svg).
\`\`\`

Inline code is also ignored: \`[missing](./also-missing.md)\`.
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('link-like frontmatter text is not validated', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: "Example [missing](./nope.md)"
---

## Overview
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('link destinations with balanced parentheses are accepted', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

See [spec](https://example.com/a_(b)).
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('angle-bracket link destinations are accepted without their wrappers', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

See [spec](<https://example.com/a_(b)>).
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('empty alt text fails', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

![](./assets/logo.svg)
`,
      'assets/logo.svg': cleanSvg,
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

  test('javascript, data, and vbscript link schemes fail', () => {
    for (const href of ['javascript:alert(1)', 'data:text/html,hi', 'vbscript:msgbox(1)']) {
      const root = makeDocs({
        'index.md': `---
title: Home
description: Home
---

See [x](${href}).
`,
      });
      const result = validateSpokeDocs(root, { spokeId: 'demo' });
      assert.equal(result.ok, false);
      assert.equal(result.errors.some((e) => e.includes('Disallowed link scheme')), true);
    }
  });

  test('HTML in frontmatter description does not fail raw HTML check', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: "how to write a <div> in Markdown"
---

## Overview
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('HTML in page body fails', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

<div>nope</div>
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Raw HTML')), true);
  });

  test('MDX component tags fail', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

<Tabs>
nope
</Tabs>
`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Raw HTML')), true);
  });

  test('CLI placeholders and mermaid br in fences pass', () => {
    const root = makeDocs({
      'index.md': `---
title: Home
description: Home
---

Run \`potpie pot use <id-or-name>\`.

See potpie graph read --subgraph <s> --view <v>.

\`\`\`mermaid
flowchart TB
  n["Agents<br/>(Claude)"]
\`\`\`

\`\`\`bash
potpie graph commit <plan_id> --verify
\`\`\`
`,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });

  test('unsafe SVG fails without rewriting the file', () => {
    const raw = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <script>alert(1)</script>
  <circle cx="5" cy="5" r="4" fill="blue"/>
</svg>`;
    const root = makeDocs({
      'index.md': validPage,
      'assets/logo.svg': raw,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('Unsafe SVG content')), true);
    assert.equal(readFileSync(join(root, 'assets/logo.svg'), 'utf8'), raw);
  });

  test('rejects SVG with DOCTYPE or ENTITY', () => {
    const root = makeDocs({
      'index.md': validPage,
      'assets/logo.svg': `<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<svg xmlns="http://www.w3.org/2000/svg">&xxe;</svg>`,
    });
    const result = validateSpokeDocs(root, { spokeId: 'demo' });
    assert.equal(result.ok, false);
    assert.equal(result.errors.some((e) => e.includes('DOCTYPE') || e.includes('ENTITY')), true);
  });

  test('clean SVG asset passes', () => {
    const root = makeDocs({
      'index.md': validPage,
      'assets/logo.svg': cleanSvg,
    });
    assert.equal(validateSpokeDocs(root, { spokeId: 'demo' }).ok, true);
  });
});
