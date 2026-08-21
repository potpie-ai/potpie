---
title: Documentation contract
description: Writing and structure rules for this Spoke during the Potpie docs MVP.
---

## Language and voice

Write in clear, simple, user-focused language. Use active voice and present tense. Use sentence case for headings.

## Page structure

Do not repeat the frontmatter title as another H1. Start page content with H2 sections. Every page requires `title` and `description` frontmatter.

## Links and media

Use relative links inside the same product. Use permanent Hub routes for cross-product links. Add meaningful alt text to images. Keep assets under `docs/assets/`.

## Procedures

Explain prerequisites before procedures. Keep commands and code examples complete and runnable. Mention expected results after important commands.

## What not to use

Do not use raw HTML, unsupported components, or Mintlify-specific MDX. Documentation pages use Markdown (`.md`) only during the MVP. Update examples when related APIs or commands change.

## Required layout

```text
docs/
├── index.md
├── getting-started/
├── guides/
├── reference/
└── assets/
```

Page and directory names use lowercase kebab-case. This Spoke publishes to `/products/potpie/`.
