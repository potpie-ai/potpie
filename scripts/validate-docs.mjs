#!/usr/bin/env bun
/**
 * Shared Spoke docs validation.
 * Rejects unsupported files, symlinks, path traversal, MDX, and missing frontmatter.
 */

import { existsSync, lstatSync, readdirSync, readFileSync, realpathSync } from 'node:fs';
import { join, relative, resolve, sep, extname, basename } from 'node:path';

const REQUIRED_DIRS = ['getting-started', 'guides', 'reference', 'assets'];
const ALLOWED_ASSET_EXTS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.svg',
  '.pdf',
  '.mp4',
  '.webm',
]);
const KEBAB_CASE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function fail(errors, message) {
  errors.push(message);
}

function isKebabName(name, { allowDotExt = false } = {}) {
  if (allowDotExt) {
    const base = name.replace(/\.[^.]+$/, '');
    const ext = extname(name).toLowerCase();
    if (name.endsWith('.md')) return KEBAB_CASE.test(base);
    return KEBAB_CASE.test(base) && ALLOWED_ASSET_EXTS.has(ext);
  }
  return KEBAB_CASE.test(name);
}

function parseFrontmatter(content) {
  if (!content.startsWith('---')) return null;
  const end = content.indexOf('\n---', 3);
  if (end === -1) return null;
  const block = content.slice(4, end);
  const data = {};
  for (const line of block.split('\n')) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    data[match[1]] = value;
  }
  return data;
}

function walk(dir, files = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isSymbolicLink() || lstatSync(full).isSymbolicLink()) {
      files.push({ path: full, symlink: true, dirent: entry });
      continue;
    }
    if (entry.isDirectory()) walk(full, files);
    else files.push({ path: full, symlink: false, dirent: entry });
  }
  return files;
}

function extractMarkdownLinks(content) {
  const links = [];
  const re = /!\[[^\]]*\]\(([^)]+)\)|\[[^\]]*\]\(([^)]+)\)/g;
  let match;
  while ((match = re.exec(content)) !== null) {
    links.push(match[1] || match[2]);
  }
  return links;
}

/**
 * @param {string} docsRoot absolute path to Spoke docs/ directory
 * @param {{ spokeId?: string }} [options]
 * @returns {{ ok: boolean, errors: string[], warnings: string[] }}
 */
export function validateSpokeDocs(docsRoot, options = {}) {
  const errors = [];
  const warnings = [];
  const root = resolve(docsRoot);

  if (!existsSync(root)) {
    fail(errors, `Missing docs directory: ${root}`);
    return { ok: false, errors, warnings };
  }

  const indexPath = join(root, 'index.md');
  if (!existsSync(indexPath)) {
    fail(errors, 'docs/index.md is mandatory');
  }

  for (const dir of REQUIRED_DIRS) {
    const p = join(root, dir);
    if (!existsSync(p) || !lstatSync(p).isDirectory()) {
      fail(errors, `Missing required directory: docs/${dir}/`);
    }
  }

  const files = walk(root);

  for (const file of files) {
    const rel = relative(root, file.path);

    if (file.symlink) {
      fail(errors, `Symlinks are not allowed: ${rel}`);
      continue;
    }

    // Path traversal / absolute escape via realpath
    try {
      const real = realpathSync(file.path);
      if (!real.startsWith(root + sep) && real !== root) {
        fail(errors, `Path escapes docs root: ${rel}`);
        continue;
      }
    } catch {
      fail(errors, `Unable to resolve path: ${rel}`);
      continue;
    }

    const parts = rel.split(sep);
    for (const part of parts) {
      if (part === '..' || part.startsWith('/')) {
        fail(errors, `Illegal path segment in ${rel}`);
      }
    }

    const name = basename(file.path);
    const ext = extname(name).toLowerCase();
    const underAssets = parts[0] === 'assets';

    if (underAssets) {
      if (!ALLOWED_ASSET_EXTS.has(ext)) {
        fail(errors, `Unsupported asset type: ${rel}`);
      }
      for (const part of parts.slice(0, -1)) {
        if (part !== 'assets' && !isKebabName(part)) {
          fail(errors, `Asset directory must be kebab-case: ${rel}`);
        }
      }
      if (!isKebabName(name, { allowDotExt: true })) {
        fail(errors, `Asset filename must be kebab-case: ${rel}`);
      }
      continue;
    }

    if (ext === '.mdx') {
      fail(errors, `Arbitrary MDX is not allowed during MVP: ${rel}`);
      continue;
    }

    if (ext !== '.md') {
      fail(errors, `Unsupported file outside docs/assets/: ${rel}`);
      continue;
    }

    for (const part of parts.slice(0, -1)) {
      if (!isKebabName(part)) {
        fail(errors, `Directory names must be lowercase kebab-case: ${rel}`);
      }
    }
    if (!isKebabName(name, { allowDotExt: true })) {
      fail(errors, `Page filenames must be lowercase kebab-case: ${rel}`);
    }

    const content = readFileSync(file.path, 'utf8');
    const fm = parseFrontmatter(content);
    if (!fm) {
      fail(errors, `Missing YAML frontmatter: ${rel}`);
    } else {
      if (!fm.title?.trim()) fail(errors, `Missing frontmatter title: ${rel}`);
      if (!fm.description?.trim()) fail(errors, `Missing frontmatter description: ${rel}`);
    }

    if (/<[a-zA-Z][^>]*>/.test(content.replace(/^---[\s\S]*?---/, ''))) {
      fail(errors, `Raw HTML is not allowed: ${rel}`);
    }

    // Local link checks
    for (const href of extractMarkdownLinks(content)) {
      const target = href.trim().split('#')[0].split('?')[0];
      if (!target) continue;
      if (/^(https?:|mailto:|tel:)/i.test(target)) continue;
      if (target.startsWith('/products/')) continue; // permanent Hub routes
      if (target.startsWith('/')) {
        // other absolute hub routes are ok as cross-product links
        continue;
      }
      if (target.startsWith('/') || /^[a-zA-Z]:\\/.test(target)) {
        fail(errors, `Absolute filesystem paths are not allowed in links (${rel}): ${href}`);
        continue;
      }
      if (target.includes('..')) {
        // relative parent is ok if it stays in docs; verify resolution
      }
      const resolved = resolve(file.path, '..', target);
      if (!resolved.startsWith(root + sep) && resolved !== root) {
        fail(errors, `Link escapes docs root (${rel}): ${href}`);
        continue;
      }
      if (!existsSync(resolved)) {
        // try with .md
        if (!existsSync(resolved + '.md') && !existsSync(join(resolved, 'index.md'))) {
          fail(errors, `Broken local link (${rel}): ${href}`);
        }
      }
    }

    // Images need alt text — flagged when empty alt
    const emptyAlt = /!\[\s*\]\(/g;
    if (emptyAlt.test(content)) {
      fail(errors, `Images must have meaningful alt text: ${rel}`);
    }
  }

  if (options.spokeId && !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(options.spokeId)) {
    fail(errors, `Invalid spoke id: ${options.spokeId}`);
  }

  return { ok: errors.length === 0, errors, warnings };
}

export function printValidationResult(result, label = 'docs') {
  if (result.ok) {
    console.log(`✓ ${label} validation passed`);
    return;
  }
  console.error(`✗ ${label} validation failed:`);
  for (const err of result.errors) console.error(`  - ${err}`);
}
