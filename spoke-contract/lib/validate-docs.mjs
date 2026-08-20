#!/usr/bin/env bun
/**
 * Shared Spoke docs validation.
 * Requires docs/index.md. Section dirs (getting-started, guides, reference, assets)
 * are optional. Rejects unsupported files, symlinks, path traversal, MDX, and
 * missing frontmatter.
 */

import { existsSync, lstatSync, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { join, relative, resolve, sep, extname, basename } from 'node:path';
import { parse as parseYaml } from 'yaml';

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
export const DEFAULT_MAX_ASSET_BYTES = 20 * 1024 * 1024;
const HUB_EXACT_ROUTES = new Set([
  '/',
  '/products',
  '/documentation-contract',
  '/architecture',
  '/contributing',
]);

function fail(errors, message) {
  errors.push(message);
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
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

/**
 * @param {string} content
 * @returns {{ data: Record<string, unknown> | null, error: string | null }}
 */
export function parseFrontmatter(content) {
  if (!content.startsWith('---')) {
    return { data: null, error: 'Missing YAML frontmatter' };
  }
  const end = content.indexOf('\n---', 3);
  if (end === -1) {
    return { data: null, error: 'Missing YAML frontmatter closing delimiter' };
  }
  const block = content.slice(4, end);
  try {
    const data = parseYaml(block);
    if (data == null || typeof data !== 'object' || Array.isArray(data)) {
      return { data: null, error: 'Frontmatter must be a YAML mapping' };
    }
    return { data, error: null };
  } catch (err) {
    return { data: null, error: `Invalid YAML frontmatter: ${err.message}` };
  }
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

function normalizeHubPath(target) {
  if (target.length > 1 && target.endsWith('/')) return target.slice(0, -1);
  return target || '/';
}

/**
 * @param {string} target path without hash/query
 * @param {string | undefined} spokeId
 * @returns {boolean}
 */
export function isAllowedAbsoluteHubRoute(target, spokeId) {
  if (!target.startsWith('/')) return false;
  const normalized = normalizeHubPath(target);
  if (HUB_EXACT_ROUTES.has(normalized)) return true;
  const productMatch = normalized.match(/^\/products\/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\/.*)?$/);
  if (!productMatch) return false;
  if (!spokeId) return true;
  return productMatch[1] === spokeId;
}

/**
 * @param {string} docsRoot absolute path to Spoke docs/ directory
 * @param {{ spokeId?: string, maxAssetBytes?: number }} [options]
 * @returns {{ ok: boolean, errors: string[], warnings: string[] }}
 */
export function validateSpokeDocs(docsRoot, options = {}) {
  const errors = [];
  const warnings = [];
  const requestedRoot = resolve(docsRoot);
  const maxAssetBytes = options.maxAssetBytes ?? DEFAULT_MAX_ASSET_BYTES;

  if (!existsSync(requestedRoot)) {
    fail(errors, `Missing docs directory: ${requestedRoot}`);
    return { ok: false, errors, warnings };
  }

  const root = realpathSync(requestedRoot);

  const indexPath = join(root, 'index.md');
  if (!existsSync(indexPath)) {
    fail(errors, 'docs/index.md is mandatory');
  }

  const files = walk(root);

  for (const file of files) {
    const rel = relative(root, file.path);

    if (file.symlink) {
      fail(errors, `Symlinks are not allowed: ${rel}`);
      continue;
    }

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
      try {
        const size = statSync(file.path).size;
        if (size > maxAssetBytes) {
          fail(
            errors,
            `Asset exceeds ${maxAssetBytes} byte limit (${rel} is ${size} bytes)`,
          );
        }
      } catch {
        fail(errors, `Unable to stat asset: ${rel}`);
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
    if (fm.error) {
      fail(errors, `${fm.error}: ${rel}`);
    } else {
      if (!isNonEmptyString(fm.data.title)) {
        fail(errors, `Missing frontmatter title: ${rel}`);
      }
      if (!isNonEmptyString(fm.data.description)) {
        fail(errors, `Missing frontmatter description: ${rel}`);
      }
    }

    if (/<[a-zA-Z][^>]*>/.test(content.replace(/^---[\s\S]*?---/, ''))) {
      fail(errors, `Raw HTML is not allowed: ${rel}`);
    }

    for (const href of extractMarkdownLinks(content)) {
      const target = href.trim().split('#')[0].split('?')[0];
      if (!target) continue;
      if (/^(https?:|mailto:|tel:)/i.test(target)) continue;
      if (/^[a-zA-Z]:[\\/]/.test(target) || /^file:/i.test(target)) {
        fail(errors, `Absolute filesystem paths are not allowed in links (${rel}): ${href}`);
        continue;
      }
      if (target.startsWith('/')) {
        if (!isAllowedAbsoluteHubRoute(target, options.spokeId)) {
          fail(errors, `Unknown absolute route (${rel}): ${href}`);
        }
        continue;
      }
      const resolved = resolve(file.path, '..', target);
      if (!resolved.startsWith(root + sep) && resolved !== root) {
        fail(errors, `Link escapes docs root (${rel}): ${href}`);
        continue;
      }
      if (!existsSync(resolved)) {
        if (!existsSync(resolved + '.md') && !existsSync(join(resolved, 'index.md'))) {
          fail(errors, `Broken local link (${rel}): ${href}`);
        }
      }
    }

    const emptyAlt = /!\[\s*\]\(/g;
    if (emptyAlt.test(content)) {
      fail(errors, `Images must have meaningful alt text: ${rel}`);
    }
  }

  if (options.spokeId && !KEBAB_CASE.test(options.spokeId)) {
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
