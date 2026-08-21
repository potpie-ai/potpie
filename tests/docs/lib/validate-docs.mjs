#!/usr/bin/env node
/**
 * Shared Spoke docs validation. Keep aligned with Hub scripts/lib/validate-docs.mjs.
 * Requires docs/index.md. Section dirs (getting-started, guides, reference, assets)
 * are optional. Rejects unsupported files, symlinks, path traversal, MDX, and
 * missing frontmatter.
 *
 * Hub ingest may rewrite a generated SVG copy. This Spoke check never writes
 * Spoke git files; unsafe SVG fails closed instead.
 */

import { existsSync, lstatSync, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { join, relative, resolve, sep, extname, basename } from 'node:path';
import { parse as parseYaml } from 'yaml';
import { sanitizeSvg } from './sanitize-svg.mjs';

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
const ALLOWED_LINK_SCHEMES = new Set(['http:', 'https:', 'mailto:', 'tel:']);

const RAW_HTML_TAGS = new Set([
  'a', 'article', 'aside', 'audio',
  'base', 'body', 'br', 'button',
  'canvas', 'col', 'colgroup',
  'div',
  'embed',
  'fieldset', 'figcaption', 'figure', 'footer', 'form',
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr', 'html',
  'iframe', 'img', 'input',
  'label', 'legend', 'li', 'link',
  'main', 'meta',
  'nav', 'noscript',
  'object', 'ol',
  'p', 'picture', 'pre',
  'script', 'section', 'select', 'span', 'style', 'svg',
  'table', 'tbody', 'td', 'template', 'textarea', 'tfoot', 'th', 'thead', 'tr',
  'ul',
  'video',
]);

/**
 * Drop fenced and inline code so CLI placeholders and mermaid <br/> stay allowed.
 * @param {string} text
 */
export function markdownWithoutCode(text) {
  const proseLines = [];
  let fence = null;

  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    if (fence) {
      const closing = line.match(/^ {0,3}(`+|~+)[ \t]*$/);
      if (closing && closing[1][0] === fence.marker && closing[1].length >= fence.length) {
        fence = null;
      }
      continue;
    }

    const opening = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (opening) {
      fence = { marker: opening[1][0], length: opening[1].length };
      continue;
    }

    proseLines.push(line);
  }

  return proseLines.join('\n').replace(/`[^`\n]*`/g, '');
}

/**
 * True when Markdown prose contains an HTML element or MDX component tag.
 * Angle-bracket CLI placeholders such as <id>, <source>, and <s> are allowed.
 * @param {string} text
 */
export function containsRawHtml(text) {
  const scanned = markdownWithoutCode(text);
  const re = /<\/?([A-Za-z][A-Za-z0-9:-]*)\b([^>]*)>/g;
  let match;
  while ((match = re.exec(scanned)) !== null) {
    const name = match[1];
    const attrs = match[2] || '';
    if (/^[A-Z]/.test(name)) return true;
    const isHtml = RAW_HTML_TAGS.has(name.toLowerCase());
    if (!isHtml) continue;
    if (name.length === 1 && !attrs.trim()) continue;
    return true;
  }
  return false;
}

/** @param {string[]} errors @param {string} message */
function fail(errors, message) {
  errors.push(message);
}

/** @param {unknown} value @returns {value is string} */
function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * @param {string} name
 * @param {{ allowDotExt?: boolean }} [options]
 * @returns {boolean}
 */
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
 * Single frontmatter boundary used by both YAML parse and body HTML checks.
 * @param {string} content
 * @returns {{ block: string, body: string } | { error: string }}
 */
export function splitFrontmatter(content) {
  if (!content.startsWith('---')) {
    return { error: 'Missing YAML frontmatter' };
  }
  const match = /\n---[ \t]*(?:\r?\n|$)/.exec(content);
  if (!match) {
    return { error: 'Missing YAML frontmatter closing delimiter' };
  }
  return {
    block: content.slice(content.indexOf('\n') + 1, match.index),
    body: content.slice(match.index + match[0].length),
  };
}

/** @param {string} target @returns {string | null} */
function linkScheme(target) {
  const match = target.match(/^([a-zA-Z][a-zA-Z0-9+.-]*:)/);
  return match ? match[1].toLowerCase() : null;
}

/**
 * @param {string} content
 * @returns {{ data: Record<string, unknown> | null, error: string | null, body?: string }}
 */
export function parseFrontmatter(content) {
  const split = splitFrontmatter(content);
  if ('error' in split) {
    return { data: null, error: split.error };
  }
  try {
    const data = parseYaml(split.block, { maxAliasCount: 0 });
    if (data == null || typeof data !== 'object' || Array.isArray(data)) {
      return { data: null, error: 'Frontmatter must be a YAML mapping' };
    }
    return { data, error: null, body: split.body };
  } catch (err) {
    return { data: null, error: `Invalid YAML frontmatter: ${err.message}` };
  }
}

/**
 * @param {string} dir
 * @param {{ path: string, symlink: boolean, dirent: import('node:fs').Dirent }[]} [files]
 * @returns {{ path: string, symlink: boolean, dirent: import('node:fs').Dirent }[]}
 */
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

/**
 * Find the closing parenthesis after an optional quoted Markdown link title.
 * @param {string} content
 * @param {number} start
 * @returns {number}
 */
function findLinkClose(content, start) {
  let cursor = start;
  while (cursor < content.length && /[ \t]/.test(content[cursor])) cursor += 1;
  if (content[cursor] === ')') return cursor;

  if (content[cursor] === '(') {
    let depth = 0;
    while (cursor < content.length) {
      if (content[cursor] === '\\') {
        cursor += 2;
        continue;
      }
      if (content[cursor] === '(') {
        depth += 1;
      } else if (content[cursor] === ')') {
        depth -= 1;
        if (depth === 0) {
          cursor += 1;
          while (cursor < content.length && /[ \t]/.test(content[cursor])) cursor += 1;
          return content[cursor] === ')' ? cursor : -1;
        }
      }
      if (content[cursor] === '\n') return -1;
      cursor += 1;
    }
    return -1;
  }

  const quote = content[cursor];
  if (quote !== '"' && quote !== "'") return -1;
  cursor += 1;
  while (cursor < content.length) {
    if (content[cursor] === '\\') {
      cursor += 2;
      continue;
    }
    if (content[cursor] === quote) {
      cursor += 1;
      while (cursor < content.length && /[ \t]/.test(content[cursor])) cursor += 1;
      return content[cursor] === ')' ? cursor : -1;
    }
    if (content[cursor] === '\n') return -1;
    cursor += 1;
  }
  return -1;
}

/**
 * Extract inline Markdown link destinations while preserving balanced parentheses.
 * @param {string} content
 * @returns {string[]}
 */
function extractMarkdownLinks(content) {
  const links = [];
  for (let index = 0; index < content.length - 1; index += 1) {
    if (content[index] !== ']' || content[index + 1] !== '(') continue;

    let cursor = index + 2;
    while (cursor < content.length && /[ \t]/.test(content[cursor])) cursor += 1;
    const destinationStart = cursor;

    if (content[cursor] === '<') {
      cursor += 1;
      const angleStart = cursor;
      while (cursor < content.length && content[cursor] !== '\n') {
        if (content[cursor] === '\\') {
          cursor += 2;
          continue;
        }
        if (content[cursor] === '>') {
          const close = findLinkClose(content, cursor + 1);
          if (close !== -1) {
            links.push(content.slice(angleStart, cursor));
            index = close;
          }
          break;
        }
        cursor += 1;
      }
      continue;
    }

    let depth = 0;
    while (cursor < content.length && content[cursor] !== '\n') {
      const character = content[cursor];
      if (character === '\\') {
        cursor += 2;
        continue;
      }
      if (character === '(') {
        depth += 1;
      } else if (character === ')') {
        if (depth === 0) {
          if (cursor > destinationStart) links.push(content.slice(destinationStart, cursor));
          index = cursor;
          break;
        }
        depth -= 1;
      } else if (/[ \t]/.test(character) && depth === 0) {
        const close = findLinkClose(content, cursor);
        if (close !== -1 && cursor > destinationStart) {
          links.push(content.slice(destinationStart, cursor));
          index = close;
        }
        break;
      }
      cursor += 1;
    }
  }
  return links;
}

/** @param {string} target @returns {string} */
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
      let sizeOk = true;
      try {
        const size = statSync(file.path).size;
        if (size > maxAssetBytes) {
          fail(
            errors,
            `Asset exceeds ${maxAssetBytes} byte limit (${rel} is ${size} bytes)`,
          );
          sizeOk = false;
        }
      } catch {
        fail(errors, `Unable to stat asset: ${rel}`);
        sizeOk = false;
      }
      if (sizeOk && ext === '.svg') {
        const raw = readFileSync(file.path, 'utf8');
        const svg = sanitizeSvg(raw);
        if (!svg.ok) {
          fail(errors, `Unsafe or invalid SVG (${rel}): ${svg.error}`);
        } else if (svg.stripped) {
          fail(
            errors,
            `Unsafe SVG content (${rel}): remove scripts, event handlers, and disallowed URLs. Hub ingest sanitizes a generated copy; this check does not rewrite Spoke files.`,
          );
        }
      }
      continue;
    }

    // Spoke metadata for CI. Not a published page.
    if (parts.length === 1 && name === 'config.json') {
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
    const split = splitFrontmatter(content);
    const body = 'body' in split ? split.body : content;
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

    if (containsRawHtml(body)) {
      fail(errors, `Raw HTML is not allowed: ${rel}`);
    }

    const prose = markdownWithoutCode(body);
    for (const href of extractMarkdownLinks(prose)) {
      const target = href.trim().split('#')[0].split('?')[0];
      if (!target) continue;
      if (/^[a-zA-Z]:[\\/]/.test(target)) {
        fail(errors, `Absolute filesystem paths are not allowed in links (${rel}): ${href}`);
        continue;
      }
      const scheme = linkScheme(target);
      if (scheme) {
        if (!ALLOWED_LINK_SCHEMES.has(scheme)) {
          fail(errors, `Disallowed link scheme (${rel}): ${href}`);
        }
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

    if (/!\[\s*\]\(/.test(prose)) {
      fail(errors, `Images must have meaningful alt text: ${rel}`);
    }
  }

  if (options.spokeId && !KEBAB_CASE.test(options.spokeId)) {
    fail(errors, `Invalid spoke id: ${options.spokeId}`);
  }

  return { ok: errors.length === 0, errors, warnings };
}

/**
 * @param {{ ok: boolean, errors: string[] }} result
 * @param {string} [label]
 */
export function printValidationResult(result, label = 'docs') {
  if (result.ok) {
    console.log(`✓ ${label} validation passed`);
    return;
  }
  console.error(`✗ ${label} validation failed:`);
  for (const err of result.errors) console.error(`  - ${err}`);
}
