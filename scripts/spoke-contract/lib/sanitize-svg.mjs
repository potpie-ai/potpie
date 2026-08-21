/**
 * Allowlist SVG sanitizer shared with Hub ingest.
 * Hub may rewrite a generated copy. This Spoke check uses the same rules
 * to fail closed and never writes Spoke git files.
 */
import { DOMParser, XMLSerializer } from '@xmldom/xmldom';

const ALLOWED_ELEMENTS = new Set([
  'svg',
  'g',
  'defs',
  'symbol',
  'marker',
  'clippath',
  'mask',
  'pattern',
  'lineargradient',
  'radialgradient',
  'stop',
  'path',
  'rect',
  'circle',
  'ellipse',
  'line',
  'polyline',
  'polygon',
  'text',
  'tspan',
  'textpath',
  'title',
  'desc',
  'metadata',
  'use',
  'image',
  'switch',
  'a',
  'filter',
  'feblend',
  'fecolormatrix',
  'fecomponenttransfer',
  'fecomposite',
  'feconvolvematrix',
  'fediffuselighting',
  'fedisplacementmap',
  'fedistantlight',
  'feflood',
  'fefunca',
  'fefuncb',
  'fefuncg',
  'fefuncr',
  'fegaussianblur',
  'feimage',
  'femerge',
  'femergenode',
  'femorphology',
  'feoffset',
  'fepointlight',
  'fespecularlighting',
  'fespotlight',
  'fetile',
  'feturbulence',
]);

const HREF_ATTRS = new Set(['href', 'xlink:href', 'src']);

/** @param {Node} node */
function localName(node) {
  return String(node.localName || node.nodeName || '').toLowerCase();
}

/** @param {Attr} attr */
function attrName(attr) {
  return String(attr.name || '').toLowerCase();
}

/**
 * @param {string} value
 * @param {string} elementName
 * @returns {string | null} kept value, or null to drop
 */
export function sanitizeSvgUrl(value, elementName) {
  const trimmed = String(value).trim();
  if (!trimmed) return null;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
    const scheme = trimmed.split(':', 1)[0].toLowerCase();
    if (scheme === 'javascript' || scheme === 'vbscript' || scheme === 'data' || scheme === 'file') {
      return null;
    }
    if (scheme !== 'http' && scheme !== 'https' && scheme !== 'mailto') {
      return null;
    }
    if (elementName === 'use') return null;
    return trimmed;
  }
  if (trimmed.startsWith('//')) return null;
  if (elementName === 'use' && !trimmed.startsWith('#')) return null;
  return trimmed;
}

/** @param {Node} node */
function removeNode(node) {
  if (node.parentNode) node.parentNode.removeChild(node);
}

/**
 * @param {string} input
 * @returns {{ ok: boolean, svg: string, stripped: boolean, error: string | null }}
 */
export function sanitizeSvg(input) {
  if (typeof input !== 'string' || !input.trim()) {
    return { ok: false, svg: '', stripped: false, error: 'SVG is empty' };
  }
  if (/<!DOCTYPE/i.test(input) || /<!ENTITY/i.test(input)) {
    return {
      ok: false,
      svg: '',
      stripped: false,
      error: 'SVG must not include DOCTYPE or ENTITY declarations',
    };
  }

  let doc;
  try {
    doc = new DOMParser({
      onError(level, message) {
        if (level === 'fatalError' || level === 'error') {
          throw new Error(String(message));
        }
      },
    }).parseFromString(input, 'image/svg+xml');
  } catch (err) {
    return { ok: false, svg: '', stripped: false, error: `Invalid SVG XML: ${err.message}` };
  }

  const root = doc.documentElement;
  if (!root || localName(root) !== 'svg') {
    return { ok: false, svg: '', stripped: false, error: 'Root element must be <svg>' };
  }

  const stripped = { current: false };

  /** @param {Node} node */
  function sanitizeNode(node) {
    if (!node || !node.parentNode && node !== root) return;

    if (node.nodeType === 7 || node.nodeType === 8) {
      stripped.current = true;
      removeNode(node);
      return;
    }
    if (node.nodeType !== 1) return;

    const name = localName(node);
    if (!ALLOWED_ELEMENTS.has(name)) {
      stripped.current = true;
      removeNode(node);
      return;
    }

    const attrs = node.attributes ? Array.from(node.attributes) : [];
    for (const attr of attrs) {
      const an = attrName(attr);
      if (an.startsWith('on') || an === 'handler') {
        stripped.current = true;
        node.removeAttribute(attr.name);
        continue;
      }
      if (HREF_ATTRS.has(an)) {
        const kept = sanitizeSvgUrl(attr.value, name);
        if (kept == null) {
          stripped.current = true;
          node.removeAttribute(attr.name);
        } else if (kept !== attr.value) {
          stripped.current = true;
          node.setAttribute(attr.name, kept);
        }
        continue;
      }
      if (an === 'style') {
        stripped.current = true;
        node.removeAttribute(attr.name);
      }
    }

    const children = Array.from(node.childNodes);
    for (const child of children) sanitizeNode(child);
  }

  if (doc.doctype) {
    stripped.current = true;
    removeNode(doc.doctype);
  }

  const prologue = Array.from(doc.childNodes).filter((n) => n !== root);
  for (const n of prologue) {
    if (n.nodeType === 7 || n.nodeType === 8) {
      stripped.current = true;
      removeNode(n);
    }
  }

  sanitizeNode(root);

  if (!doc.documentElement || localName(doc.documentElement) !== 'svg') {
    return { ok: false, svg: '', stripped: true, error: 'SVG sanitizer removed the root element' };
  }

  const svg = new XMLSerializer().serializeToString(doc);
  return { ok: true, svg, stripped: stripped.current, error: null };
}
