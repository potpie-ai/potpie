import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import { sanitizeSvg, sanitizeSvgUrl } from './sanitize-svg.mjs';

describe('sanitizeSvgUrl', () => {
  test('drops javascript, data, and vbscript', () => {
    assert.equal(sanitizeSvgUrl('javascript:alert(1)', 'a'), null);
    assert.equal(sanitizeSvgUrl('data:image/svg+xml,x', 'image'), null);
    assert.equal(sanitizeSvgUrl('vbscript:x', 'a'), null);
  });

  test('allows https on image and hash-only on use', () => {
    assert.equal(sanitizeSvgUrl('https://example.com/a.png', 'image'), 'https://example.com/a.png');
    assert.equal(sanitizeSvgUrl('https://example.com/a.svg#x', 'use'), null);
    assert.equal(sanitizeSvgUrl('#icon', 'use'), '#icon');
  });
});

describe('sanitizeSvg', () => {
  test('strips script and onload', () => {
    const { ok, svg, stripped, error } = sanitizeSvg(
      `<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">
  <script>alert(1)</script>
  <rect width="10" height="10"/>
</svg>`,
    );
    assert.equal(error, null);
    assert.equal(ok, true);
    assert.equal(stripped, true);
    assert.equal(svg.toLowerCase().includes('script'), false);
    assert.equal(svg.toLowerCase().includes('onload'), false);
    assert.match(svg, /rect/);
  });

  test('strips javascript href', () => {
    const { ok, svg, stripped } = sanitizeSvg(
      `<svg xmlns="http://www.w3.org/2000/svg"><a href="javascript:alert(1)"><text>x</text></a></svg>`,
    );
    assert.equal(ok, true);
    assert.equal(stripped, true);
    assert.equal(svg.toLowerCase().includes('javascript'), false);
  });

  test('leaves a clean icon unchanged in meaning', () => {
    const input = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><circle cx="4" cy="4" r="3"/></svg>`;
    const { ok, stripped, error } = sanitizeSvg(input);
    assert.equal(ok, true);
    assert.equal(error, null);
    assert.equal(stripped, false);
  });
});
