/**
 * Spoke PR metadata checks: title convention + required description sections.
 * Used by CI; also unit-tested via node:test.
 */

const APPROVED_TYPES = ['docs', 'feat', 'fix', 'refactor', 'chore'];

/** Conventional title: type(scope): description — no trailing period */
export const PR_TITLE_PATTERN = /^(docs|feat|fix|refactor|chore)\([a-z0-9]+(?:-[a-z0-9]+)*\): [A-Za-z0-9].*[^.]$/;

const REQUIRED_HEADINGS = [
  'Summary',
  'User-facing impact',
  'Documentation impact',
  'Files or sections updated',
  'Testing performed',
  'Related issue or specification',
];

/**
 * @param {string} title
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validatePrTitle(title) {
  const errors = [];
  const value = (title || '').trim();

  if (!value) {
    errors.push('PR title is empty');
    return { ok: false, errors };
  }

  if (value.endsWith('.')) {
    errors.push('PR title must not end with a period');
  }

  if (!PR_TITLE_PATTERN.test(value)) {
    errors.push(
      `PR title must match: <type>(<scope>): <short description> (types: ${APPROVED_TYPES.join(', ')}; kebab-case scope; no trailing period)`,
    );
    errors.push('Example: docs(potpie): clarify installation prerequisites');
  }

  return { ok: errors.length === 0, errors };
}

/**
 * @param {string} body
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validatePrDescription(body) {
  const errors = [];
  const text = body || '';

  if (!text.trim()) {
    errors.push('PR description is empty; use the pull request template');
    return { ok: false, errors };
  }

  for (const heading of REQUIRED_HEADINGS) {
    const re = new RegExp(`^##\\s+${heading}\\s*$`, 'im');
    if (!re.test(text)) {
      errors.push(`PR description is missing heading: ## ${heading}`);
    }
  }

  const docsUpdated = /-\s*\[[xX]\]\s*Documentation was updated in this PR/i.test(text);
  const docsNotRequired = /-\s*\[[xX]\]\s*Documentation is not required/i.test(text);

  if (!docsUpdated && !docsNotRequired) {
    errors.push(
      'Documentation impact: check exactly one of "Documentation was updated in this PR" or "Documentation is not required"',
    );
  }

  if (docsUpdated && docsNotRequired) {
    errors.push('Documentation impact: do not check both documentation options');
  }

  if (docsNotRequired) {
    const afterReason = text.split(/Reason documentation is not required:\s*/i)[1] || '';
    const reasonOnly = afterReason.split(/\n##\s+/)[0] || '';
    const reasonLines = reasonOnly
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith('<!--') && !l.startsWith('-->') && !l.startsWith('#'));
    if (reasonLines.length === 0) {
      errors.push(
        'When documentation is not required, write a non-empty reason under "Reason documentation is not required"',
      );
    }
  }

  // Summary must have more than the HTML comment placeholder
  const summaryMatch = text.match(/##\s+Summary\s*([\s\S]*?)(?=\n##\s+)/i);
  if (summaryMatch) {
    const summary = summaryMatch[1]
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith('<!--'))
      .join(' ');
    if (!summary) {
      errors.push('Summary section must describe the change (not only an HTML comment)');
    }
  }

  return { ok: errors.length === 0, errors };
}

export function printResult(label, result) {
  if (result.ok) {
    console.log(`✓ ${label}`);
    return;
  }
  console.error(`✗ ${label}`);
  for (const err of result.errors) console.error(`  - ${err}`);
}
