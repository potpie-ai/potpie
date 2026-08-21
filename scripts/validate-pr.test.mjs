import test from 'node:test';
import assert from 'node:assert/strict';
import { validatePrTitle, validatePrDescription } from './validate-pr.mjs';

test('accepts conventional docs title', () => {
  const result = validatePrTitle('docs(potpie): clarify installation prerequisites');
  assert.equal(result.ok, true);
});

test('rejects title with trailing period', () => {
  const result = validatePrTitle('docs(potpie): clarify installation.');
  assert.equal(result.ok, false);
});

test('rejects vague title without scope', () => {
  const result = validatePrTitle('Updated docs');
  assert.equal(result.ok, false);
});

test('accepts complete PR description with docs updated', () => {
  const body = `## Summary

Clarify install steps for the Potpie CLI.

## User-facing impact

Readers see clearer prerequisites.

## Documentation impact

- [x] Documentation was updated in this PR.
- [ ] Documentation is not required.

Reason documentation is not required:

## Files or sections updated

- docs/getting-started/installation.md

## Testing performed

- Local docs validator

## Related issue or specification

- N/A

## Screenshots
`;
  const result = validatePrDescription(body);
  assert.equal(result.ok, true, result.errors.join('; '));
});

test('rejects description without checked documentation impact', () => {
  const body = `## Summary
x
## User-facing impact
x
## Documentation impact
- [ ] Documentation was updated in this PR.
- [ ] Documentation is not required.
Reason documentation is not required:
## Files or sections updated
x
## Testing performed
x
## Related issue or specification
x
## Screenshots
`;
  const result = validatePrDescription(body);
  assert.equal(result.ok, false);
});

test('requires reason when docs-not-required is checked', () => {
  const body = `## Summary
Internal CI tweak.
## User-facing impact
None.
## Documentation impact
- [ ] Documentation was updated in this PR.
- [x] Documentation is not required.
Reason documentation is not required:
<!-- still empty -->
## Files or sections updated
.github/workflows/docs-validate.yml
## Testing performed
node --test
## Related issue or specification
N/A
## Screenshots
`;
  const result = validatePrDescription(body);
  assert.equal(result.ok, false);
});
