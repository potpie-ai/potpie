import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, test } from 'node:test';
import { parse } from 'yaml';

const workflowPath = fileURLToPath(
  new URL('../../.github/workflows/docs-dispatch.yml', import.meta.url),
);

function loadDispatchSteps() {
  const workflow = parse(readFileSync(workflowPath, 'utf8'));
  return workflow.jobs.dispatch.steps;
}

describe('docs dispatch workflow', () => {
  test('checks whether docs changed before minting a GitHub App token', () => {
    const steps = loadDispatchSteps();
    const docsChangeIndex = steps.findIndex((step) => step.id === 'docs_change');
    const tokenIndex = steps.findIndex((step) => step.id === 'hub_token');

    assert.notEqual(docsChangeIndex, -1);
    assert.notEqual(tokenIndex, -1);
    assert.ok(docsChangeIndex < tokenIndex);
    assert.equal(
      steps[tokenIndex].if,
      "steps.docs_change.outputs.should_dispatch == 'true'",
    );
  });
});
