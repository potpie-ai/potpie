import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, test } from 'node:test';
import { parse } from 'yaml';

const workflowPath = fileURLToPath(
  new URL('../../.github/workflows/docs-release-dispatch.yml', import.meta.url),
);

describe('docs release dispatch workflow', () => {
  test('validates the selected stable tag before dispatching its exact tag and SHA', () => {
    const workflow = parse(readFileSync(workflowPath, 'utf8'));
    const steps = workflow.jobs.dispatch.steps;
    const select = steps.findIndex((step) => step.id === 'release');
    const readiness = steps.findIndex((step) => step.name === 'Verify documentation release readiness');
    const dispatch = steps.find((step) => step.name === 'Send release dispatch to Hub');

    assert.ok(select !== -1 && readiness > select);
    assert.match(steps[readiness].run, /--tag/);
    assert.match(dispatch.run, /tag:process\.env\.TAG/);
    assert.match(dispatch.run, /sha:process\.env\.SHA/);
  });
});
