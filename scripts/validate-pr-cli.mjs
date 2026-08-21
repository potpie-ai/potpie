#!/usr/bin/env node
import { validatePrTitle, validatePrDescription, printResult } from './validate-pr.mjs';

const title = process.env.PR_TITLE || '';
const body = process.env.PR_BODY || '';

const titleResult = validatePrTitle(title);
const bodyResult = validatePrDescription(body);

printResult('PR title', titleResult);
printResult('PR description', bodyResult);

if (!titleResult.ok || !bodyResult.ok) {
  process.exit(1);
}
