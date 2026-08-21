#!/usr/bin/env node
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const mod = await import(pathToFileURL(join(__dirname, 'validate-docs.mjs')).href);
const docsRoot = resolve(process.argv[2] || 'docs');
const spokeId = process.argv[3] || 'potpie';
const result = mod.validateSpokeDocs(docsRoot, { spokeId });
mod.printValidationResult(result, docsRoot);
process.exit(result.ok ? 0 : 1);
