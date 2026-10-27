# Engine tests

Tests have two independent organizing dimensions. The directory at the root
states the execution purpose; its domain subdirectory owns the behavior.
`core` exercises pure parsing, validation and lowering. `unit` exercises an
engine service or adapter with local dependencies. `integration` exercises
native providers, subprocesses or the HTTP boundary. `conformance` runs the
same public contract across backend and resource profiles. `contracts` holds
shared value case catalogs.

The ownership catalog is [suite_catalog.json](suite_catalog.json). Every test
module appears once, including its definition count. Counts describe Python
test functions; pytest can collect several named cases from one function.

| Domain | Test modules | Test definitions |
| --- | ---: | ---: |
| `agents` | 15 | 76 |
| `backends` | 28 | 289 |
| `benchmarks` | 12 | 54 |
| `durability` | 16 | 135 |
| `ingestion` | 24 | 165 |
| `mutations` | 43 | 540 |
| `observability` | 14 | 76 |
| `ontology` | 17 | 196 |
| `resources` | 13 | 180 |
| `retrieval` | 19 | 220 |
| `runtime` | 15 | 128 |

`graph_reliability.py` owns backend profile construction and isolation shared
by the protocol, temporal and backend conformance suites. It checks optional
native packages and explicit test-server endpoints. Integration fixtures stay
under `integration/conftest.py`, so moving a test into a domain preserves its
live-service fixture scope. Stateful recovery and failure scenarios remain
ordinary test functions; helper classes represent actual fakes or value
objects. Test classes have been flattened into named functions.

Pure value contracts use named data in `cases/values.py` and the functions in
`contracts/test_values.py`. Identity, classification, source reference and
resource port cases share the same assertion structure and keep case IDs in
pytest output. Tests that inspect repository sources carry `source_contract`;
installed-artifact acceptance excludes them. Native process tests use isolated
interpreters so their imports come from the installed package.

The migration evidence is [migration_map.json](migration_map.json) for module
moves and [definition_migration.json](definition_migration.json) for the exact 2,289 original test
node identities at Git HEAD, with explicit consolidation targets and retirement
reasons. New tests appear only in the ownership catalog. Historical repair, receipt upgrade, snapshot conversion and
removed API shim contracts were retired with their production implementation.
Current corruption rejection, mutation atomicity, recovery, idempotency,
endpoint scope and backend capability checks remain part of acceptance.

Run the local suite from the repository root after installing the workspace:

```sh
rtk pytest potpie/context-engine/tests
```

The CI suite runner builds distributions first and runs provider acceptance
against installed artifacts. Provider tests require their declared native
package or isolated service endpoint; the required acceptance tier turns
unexpected skips into failures.
