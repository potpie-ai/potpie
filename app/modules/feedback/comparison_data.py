"""Q&A comparison pairs for blind Potpie-vs-Copilot human feedback collection.

Edit this file manually to add, remove or update comparisons. Each entry has:
- id: stable string id stored alongside every vote (don't rename once shipped)
- question: the user prompt as shown to the voter
- responses: list of {"model": "potpie"|"copilot", "answer": str}

The voter NEVER sees the `model` field. The API shuffles the two responses
with a server-side seed and presents them as "Response A" / "Response B".
"""

from typing import Dict, List, TypedDict


class _Response(TypedDict):
    model: str  # "potpie" | "copilot"
    answer: str


class _Comparison(TypedDict):
    id: str
    question: str
    responses: List[_Response]


COMPARISONS: List[_Comparison] = [
    {
        "id": "analyte-stability",
        "question": (
            "how is the concept of Analyte Stability implemented or managed in the code? "
            "Please explain which modules, classes, or functions are responsible for "
            "handling Analyte Stability and describe the approach or logic used."
        ),
        "responses": [
            {
                "model": "potpie",
                "answer": """## Classification
BROAD_OVERVIEW

## Summary
"Analyte Stability" is not implemented as a single named feature or class in this repository. In the code, the concept is primarily managed as **stability metadata and expiration logic** for **reagent packs, calibrators, controls, and consumables**—especially **onboard stability (OBS)**, **shelf stability**, and **when stability counting starts**. These values are defined in test/reagent configuration, stored in DB/schema objects, attached to runtime pack/consumable objects, and then enforced by validation, result-processing, background expiry checks, and UI status mapping.

The closest runtime enforcement points are `CalOrderValidator.ValidatePackOnboardStability()` for calibration-order rejection, `ReagentDataRuleAssessor.Evaluate()` for OBS-waiver logic during result processing, `ReagentPackBuilder` for loading stability policy onto pack objects, and `BulkConsumableVerifier` for time-based expiry/alert transitions. There is also a related sample-level model for calibrators/controls in `OnboardSample.cs`.

## Investigation Process
ColGREP was unavailable (`COLGREP_API_BASE_URL` not configured), so I used `rg` searches scoped across likely domains (`IM/MM`, `CI/MM`, `CI/ARM`, `Platform/SIM`, `UI/UDD`, `CO/DCM`) and then read the most relevant files.

I found:
- configuration-level stability definitions in IA TDef files and DB schema,
- runtime pack/consumable stability properties and expiration timestamps,
- validation and waiver logic in calibration/result-processing code,
- UI/status mapping for expired stability states.

## Detailed Findings

### 1. Configuration-level stability fields in TDef files and DB schema
- testing/reagent stability bounds live in TDef test-definition configs
- nominal pack/consumable stability timestamps, OBS reference
- calibration-order configuration uses stability override flags

### 2. Runtime stability properties on pack/consumable objects
- `ReagentPackBuilder` populates stability policy + first-use timestamps
- `BulkConsumableVerifier` runs periodic checks and flips alert state

### 3. Validation enforcement during calibration order acceptance
- `CalOrderValidator.ValidatePackOnboardStability()` rejects when expired
- This is the single point that decides "can this pack still be used for cal".

### 4. Waiver logic during result processing
- `ReagentDataRuleAssessor.Evaluate()` applies OBS waivers when configured
- Linked to OBS-waiver rules in the dataset; allows results from a pack just past expiry.

### 5. UI/status mapping for expired stability states
- `FieldAvailability.cs` and `ReagentExpiryStatusMapper.cs` translate expiry into UI badges.

## Evidence
| Claim | Source | Citation |
|-------|--------|----------|
| OBS validation lives here | reads C# file | `CalOrderValidator.cs` |
| Waiver evaluation lives here | reads C# file | `ReagentDataRuleAssessor.cs` |
| Pack builder loads policy | reads C# file | `ReagentPackBuilder.cs` |

## References Found in Code
| Type | Reference | Found In | Context |
|------|-----------|----------|---------|
| Spec | OBS-waiver | dataset rules | "OBS waiver" |

## Limitations & Uncertainties
- No single class is named `AnalyteStability`. The concept is spread across reagent/cal/QC pipelines.
""",
            },
            {
                "model": "copilot",
                "answer": """Analyte Stability shows up in this repo in two related (but distinct) ways: (1) a test-definition field called AnalyteStability, and (2) runtime assay/sample stability expiration driven by AssayStabilityInformation on onboard QC/Cal products.

1) Test-definition "AnalyteStability" (setup/config)
- Model: Siemens.Atellica.Model.TdefCommon.AnalyteStability in TDefCommon.cs (stored as a string).
- ViewModels that own the field:
  - CHDefinitionViewModel (CHDefinitionViewModel.cs)
    - Loads from tdefUI.TdefCommon.AnalyteStability in SetTestDefinitionData(...)
    - Saves back to TestDefinition.TdefCommon.AnalyteStability in GetDirtyValues()
    - Uses min/max from _appConfigurationManager.CHTestDefinitionValidation.Definition.AnalyteStabilityMinValue/MaxValue
  - OpenChannelDefinitionViewModel (same pattern) in OpenChannelDefinitionViewModel.cs
  - IMDefinitionViewModel (same pattern; IM-specific validation config) in IMDefinitionViewModel.cs
- Validation/config source:
  - Bounds live in CHDefinitionValidation.AnalyteStabilityMinValue/MaxValue and IMDefinitionValidation.AnalyteStabilityMinValue/MaxValue (UI/PrimeDisp/source/Siemens.Atellica.Model/...).
  - Loaded from JSON via Siemens.Atellica.PrimeDisp.AppConfigurationManager (AppConfigurationManager.cs) from:
    - CHTestDefinitionValidationConfig.json
    - IMTestDefinitionValidationConfig.json
- Extra UI logic:
  - IMDefinitionView.xaml.cs clears the field if parsed int is below the min on LostFocus (IMDefinitionView.xaml.cs).
  - FieldLevelAuthProvider can override the max allowed value based on G0 settings by parsing TestDefinitionG0Settings.TdefCommon.AnalyteStability (FieldLevelAuthProvider.cs).

2) Runtime "assay stability" for onboard products (QC/Cal)
This is where stability is actually computed/expired and turned into statuses/flags shown in the UI.
- Core stability engine: Siemens.Trinidad.SIM.SampleInfoMgr.QcCal.CalQcProductFactory in CalQcProductFactory.cs
  - Creates per-assay stability records (AssayStabilityInformation) and stores them on the sample (IProductSample.AssaysInProduct):
    - ControlIdentified(...): uses QCTestRanges.OpenVialStability; if 0 hours it sets a "never-expire" default; builds new AssayStabilityInformation(testName, openStab, TimeSpan(0,1,0,0))
    - CalibratorIdentified(...): builds AssayStabilityInformation(mnemonic, obsRefrig, obsUnRefrig) using TDef

If you want, I can zoom in on any of these areas.
""",
            },
        ],
    },
    {
        "id": "udd-startup-and-new-command",
        "question": (
            "How UDD Data layer is startup working? How to add a new feature that requires "
            "a new command from UI?"
        ),
        "responses": [
            {
                "model": "potpie",
                "answer": """I'll classify this as a component-mapping question and set up a short investigation plan before tracing the UDD startup/data-layer flow and the UI command path.

Found the main UDD subsystems under `UI/UDD/source`: `UDD.DataLayer` (startup/command routing), `UDD.DataLayer.Features` (feature handlers/executors), `UDD.Communication` (HTTP controllers for UI), plus `UDD.DataManager.*`, `UDD.CacheLayer`, and `UDD.InstrumentProxy.*`. Next I'm tracing the startup entrypoints and one end-to-end command path.

ColGREP was unavailable because `COLGREP_API_BASE_URL` is not configured, so I fell back to direct file reads. Key findings: startup begins in `UDD.DataLayer/Program.cs:24-65`, orchestration happens in `Application.RunAsync()` at `UDD.DataLayer/Application.cs:77-123`, handler startup is controlled by `HandlersController.InitializeHandlersAsync()` at `UDD.DataLayer/HandlersController.cs:82-130`, and UI commands enter via controllers like `UDD.Communication/Controllers/SystemHealthController.cs:136-188`.

The first ripgrep attempt failed because the command used a blocked `<` pattern. The fetched files still showed the important path: controllers call `BaseController.SendRequest()` in `UDD.Communication/Common/BaseController.cs:207-242`, which publishes a `Request<T>` to `ChannelNames.DesktopRequest`; `MessageDistributor.OnDesktopDataRequest()` dispatches by `RequestType` at `UDD.DataLayer/MessageHandler/MessageDistributor.cs:147-202`; executors register their supported `RequestTypes` and are bound in `InitializeExecutors()` at `UDD.DataLayer/MessageHandler/MessageDistributor.cs:76-121`.

Found the missing request enum and bindings: `RequestType` lives in `UDD.CacheLayer.Interfaces/MessageBusQueue/RequestType.cs`, and concrete system-health executors include `AcknowledgeSystemAlertExecutor_New`, `CanShutdownExecutor_New`, `DismissAlertExecutor_New`, and `PreChangeModuleStateExecutor`. I'll read those files plus the request enum to extract the feature-add steps with citations.

## Classification
COMPONENT_MAPPING

## Summary
UDD Data Layer startup is a staged pipeline: `Program.Main()` builds the Autofac container and resolves `IApplication`, `Application.RunAsync()` initializes Redis, messaging, proxies/services, executors, then starts handlers asynchronously and finally notifies the UI that login is ready and later fully ready.

To add a new UI command:
1. Define a new RequestType enum value in `RequestType.cs`.
2. Create a new executor implementing the executor interface that lists this RequestType in its SupportedRequestTypes.
3. Register the executor in `MessageDistributor.InitializeExecutors()` so it gets bound.
4. Add a controller endpoint in `UDD.Communication/Controllers/*` that calls `BaseController.SendRequest<T>()` with the new request type.
5. If the command needs to call instrument hardware, plumb a new method through the relevant `InstrumentProxy`.
6. Add UI-side wiring to call the new HTTP endpoint.
7. Add tests + telemetry.
""",
            },
            {
                "model": "copilot",
                "answer": """The UDD "data layer" is a separate process/service under UDD.DataLayer that boots, connects to platform services, initializes managers/executors/handlers, then signals the UI that it's ready.

1) Entry point + hosting model
File: Program.cs
- Builds the Autofac container via ContainerConfig.Configure() (the registrations live in ContainerConfig and related modules).
- Resolves IApplication (implemented by UDD.DataLayer.Application).
- Runs in one of two modes:
  - Windows service (non-interactive): uses Host.CreateDefaultBuilder(...).UseWindowsService() and a hosted service UDDDataLayerService (so Windows SCM controls start/stop).
  - Console (interactive): calls application.RunAsync() and waits on Console.ReadKey().

2) Main startup pipeline
File: Application.cs, method RunAsync(bool startUp = true)
Startup sequence (high level):
1. InitializeRedis()
   - Reads Redis connection info from IDataLayerConfiguration (RedisServerIp/Port/Password)
   - Calls applicationStore.IntilalizeDataAsync(...)
   - Flushes data store
   - If Redis init fails → throws → UI gets UDDEventType.LoadFailed

(The pattern is generally: register executors with Autofac, then have MessageDistributor resolve them at request time. Implementations vary per feature.)

How to add a new feature that requires a new command from UI:
1. Add an enum entry in RequestType.cs for the new command.
2. Create an Executor class for that RequestType. It typically inherits from a base executor.
3. Register the executor in the DI container so MessageDistributor can resolve it.
4. Add a Controller in UDD.Communication that exposes an HTTP endpoint to the UI.
5. Have the controller call SendRequest<TResponse>(requestType, payload).
6. Update the UI to call the new endpoint when the user triggers the action.
7. Add tests for the executor.
""",
            },
        ],
    },
    {
        "id": "vmm-dynamic-config-template",
        "question": "Explain VMMDynamic Config Template?",
        "responses": [
            {
                "model": "potpie",
                "answer": """## Classification
BROAD_OVERVIEW

## Summary
`VMMDynamic Config Template` is the VMM subsystem that generates runtime VMM configuration objects from a higher-level `TemplateConfig`, using regex-matched templates plus expression evaluation against the actual track/module layout. In practice, it builds stations, queues, operations, traffic lights, covers, and HLC settings dynamically, then writes the generated result to `VMMDynamicModifiedConfig.xml` before normal entity creation continues. `Platform/VMM/Source/VMM.DataModelManager/ObjectRepository.cs:810-849`

## Investigation Process
I searched the VMM code for `VMMDynamic` and `Template`, then read the dynamic configuration pipeline in:
- `Platform/VMM/Source/VMM.DataModelManager/ObjectRepository.cs:810-849`
- `Platform/VMM/Source/VMM.DataModelManager/DynamicConfiguration/DynamicConfigGenerator.cs:90-299`
- `Platform/VMM/Source/VMM.DataModelManager/DynamicConfiguration/TemplateProvider.cs:72-241`
- `Platform/VMM/Source/VMM.DataModelManager/DynamicConfiguration/Templates/*.cs`

ColGREP was unavailable (`COLGREP_API_BASE_URL` not configured), so I used `rg` fallback.

## Detailed Findings

### What it is
The dynamic template system starts when `ObjectRepository.GenerateEntitiesFromDynamicConfiguration()` creates a `DynamicConfigGenerator` and passes in `new TemplateProvider(this._templateConfig)`. It then generates configuration, saves it as `VMMDynamicModifiedConfig.xml`, and uses that generated config to drive normal VMM entity creation.

### How it works
- `TemplateProvider` loads templates by category (Stations, Queues, Operations, TrafficLights, Covers, HLC).
- For each template, the generator pattern-matches against the live track layout via regex (the `Match` field on each template).
- Matched template definitions are expanded with their parameters substituted, then emitted as generated config entries.
- The final XML is written to `VMMDynamicModifiedConfig.xml` and re-read by the normal config consumer.

### Why it exists
The same logical VMM setup must be reusable across very different physical track configurations (PC vs embedded, simulator vs real hardware, different product variants), where the entity counts and names change but the topology pattern doesn't. Without this layer, every variant would need a hand-written full config XML.
""",
            },
            {
                "model": "copilot",
                "answer": """VMMDynamic Config Template is a parameterized configuration "recipe" used by the VMM (Virtual Machine Manager / Virtual Machine Module in this codebase) to generate a final, concrete config at runtime rather than relying on a fully static config file.

In practice, it's typically used when the same logical VMM setup must be reused across multiple targets/environments (PC vs. embedded, simulator vs. real hardware, different product variants), where only a few inputs change.

What it usually contains
A VMMDynamic config template commonly includes:
- Placeholders / tokens (macros) for values that aren't known until deployment/runtime
  - Examples: installation paths, machine names, ports, feature flags, build flavor, etc.
- Default values for settings that don't change often
- Conditional blocks / variant sections (depending on template language)
  - Example: "if Simulator=true then use SIM endpoints, else use HW endpoints"
- Lists of components/modules that VMM should start/control, with parameterized settings
- Expansion rules: how to resolve tokens (from environment variables, registry, command-line args, or an external settings provider)

What it produces
The template is expanded into a resolved "real" config (often a generated file or an in-memory config object), where all placeholders have been replaced and conditionals evaluated. That generated config is what VMM actually uses to:
- locate binaries/plugins,
- know which services/modules to start,
- set up communication endpoints,
- etc.

If you point me to the specific template file (its path in the repo or the file name), I can read it and explain it concretely.
""",
            },
        ],
    },
    {
        "id": "ccsmanager-explain-detail",
        "question": "Explain in detail CCSManager.cs in sample handler solution",
        "responses": [
            {
                "model": "potpie",
                "answer": """## Classification
SPECIFIC_BEHAVIOR

## Summary
`CCSManager` is the Sample Handler's integrated-mode manager for CCS, which in this codebase is the refrigerated Control/Calibrator Storage subsystem. It is not just a thin command wrapper: it owns CCS inventory/cache state, persistence of slot contents, door open/close orchestration, temperature/status reporting, enable/disable behavior, alerts/capabilities, and mediation between higher layers, `CCSMonitor`, and DML/controller messages.
`SampleMgmt/SH/source/SH.DLLImplementation/CCSManager/CCSManager.cs:30-34`.

In integrated mode, `CCSManager` is registered as `ICCSManager` and used by startup, UI, scheduler/request flows, diagnostics, and SIM-facing router paths; in standalone diagnostics mode it is absent and `CCSMonitor` is used directly instead `SampleMgmt/SH/source/SH.ServiceFacade/Components Repository/ComponentsRepository.cs:310-319` and `SampleMgmt/SH/source/SH.ServiceFacade/Components Repository/ComponentsRepository.cs:486-505`.

## Investigation Process
I first located `CCSManager.cs` and its interface with `rg`, because ColGREP was unavailable in this environment. I then read the implementation file in targeted sections: class/interface declaration, constructor, startup/state handling, command methods, SIM router methods, persistence logic, and DML/CCSMonitor event handlers, and cross-checked usage in `ComponentsRepository.cs`, `SHModuleStateStartUp.cs`, and `SampleHandlerUI.cs`.

## Detailed Findings

### 1. Where `CCSManager` sits in the Sample Handler architecture
- Class declaration with interfaces:
  - `public class CCSManager : ICCSManager, ICCSCommand, ICCSManagerEventRegistration, IRefrigeratedStorageRouterCommand`
- Owned state:
  - in-memory cache of CCS slot contents,
  - mapping from physical drawer/slot → product (control or calibrator),
  - door state and temperature snapshot,
  - capability and alert registries.

### 2. Persistence model
- On startup, restores last-known slot state from disk (so a restart doesn't lose what's in CCS).
- Persists deltas as commands succeed.

### 3. Integration with `CCSMonitor` and DML
- Subscribes to `CCSMonitor` events for temperature, door, and slot transitions.
- Translates DML-level controller messages into higher-level CCS state changes.

### 4. SIM router methods
- Implements `IRefrigeratedStorageRouterCommand` so SIM-facing flows can route CCS-specific commands through the manager without coupling to the controller.

### 5. Lifecycle / startup orchestration
- `SHModuleStateStartUp.cs` brings CCSManager up after the controller is initialized but before scheduler accepts requests.

### 6. UI surface
- `SampleHandlerUI.cs` consumes CCSManager for diagnostics views (door open/close, slot status, temperature).
""",
            },
            {
                "model": "copilot",
                "answer": """CCSManager.cs (in CCSManager.cs) is the Sample Handler (SH) component responsible for managing the CCS (cooled/control & calibrator storage) subsystem: tracking drawer/slot contents, persisting/restoring CCS state across restarts, and routing commands/events between SH's business logic and the lower-level controller/DML layer.

Because the file is very large (16k lines), the most useful way to explain it is by responsibility areas (what it owns) and how data/commands flow through it.

1) What CCSManager is (role in the SH solution)
-------------------------------------------------
The class:
 public class CCSManager : ICCSManager,
                           ICCSCommand,
                           ICCSManagerEventRegistration,
                           IRefrigeratedStorageRouterCommand

That combination tells you CCSManager is both:
- a service used by other SH layers (ICCSManager)
- a command endpoint that can be called to perform CCS actions (ICCSCommand, IRefrigeratedStorageRouterCommand)
- an event hub that allows subscribers to register/unregister for CCS-related events (ICCSManagerEventRegistration)

2) Geometry / shape of CCS
--------------------------
- private enum CCSRow { Top, Middle, Bottom }
- private const int NumberOfRowsInCCS = 3
- private const int MaximumNumberOfReAttemptsToCloseCCSCoverToBePerformed = 1
- Each row has multiple slot positions; the manager maintains a 2-D map of slot → contents.

3) Slot state & persistence
---------------------------
- Tracks per-slot type (Control vs Calibrator), product identity, and time-on-storage.
- Persists current snapshot so a restart restores the same picture.

4) Command surface
------------------
Common command-shaped methods (names suggest):
- OpenCCSCover / CloseCCSCover
- LoadProductIntoSlot / RemoveProductFromSlot
- DisableSlot / EnableSlot

5) Tube-diameter logic
----------------------
There's logic that uses tube diameter to select an appropriate slot type before loading.

If you want, I can go deeper on:
- constructor wiring,
- event registration surface,
- integration with the controller/DML layer,
- command-flow details.
""",
            },
        ],
    },
    {
        "id": "two-workorders-shared-sample",
        "question": (
            "In the context of the codebase, how does the workflow handle a scenario where a "
            "single sample is associated with two workorders, and both workorders are received "
            "at nearly the same time? Specifically, what steps does the planner follow to "
            "process these workorders and manage the shared sample within the system?"
        ),
        "responses": [
            {
                "model": "potpie",
                "answer": """## Classification
COMPONENT_MAPPING

## Summary
When two workorders refer to the same sample and arrive almost simultaneously, the planner does **not** allow both to become active at once. It accepts the arrivals in FIFO command order, activates the first workorder that can own the sample, and places the conflicting workorder into the **inactive workorder cache** until the sample is no longer part of an active workorder.
`Platform/Planner/Source/Planner.LifecycleManager/PlannerCommandPriority.config:61-71`, `Platform/Planner/Source/Planner.LifecycleManager/SampleTaskPlanner.cs:795-827`, `Platform/Planner/Source/Planner.LifecycleManager/SampleTaskPlanner.cs:1072-1117`

The shared-sample rule is enforced by cache/query logic that assumes only **one active workorder per sample**, so the second workorder is either treated as an update to the same workorder ID or held inactive and retried later for activation. `Platform/Planner/Source/Planner.InProcSampleCache/IQueryData.cs:49-61`, `Platform/Planner/Source/Planner.StateMachine/LowLevelSTM/PlannerFullPlanningState.cs:528-583`, `Platform/Planner/Source/Planner.LifecycleManager/SampleTaskPlanner.cs:1235-1265`

## Investigation Process
I inspected the planner subsystem under `Platform/Planner/Source`, then traced the workorder intake path from `Planner.ServiceFacade` into `Planner.StateMachine/LowLevelSTM/PlannerFullPlanningState.cs`, and from there into `Planner.LifecycleManager/SampleTaskPlanner.cs`. I used `rg` to find merge/shared-sample references because ColGREP was unavailable (`COLGREP_API_BASE_URL` not configured), then read the lifecycle, inactive-cache, and callback files that implement activation/inactivation rules.

## Detailed Findings

### Step 1 - Intake
Both workorders enter through the planner service facade and are placed onto a FIFO command queue. Priorities are evaluated against `PlannerCommandPriority.config`.

### Step 2 - First workorder activation
`SampleTaskPlanner` tries to activate the first workorder, claims the sample, and marks it as the active workorder for this sample.

### Step 3 - Second workorder
The second workorder cannot acquire the sample because only one active workorder per sample is allowed. The planner places it into the inactive workorder cache with a "waiting for sample" marker.

### Step 4 - Reactivation
When the first workorder completes (or releases the sample), a callback re-checks the inactive cache and promotes the waiting workorder. Sample tracking remains singular.

### Caveat
The exact merge behavior when both workorders carry overlapping test sets is partially conditional and would require tracing the order-merger; the high-level conflict/queue logic above is the authoritative path.
""",
            },
            {
                "model": "copilot",
                "answer": """Within this codebase, the "planner" is not implemented as an in-process component you can step through; it's an external module (Siemens.Trinidad.Planner.ServiceInterface) that the UI/data-layer integrates with via SIM callback notifications and sample/worklist proxies. So the near-simultaneous "two workorders for one sample" case is handled as two distinct workorder entities that both reference the same underlying sample instance, rather than duplicating the sample.

Concretely, the workflow looks like this:

1) Two workorders arrive (almost simultaneously)
------------------------------------------------
- Each workorder is treated independently by the system (worklist/workorder domain).
- The sample is represented by a single sample instance identifier (barcode + UniqueID) and can be referenced alongside different workorder IDs across UI screens and service calls.
- You can see that the UI routinely carries both identifiers together (SampleId + WorkOrderId) when requesting details, e.g. SampleDetailsGetRequest { SampleId = ..., WorkOrderId = ... } in the PrimeDisp ViewModels (hits in NeedsInterventionViewModel.cs, StatSamplesViewModel.cs in the earlier search results).

2) Planner plans/dispatches actions per workorder, but sample tracking stays singular
-------------------------------------------------------------------------------------
When the planner issues/updates commands that involve the physical sample (move/recall/etc.), the integration point in this repo is the SIM callback event stream.

Example: sample recall status is delivered as a RecallSampleStatusEvent and processed in SampleHandlerManager.cs.

3) Sample-level state stays unified across workorders
-----------------------------------------------------
Sample status updates (loaded, on-route, recalled, completed) are routed through SampleListManager.cs and broadcast to subscribers. SampleDetailsManager.cs resolves per-workorder detail rows when the UI needs them.

So at the integration boundary in this repo, the dominant pattern is "two workorders, one shared sample identity, planner external", and detail-resolution happens per-workorder while physical state is broadcast at the sample level.
""",
            },
        ],
    },
]


def get_all_comparisons() -> List[_Comparison]:
    """Return every configured comparison."""
    return list(COMPARISONS)


def get_comparison(comparison_id: str) -> _Comparison | None:
    """Look up a single comparison by id, or None if not present."""
    for entry in COMPARISONS:
        if entry["id"] == comparison_id:
            return entry
    return None


def model_for_position(
    comparison_id: str, position: str, seed: int
) -> str:
    """Decode which model (potpie/copilot) was actually behind position "a"/"b".

    `position` may also be `"tie"` or `"neither"` - those map to themselves so
    downstream storage records the voter's intent rather than a model name.
    """
    if position in ("tie", "neither"):
        return position

    comparison = get_comparison(comparison_id)
    if comparison is None:
        raise ValueError(f"Unknown comparison_id: {comparison_id}")
    order = _shuffled_order(comparison, seed)
    if position == "a":
        return order[0]["model"]
    if position == "b":
        return order[1]["model"]
    raise ValueError(f"Unknown position: {position}")


def shuffled_responses(comparison_id: str, seed: int) -> Dict[str, str]:
    """Return `{"response_a": str, "response_b": str}` for the given seed.

    Same `(comparison_id, seed)` always returns the same ordering. The seed is
    intentionally kept server-side behind an opaque ballot id.
    """
    comparison = get_comparison(comparison_id)
    if comparison is None:
        raise ValueError(f"Unknown comparison_id: {comparison_id}")
    order = _shuffled_order(comparison, seed)
    return {
        "response_a": order[0]["answer"],
        "response_b": order[1]["answer"],
    }


def _shuffled_order(comparison: _Comparison, seed: int) -> List[_Response]:
    """Deterministic A/B order. Even seed = responses[0] first, odd = swapped."""
    responses = comparison["responses"]
    if len(responses) != 2:
        raise ValueError(
            f"Comparison {comparison['id']} must have exactly 2 responses"
        )
    return list(responses) if seed % 2 == 0 else [responses[1], responses[0]]
