# Spec Regenerate API (Workflows Backend)

The UI calls **regenerate spec** when the user clicks the refresh button on the Specification panel. The recipe is already in `SPEC_READY`; `POST .../spec/generate` only accepts `ANSWERS_SUBMITTED`, so a dedicated regenerate endpoint is required.

## Required endpoint

**`POST /api/v1/recipes/{recipe_id}/spec/regenerate`**

- **When**: User clicks the refresh (regenerate) button on the spec page.
- **Recipe state**: Recipe is in `SPEC_READY` (or similar; spec already exists).
- **Expected behavior**:
  1. Accept the request for recipes in `SPEC_READY` (and any state where a spec exists).
  2. Reset or transition the recipe so spec generation can run again (e.g. set state to allow spec generation, clear cached spec).
  3. Start spec generation using the same pipeline as `POST .../spec/generate` (reuse existing logic).
- **Response**: Same shape as `POST .../spec/generate` (e.g. `TriggerSpecGenerationResponse`) so the frontend can treat it the same and resume polling `GET .../spec`.

## Frontend

- **Service**: `SpecService.regenerateSpec(recipeId)` in `potpie-ui/services/SpecService.ts`.
- **Base URL**: `NEXT_PUBLIC_WORKFLOWS_URL` (e.g. `http://localhost:8002`).
- **On 404**: The UI shows a toast explaining that the workflows backend must implement this endpoint.
