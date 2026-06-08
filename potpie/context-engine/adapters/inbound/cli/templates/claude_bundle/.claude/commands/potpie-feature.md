Use Potpie context before feature work.

Run `context_resolve` with this recipe:

```json
{"intent":"feature","include":["purpose","feature_map","service_map","docs","tickets","decisions","recent_changes","owners","preferences","source_status"],"mode":"fast","source_policy":"references_only"}
```

Inspect coverage, freshness, quality, fallbacks, open conflicts, and source refs before relying on the result.
