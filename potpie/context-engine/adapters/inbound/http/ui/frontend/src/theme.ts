// Neo4j-Browser-ish categorical palette, keyed by entity type (the non-:Entity
// label). Unknown types fall back to a stable hashed hue so every type gets a
// consistent color across renders.
const PALETTE: Record<string, string> = {
  Repository: "#4C8EDA",
  Service: "#57C7E3",
  Person: "#F79767",
  Team: "#FFC454",
  Activity: "#8DCC93",
  Dependency: "#C990C0",
  Environment: "#569480",
  DataStore: "#D9C8AE",
  Cluster: "#A5ABB6",
  APIContract: "#ECB5C9",
  Feature: "#A3A1FB",
  Preference: "#DA7194",
  Policy: "#849Ee2",
  BugPattern: "#F16667",
  Fix: "#8DCC93",
  Decision: "#FFD86E",
  Document: "#6DCE9E",
  Period: "#A5ABB6",
  // soft-fail fallback types — muted so they read as "uncategorized"
  Observation: "#9FB3C8",
  QualityIssue: "#D7A65E",
};

function hashHue(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

export function typeColor(type: string): string {
  return PALETTE[type] || `hsl(${hashHue(type)}, 55%, 62%)`;
}

// The 4 ontology categories (from `graph catalog`). Used for legend grouping
// and show/hide filtering so the explorer stays legible as the graph fills in.
export const CATEGORY_ORDER = ["topology", "people", "timeline", "memory"] as const;
export type Category = (typeof CATEGORY_ORDER)[number];

const TYPE_CATEGORY: Record<string, Category> = {
  Repository: "topology",
  Service: "topology",
  Environment: "topology",
  DataStore: "topology",
  Cluster: "topology",
  Dependency: "topology",
  APIContract: "topology",
  Feature: "topology",
  Team: "people",
  Person: "people",
  Activity: "timeline",
  Period: "timeline",
  Preference: "memory",
  Policy: "memory",
  BugPattern: "memory",
  Fix: "memory",
  Decision: "memory",
  Document: "memory",
  Observation: "memory",
  QualityIssue: "memory",
};

export function typeCategory(type: string): Category {
  return TYPE_CATEGORY[type] || "topology";
}

// Activity `kind` (classified at ingest: feature/fix/chore/security/removal/…),
// used to color timeline dots so the change-history reads at a glance.
const KIND_PALETTE: Record<string, string> = {
  feature: "#4C8EDA",
  fix: "#F16667",
  chore: "#A5ABB6",
  security: "#F79767",
  removal: "#C990C0",
  change: "#8DCC93",
};

export function kindColor(kind: string | undefined): string {
  return (kind && KIND_PALETTE[kind]) || "#8DCC93";
}

export const KIND_ORDER = Object.keys(KIND_PALETTE);
