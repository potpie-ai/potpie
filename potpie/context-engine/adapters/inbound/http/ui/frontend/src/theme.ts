// Potpie brand (potpie.ai): green terminal aesthetics — lime accent, cream
// text, coral alerts. Canvas/SVG painters import these so the views stay in
// lockstep with the CSS variables in styles.css.
// The chrome (topbar/sidebar) is a near-black neutral chassis; the graph and
// timeline render on a dark neutral "screen" inside it, matching the side panels.
export const UI = {
  bg: "#151916",
  panel2: "#222822",
  accent: "#b6e343",
  glow: "rgba(182,227,67,0.9)",
  ring: "#efe6df",
  ringSoft: "rgba(239,230,223,0.5)",
  link: "rgba(138,167,155,0.32)",
  label: "rgba(239,230,223,0.88)",
  labelBright: "#fff9f5",
  halo: "rgba(21,25,22,0.85)",
  haloStrong: "rgba(21,25,22,0.95)",
  iconStroke: "rgba(10,12,11,0.85)",
  textMuted: "#8aa79b",
  gridDot: "rgba(182,227,67,0.07)",
  font: 'SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace',
} as const;

// Categorical palette, keyed by entity type (the non-:Entity label) and tuned
// toward the brand hues. Unknown types fall back to a stable hashed hue so
// every type gets a consistent color across renders.
const PALETTE: Record<string, string> = {
  Repository: "#B6E343",
  Service: "#45C7A8",
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
  BugPattern: "#F15B5B",
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

export function typesInCategory(category: Category): string[] {
  return Object.entries(TYPE_CATEGORY)
    .filter(([, cat]) => cat === category)
    .map(([type]) => type);
}

// Activity `kind` (classified at ingest: feature/fix/chore/security/removal/…),
// used to color timeline dots so the change-history reads at a glance.
const KIND_PALETTE: Record<string, string> = {
  feature: "#B6E343",
  fix: "#F15B5B",
  chore: "#8FA396",
  security: "#F79767",
  removal: "#C990C0",
  change: "#8DCC93",
};

export function kindColor(kind: string | undefined): string {
  return (kind && KIND_PALETTE[kind]) || "#8DCC93";
}

export const KIND_ORDER = Object.keys(KIND_PALETTE);
