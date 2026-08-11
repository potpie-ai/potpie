export interface GraphNode {
  id: string;
  key: string;
  labels: string[];
  type: string;
  caption: string;
  summary?: string;
  properties: Record<string, unknown>;
  // runtime-only (added client-side for layout/sizing)
  degree?: number;
  x?: number;
  y?: number;
}

export interface GraphEdge {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  predicate: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated?: boolean;
}

/** Which context-engine a pot lives on. Pot ids are unique per host, not
 *  globally, so nothing may be keyed on `id` alone. */
export type Origin = "local" | "managed";

export interface PotRef {
  id: string;
  name: string;
  origin: Origin;
  active?: boolean;
  source_count?: number;
  counts?: Record<string, number>;
}

export interface PotsResponse {
  pots: PotRef[];
  active: { id: string; name: string; origin: Origin } | null;
  active_origin: Origin;
  /** origin -> why it could not be listed; the other host's pots still load. */
  unavailable?: Record<string, string>;
  /** False when a remote host had too many pots to count them all cheaply;
   *  those pots arrive without `counts`, which is not the same as zero. */
  counts_complete?: boolean;
}

export interface StatusResponse {
  pot_id: string;
  origin?: Origin;
  backend_profile: string;
  backend_ready: boolean;
  counts: Record<string, number>;
}

export interface SearchEntity {
  key: string;
  labels: string[];
  score: number;
}
