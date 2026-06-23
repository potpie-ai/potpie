export interface GraphNode {
  id: string;
  key: string;
  labels: string[];
  type: string;
  caption: string;
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

export interface PotRef {
  id: string;
  name: string;
  active?: boolean;
  source_count?: number;
  counts?: Record<string, number>;
}

export interface PotsResponse {
  pots: PotRef[];
  active: { id: string; name: string } | null;
}

export interface StatusResponse {
  pot_id: string;
  backend_profile: string;
  backend_ready: boolean;
  counts: Record<string, number>;
}

export interface SearchEntity {
  key: string;
  labels: string[];
  score: number;
}
