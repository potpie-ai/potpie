import { diffGraph } from "./commitDiff.ts";
import type { CommitDetail } from "./commitTypes";
import type { GraphData, GraphEdge, GraphNode } from "./types";

export const endpointId = (endpoint: GraphEdge["source"]) =>
  typeof endpoint === "string" ? endpoint : endpoint.id;

export function mergeContext(left: GraphData, right: GraphData): GraphData {
  return {
    nodes: [
      ...new Map(
        [...left.nodes, ...right.nodes].map((node) => [node.id, node]),
      ).values(),
    ],
    edges: [
      ...new Map(
        [...left.edges, ...right.edges].map((edge) => [edge.id, edge]),
      ).values(),
    ],
    truncated: left.truncated || right.truncated,
  };
}

/** Keep receipt incarnation IDs; only attach live connections to a matching incarnation. */
export function mutationInContext(
  detail: CommitDetail,
  existing: GraphData,
  expanded: string[] = [],
  contextLimit = 120,
): GraphData {
  const mutation = diffGraph(detail);
  const recordedById = new Map(mutation.nodes.map((node) => [node.id, node]));
  const recordedByKey = new Map<string, GraphNode[]>();
  for (const node of mutation.nodes)
    recordedByKey.set(node.key, [...(recordedByKey.get(node.key) || []), node]);
  const aliases = new Map<string, string>();
  const liveNodes = new Map<string, GraphNode>();
  for (const node of existing.nodes) {
    const recordId = node.properties.record_id;
    const candidates = recordedByKey.get(node.key) || [];
    const match =
      typeof recordId === "string"
        ? recordedById.get(recordId)
        : candidates.length === 1
          ? candidates[0]
          : undefined;
    const id = match?.id || `context:${node.id}`;
    aliases.set(node.id, id);
    const name = node.properties.name;
    liveNodes.set(id, {
      ...node,
      id,
      caption: typeof name === "string" && name ? name : node.caption,
      diff_status: "context",
      ghost: false,
    });
  }
  const nodes = new Map<string, GraphNode>(
    mutation.nodes.map((node) => [
      node.id,
      {
        ...liveNodes.get(node.id),
        ...node,
        type: liveNodes.get(node.id)?.type || node.type,
        diff_status:
          node.diff_status === "recorded context"
            ? "context"
            : node.diff_status,
      },
    ]),
  );
  const anchors = new Set([...nodes.keys(), ...expanded]);
  const changedIds = new Set(detail.changes.map((change) => change.record_id));
  const liveEdges = existing.edges.flatMap((edge) => {
    const source = aliases.get(endpointId(edge.source));
    const target = aliases.get(endpointId(edge.target));
    if (!source || !target || changedIds.has(edge.record_id || edge.id))
      return [];
    return [
      {
        ...edge,
        id: `context:${edge.id}`,
        record_id: undefined,
        source,
        target,
        diff_status: "context",
      },
    ];
  });
  const expansion = new Set(expanded);
  liveEdges.sort(
    (a, b) =>
      Number(expansion.has(b.source) || expansion.has(b.target)) -
      Number(expansion.has(a.source) || expansion.has(a.target)),
  );
  let contextCount = 0;
  let limited = Boolean(existing.truncated);
  // One-hop context explains where the mutation lands without pulling in the entire pot.
  // Expanding a selected existing node adds another local neighborhood.
  for (const edge of liveEdges) {
    if (!anchors.has(edge.source) && !anchors.has(edge.target)) continue;
    const missing = [edge.source, edge.target].filter((id) => !nodes.has(id));
    if (contextCount + missing.length > contextLimit) {
      limited = true;
      continue;
    }
    for (const id of missing) {
      const node = liveNodes.get(id);
      if (node) {
        nodes.set(id, node);
        contextCount++;
      }
    }
  }
  const edges: GraphEdge[] = [
    ...liveEdges.filter(
      (edge) => nodes.has(edge.source) && nodes.has(edge.target),
    ),
    ...mutation.edges,
  ];
  return { nodes: [...nodes.values()], edges, truncated: limited };
}
